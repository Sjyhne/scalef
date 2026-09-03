"""Gradient accumulation over fused tile batches."""

from __future__ import annotations

import argparse

import torch

from optimize import _split_tile_batch, resolve_grad_accum_groups


def _args(**kw):
    return argparse.Namespace(**kw)


def test_groups_default_to_one():
    assert resolve_grad_accum_groups(_args(grad_accum=1), 16) == 1
    assert resolve_grad_accum_groups(_args(grad_accum=0), 16) == 1
    assert resolve_grad_accum_groups(_args(), 16) == 1


def test_groups_never_exceed_batch():
    assert resolve_grad_accum_groups(_args(grad_accum=8), 4) == 4
    assert resolve_grad_accum_groups(_args(grad_accum=8), 1) == 1


def test_groups_divide_the_batch_evenly():
    # 16 tiles in 4 groups of 4.
    assert resolve_grad_accum_groups(_args(grad_accum=4), 16) == 4
    # 6 tiles cannot split 4 ways evenly, so fall back to the largest divisor.
    assert resolve_grad_accum_groups(_args(grad_accum=4), 6) == 3


def test_split_covers_the_batch_exactly_once():
    coords = torch.arange(8 * 2 * 2 * 2, dtype=torch.float32).reshape(8, 2, 2, 2)
    target = torch.arange(8 * 2 * 2 * 3, dtype=torch.float32).reshape(8, 2, 2, 3)
    sid = torch.arange(8)
    mask = torch.ones(8, 2, 2)
    dx = torch.arange(8, dtype=torch.float32)
    dy = torch.arange(8, dtype=torch.float32)

    chunks = list(_split_tile_batch(4, coords, target, sid, mask, dx, dy))
    assert len(chunks) == 4
    assert all(c[0].shape[0] == 2 for c in chunks)
    torch.testing.assert_close(torch.cat([c[0] for c in chunks]), coords)
    torch.testing.assert_close(torch.cat([c[1] for c in chunks]), target)
    torch.testing.assert_close(torch.cat([c[2] for c in chunks]), sid)
    torch.testing.assert_close(torch.cat([c[3] for c in chunks]), mask)


def test_split_passes_through_missing_mask():
    coords = torch.zeros(4, 1, 1, 2)
    chunks = list(_split_tile_batch(
        2, coords, torch.zeros(4, 1, 1, 3), torch.zeros(4, dtype=torch.long),
        None, torch.zeros(4), torch.zeros(4),
    ))
    assert len(chunks) == 2
    assert all(c[3] is None for c in chunks)


def test_single_group_yields_the_untouched_batch():
    coords = torch.zeros(3, 1, 1, 2)
    mask = torch.ones(3, 1, 1)
    chunks = list(_split_tile_batch(
        1, coords, torch.zeros(3, 1, 1, 3), torch.zeros(3, dtype=torch.long),
        mask, torch.zeros(3), torch.zeros(3),
    ))
    assert len(chunks) == 1
    assert chunks[0][0] is coords
    assert chunks[0][3] is mask


def test_accumulated_gradient_matches_full_batch():
    """Averaging per-group means must reproduce the full-batch gradient."""
    torch.manual_seed(0)
    layer = torch.nn.Linear(4, 3)
    x = torch.randn(8, 4)
    y = torch.randn(8, 3)

    layer.zero_grad(set_to_none=True)
    torch.nn.functional.mse_loss(layer(x), y).backward()
    full = layer.weight.grad.clone()

    groups = 4
    per = x.shape[0] // groups
    layer.zero_grad(set_to_none=True)
    for start in range(0, x.shape[0], per):
        sl = slice(start, start + per)
        (torch.nn.functional.mse_loss(layer(x[sl]), y[sl]) / groups).backward()

    torch.testing.assert_close(layer.weight.grad, full, rtol=1e-5, atol=1e-7)
