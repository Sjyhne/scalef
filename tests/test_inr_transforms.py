"""Equivalence tests for the vectorized affine / color transforms in INRBase."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from models.inr import INRBase
from optimize import build_model


def _module(num_frames: int = 6) -> INRBase:
    torch.manual_seed(0)
    module = INRBase(None, torch.nn.Identity(), num_samples=num_frames)
    with torch.no_grad():
        for sid in range(1, num_frames):
            module.affine_params[sid].copy_(
                torch.tensor([[1.02, 0.03, -0.01, -0.04, 0.98, 0.02]])
            )
            for ch in range(3):
                module.color_transforms[sid][ch].weight.fill_(1.0 + 0.05 * sid)
                module.color_transforms[sid][ch].bias.fill_(0.01 * ch - 0.02)
    return module


def _coords(b: int, h: int = 7, w: int = 5) -> torch.Tensor:
    torch.manual_seed(1)
    return torch.rand(b, h, w, 2)


def test_apply_affine_matches_matmul_forward():
    module = _module()
    for ids in ([0], [3], [2, 5], [0, 4]):
        sid = torch.tensor(ids)
        coords = _coords(len(ids))
        A = module.get_direct_affine(sid)
        fast = module.apply_affine(coords, A)
        ref = module.apply_affine_matmul(coords, A)
        assert fast.shape == ref.shape
        torch.testing.assert_close(fast, ref, rtol=1e-6, atol=1e-6)


def test_apply_affine_matches_matmul_gradients():
    module = _module()
    sid = torch.tensor([2, 5])

    grads = []
    for fn in (module.apply_affine, module.apply_affine_matmul):
        module.zero_grad(set_to_none=True)
        coords = _coords(2).requires_grad_(True)
        A = module.get_direct_affine(sid)
        fn(coords, A).square().sum().backward()
        grads.append(
            (
                coords.grad.clone(),
                module.affine_params[2].grad.clone(),
                module.affine_params[5].grad.clone(),
            )
        )

    for fast, ref in zip(grads[0], grads[1]):
        torch.testing.assert_close(fast, ref, rtol=1e-5, atol=1e-6)


def test_apply_color_transform_matches_loop():
    module = _module()
    for ids in ([0], [3], [1, 4], [0, 2]):
        sid = torch.tensor(ids)
        x = torch.rand(len(ids), 6, 4, 3)
        torch.testing.assert_close(
            module.apply_color_transform(x, sid),
            module.apply_color_transform_loop(x, sid),
            rtol=1e-6,
            atol=1e-6,
        )


def test_color_transform_leaves_frame_zero_without_grad():
    module = _module()
    sid = torch.tensor([0, 3])
    x = torch.rand(2, 6, 4, 3, requires_grad=True)
    module.apply_color_transform(x, sid).sum().backward()

    for ch in range(3):
        assert module.color_transforms[0][ch].weight.grad is None
        assert module.color_transforms[3][ch].weight.grad is not None


def test_control_flags_freeze_alignment_and_radiometry():
    args = SimpleNamespace(
        num_samples=3,
        use_gnll=False,
        use_laplace_nll=False,
        hetero_scale="pixel",
        hetero_region_size=4,
        lr_degradation="s2_psf_m",
        freeze_affines=True,
        freeze_radiometry=True,
    )
    model = build_model(args, None, torch.nn.Identity(), torch.device("cpu"))
    assert not any(parameter.requires_grad for parameter in model.affine_params.parameters())
    assert not any(parameter.requires_grad for parameter in model.color_transforms.parameters())
