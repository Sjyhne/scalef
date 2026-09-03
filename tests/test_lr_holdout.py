"""Unit tests for LR pixel-block holdout helpers."""

from __future__ import annotations

import torch

from eval.lr_holdout import (
    EarlyStopState,
    build_frame_masks,
    build_holdout_mask,
    elementwise_recon,
    gather_train_masks,
    holdout_block_origins,
    masked_mean,
    resolve_early_stop_score,
    resolve_holdout_block,
    resolve_holdout_patch_batch,
)


def test_holdout_mask_frac_and_shape():
    m = build_holdout_mask(64, 64, block=8, frac=0.25, seed=0)
    assert m.shape == (1, 64, 64, 1)
    assert m.dtype == torch.bool
    held = float((~m).float().mean())
    assert 0.1 < held < 0.4


def test_frame_masks_differ_by_seed():
    masks = build_frame_masks(3, 32, 32, block=8, frac=0.2)
    assert len(masks) == 3
    assert not torch.equal(masks[0], masks[1])


def test_masked_mean_matches_selected_average():
    pred = torch.arange(12, dtype=torch.float32).view(1, 2, 2, 3)
    target = torch.zeros_like(pred)
    elem = (pred - target) ** 2
    mask = torch.tensor([[[[1], [0]], [[1], [0]]]], dtype=torch.bool)
    got = masked_mean(elem, mask)
    # Selected pixels (0,0) and (1,0): values [0,1,2] and [6,7,8]
    expected = ((0 + 1 + 4) + (36 + 49 + 64)) / 2 / 3
    assert abs(float(got) - expected) < 1e-5


def test_gather_train_masks_batches():
    masks = build_frame_masks(4, 16, 16, block=4, frac=0.1)
    sid = torch.tensor([0, 2])
    batched = gather_train_masks(masks, sid, torch.device("cpu"))
    assert batched.shape == (2, 16, 16, 1)
    assert torch.equal(batched[0:1], masks[0])
    assert torch.equal(batched[1:2], masks[2])


def test_resolve_early_stop_score_mae():
    score = resolve_early_stop_score("mae", holdout_mse=0.5, hr_metrics={"model_mae": 0.017})
    assert score == ("mae", 0.017)


def test_early_stop_patience_and_restore():
    model = torch.nn.Linear(2, 2)
    state = EarlyStopState(
        train_masks=[build_holdout_mask(8, 8, 4, 0.25, seed=0)],
        val_ids=[0],
        patience=2,
        min_iters=10,
    )
    assert not state.observe(5, 1.0, model)
    assert state.best_state is None
    assert not state.observe(15, 1.1, model)  # first eligible check becomes best
    assert state.best_iter == 15
    assert not state.observe(20, 1.2, model)  # no improve, check 1
    assert state.observe(25, 1.3, model)  # no improve, check 2 -> stop
    assert state.stopped
    assert state.stopped_iter == 25
    # Corrupt weights then restore
    with torch.no_grad():
        model.weight.fill_(99.0)
    assert state.restore_best(model)
    assert float(model.weight.detach().abs().max()) < 50.0


def test_early_stop_regression_guard():
    model = torch.nn.Linear(2, 2)
    state = EarlyStopState(
        train_masks=[build_holdout_mask(8, 8, 4, 0.25, seed=0)],
        val_ids=[0],
        patience=5,
        min_iters=100,
        metric="lpips",
        max_regression=0.01,
    )
    assert not state.observe(100, 0.40, model)
    assert state.best_iter == 100
    # Clear regression beyond tolerance -> immediate patience exhaustion
    assert state.observe(200, 0.42, model)
    assert state.stopped
    assert state.stopped_iter == 200


def test_holdout_block_origins_only_held_blocks():
    mask = torch.ones(1, 16, 16, 1, dtype=torch.bool)
    mask[:, 8:16, 0:8, :] = False
    origins = holdout_block_origins(mask, block=8)
    assert origins == [(8, 0, 8, 8)]


def test_holdout_block_origins_none_when_all_train():
    mask = torch.ones(1, 16, 16, 1, dtype=torch.bool)
    assert holdout_block_origins(mask, block=8) == []


def test_elementwise_recon_mse():
    p = torch.ones(1, 2, 2, 3)
    t = torch.zeros(1, 2, 2, 3)
    e = elementwise_recon("mse", p, t)
    assert torch.allclose(e, torch.ones_like(e))


def test_holdout_val_forwards_only_holdout_blocks():
    from eval.lr_holdout import compute_holdout_val_loss

    class _DS:
        lr_height = 16
        lr_width = 16

        def get_hr_coordinates(self):
            return torch.zeros(64, 64, 2)

        def get_lr_sample_hwc(self, idx):
            return torch.zeros(16, 16, 3)

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.shapes = []

        def forward(self, coords, sample_idx=None, lr_frames=None, **kwargs):
            self.shapes.append(tuple(coords.shape))
            return lr_frames, None

    mask = torch.ones(1, 16, 16, 1, dtype=torch.bool)
    mask[:, 0:8, 0:8, :] = False
    state = EarlyStopState(
        train_masks=[mask], val_ids=[0], patience=0, min_iters=0, holdout_block=8
    )
    model = _Model()
    args = type(
        "A",
        (),
        {"recon_loss": "mae", "holdout_block": 8, "holdout_patch_batch": 64, "holdout_psf_pad_lr": 0},
    )()
    loss = compute_holdout_val_loss(model, _DS(), state, args, torch.device("cpu"))
    assert model.shapes == [(1, 32, 32, 2)]
    assert loss == 0.0


def test_resolve_holdout_block_auto_scales_with_lr():
    assert resolve_holdout_block(512, 512, 0) == 8
    assert resolve_holdout_block(2048, 2048, 0) == 32
    assert resolve_holdout_block(1024, 1024, 0) == 16
    assert resolve_holdout_block(2048, 2048, 8) == 8  # explicit wins


def test_resolve_holdout_patch_batch_auto():
    assert resolve_holdout_patch_batch(8, 0) == 64
    assert resolve_holdout_patch_batch(32, 0) == 64  # 4096 / 1024
    assert resolve_holdout_patch_batch(8, 128) == 128

    from models.s2_psf_forward import degrade_hr_bchw
    from eval.lr_holdout import holdout_psf_pad_lr

    torch.manual_seed(0)
    df = 4
    pad = holdout_psf_pad_lr(df)
    hr = torch.rand(1, 3, 128, 128)
    lr_full = degrade_hr_bchw(hr, df, "s2_psf_m")
    r, c, bh, bw = 8, 8, 8, 8
    hr_crop = hr[:, :, (r - pad) * df : (r + bh + pad) * df, (c - pad) * df : (c + bw + pad) * df]
    lr_crop = degrade_hr_bchw(hr_crop, df, "s2_psf_m")
    inner = lr_crop[:, :, pad : pad + bh, pad : pad + bw]
    assert torch.allclose(inner, lr_full[:, :, r : r + bh, c : c + bw], atol=1e-5, rtol=1e-5)
