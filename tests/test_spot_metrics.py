"""Tests for fixed-spot SR evaluation metrics."""

from __future__ import annotations

import torch

from eval.spot_metrics import center_window_slices, compute_fixed_spot_metrics


class _DummyLPIPS:
    def __call__(self, pred, gt):
        return torch.tensor(0.1, device=pred.device)


def test_center_window_slices_even_size():
    y0, y1, x0, x1 = center_window_slices(256, 512, 32)
    assert y1 - y0 == 32
    assert x1 - x0 == 32
    assert y0 == 256 // 2 - 16
    assert x0 == 512 // 2 - 16


def test_fixed_spot_metrics_perfect_prediction():
    gt = torch.rand(1, 3, 128, 128)
    pred = gt.clone()
    bil = gt.clone()
    out = compute_fixed_spot_metrics(
        pred,
        gt,
        bil,
        spot_hr_px=32,
        device=torch.device("cpu"),
        lpips_fn=_DummyLPIPS(),
    )
    assert out["hr_pixels"] == 32
    assert out["model_psnr"] > 80.0
    assert out["model_ssim"] > 0.99
    assert out["bilinear_ssim"] > 0.99
