"""Fixed-location HR patch metrics for comparing runs at different LR sizes."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torchmetrics.functional.image import peak_signal_noise_ratio
from torchmetrics.functional.image import structural_similarity_index_measure as ssim

DEFAULT_SPOT_HR_PX = 32


def center_window_slices(height: int, width: int, size: int) -> tuple[int, int, int, int]:
    """Return ``(y0, y1, x0, x1)`` for a ``size×size`` window at the image center."""
    h, w = int(height), int(width)
    size = int(min(max(1, size), h, w))
    cy, cx = h // 2, w // 2
    half = size // 2
    y0 = max(0, cy - half)
    x0 = max(0, cx - half)
    y1 = min(h, y0 + size)
    x1 = min(w, x0 + size)
    y0 = max(0, y1 - size)
    x0 = max(0, x1 - size)
    return y0, y1, x0, x1


def crop_bchw(tensor: torch.Tensor, slices: tuple[int, int, int, int]) -> torch.Tensor:
    y0, y1, x0, x1 = slices
    return tensor[:, :, y0:y1, x0:x1]


def compute_fixed_spot_metrics(
    pred_bchw: torch.Tensor,
    gt_bchw: torch.Tensor,
    bilinear_bchw: torch.Tensor,
    *,
    spot_hr_px: int,
    device: torch.device,
    lpips_fn,
    location: str = "center",
) -> dict[str, Any]:
    """PSNR / SSIM / LPIPS on the same HR window for all LR-size runs."""
    if int(spot_hr_px) <= 0:
        return {}

    _, _, h, w = gt_bchw.shape
    slices = center_window_slices(h, w, int(spot_hr_px))
    pred = crop_bchw(pred_bchw, slices).cpu()
    gt = crop_bchw(gt_bchw, slices).cpu()
    bil = crop_bchw(bilinear_bchw, slices).cpu()

    model_psnr = peak_signal_noise_ratio(pred, gt, data_range=1.0).item()
    bilinear_psnr = peak_signal_noise_ratio(bil, gt, data_range=1.0).item()
    model_ssim = ssim(pred, gt, data_range=1.0).item()
    bilinear_ssim = ssim(bil, gt, data_range=1.0).item()

    pred_dev = (pred * 2 - 1).to(device)
    gt_dev = (gt * 2 - 1).to(device)
    bil_dev = (bil * 2 - 1).to(device)
    model_lpips = float(lpips_fn(pred_dev, gt_dev).item())
    bilinear_lpips = float(lpips_fn(bil_dev, gt_dev).item())

    y0, y1, x0, x1 = slices
    return {
        "hr_pixels": int(y1 - y0),
        "location": str(location),
        "slices_hr": [int(y0), int(y1), int(x0), int(x1)],
        "model_psnr": model_psnr,
        "bilinear_psnr": bilinear_psnr,
        "psnr_improvement": model_psnr - bilinear_psnr,
        "model_ssim": model_ssim,
        "bilinear_ssim": bilinear_ssim,
        "ssim_improvement": model_ssim - bilinear_ssim,
        "model_lpips": model_lpips,
        "bilinear_lpips": bilinear_lpips,
        "lpips_improvement": bilinear_lpips - model_lpips,
        "model_mse": F.mse_loss(pred, gt).item(),
        "bilinear_mse": F.mse_loss(bil, gt).item(),
        "mse_improvement": F.mse_loss(bil, gt).item() - F.mse_loss(pred, gt).item(),
        "model_mae": F.l1_loss(pred, gt).item(),
        "bilinear_mae": F.l1_loss(bil, gt).item(),
        "mae_improvement": F.l1_loss(bil, gt).item() - F.l1_loss(pred, gt).item(),
    }
