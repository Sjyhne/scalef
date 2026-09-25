"""Masked HR evaluation for partial NIB coverage within a fixed crop."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics.functional.image import peak_signal_noise_ratio
from torchmetrics.functional.image import structural_similarity_index_measure as ssim


def mask_bbox_slices(mask_hw: torch.Tensor | np.ndarray) -> tuple[int, int, int, int]:
    """Return ``(y0, y1, x0, x1)`` bounding box of True pixels."""
    if isinstance(mask_hw, torch.Tensor):
        mask = mask_hw.detach().cpu().numpy()
    else:
        mask = np.asarray(mask_hw, dtype=bool)
    if not np.any(mask):
        raise ValueError("eval mask is empty")
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y0, y1 = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1]) + 1
    x0, x1 = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1]) + 1
    return y0, y1, x0, x1


def masked_mse(pred_bchw: torch.Tensor, gt_bchw: torch.Tensor, mask_hw: torch.Tensor) -> float:
    mask = mask_hw.to(pred_bchw.device).unsqueeze(0).unsqueeze(0).to(pred_bchw.dtype)
    diff2 = (pred_bchw - gt_bchw) ** 2
    return float((diff2 * mask).sum() / mask.sum().clamp(min=1.0) / pred_bchw.shape[1])


def masked_mae(pred_bchw: torch.Tensor, gt_bchw: torch.Tensor, mask_hw: torch.Tensor) -> float:
    mask = mask_hw.to(pred_bchw.device).unsqueeze(0).unsqueeze(0).to(pred_bchw.dtype)
    diff = (pred_bchw - gt_bchw).abs()
    return float((diff * mask).sum() / mask.sum().clamp(min=1.0) / pred_bchw.shape[1])


def masked_psnr(pred_bchw: torch.Tensor, gt_bchw: torch.Tensor, mask_hw: torch.Tensor) -> float:
    mse = masked_mse(pred_bchw, gt_bchw, mask_hw)
    if mse <= 0:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def _fill_invalid(pred_bchw: torch.Tensor, gt_bchw: torch.Tensor, mask_hw: torch.Tensor) -> torch.Tensor:
    return torch.where(mask_hw.to(pred_bchw.device).bool()[None, None], pred_bchw, gt_bchw)


def masked_lpips(lpips_fn, pred_bchw: torch.Tensor, gt_bchw: torch.Tensor, mask_hw: torch.Tensor) -> float:
    """LPIPS averaged over feature cells whose footprint lies entirely inside ``mask_hw``.

    Invalid prediction pixels are replaced by the reference so both inputs agree there; each
    layer's distance map is then averaged over fully valid cells only. With an all-valid mask
    this equals ``lpips_fn(pred, gt)``.
    """
    mask = mask_hw.to(pred_bchw.device).float()[None, None]
    pred = _fill_invalid(pred_bchw, gt_bchw, mask_hw)
    in0, in1 = pred * 2 - 1, gt_bchw * 2 - 1
    if getattr(lpips_fn, "version", "0.1") == "0.1":
        in0, in1 = lpips_fn.scaling_layer(in0), lpips_fn.scaling_layer(in1)
    outs0, outs1 = lpips_fn.net.forward(in0), lpips_fn.net.forward(in1)
    total = 0.0
    for kk in range(lpips_fn.L):
        f0 = outs0[kk] / (torch.sqrt((outs0[kk] ** 2).sum(dim=1, keepdim=True)) + 1e-10)
        f1 = outs1[kk] / (torch.sqrt((outs1[kk] ** 2).sum(dim=1, keepdim=True)) + 1e-10)
        dist = lpips_fn.lins[kk]((f0 - f1) ** 2)
        weight = (F.interpolate(mask, size=dist.shape[-2:], mode="area") > 0.999).to(dist.dtype)
        if float(weight.sum()) == 0.0:
            raise ValueError(f"no fully valid LPIPS cells at layer {kk}")
        total += float((dist * weight).sum() / weight.sum())
    return total


def masked_ssim(pred_bchw: torch.Tensor, gt_bchw: torch.Tensor, mask_hw: torch.Tensor,
                *, kernel_size: int = 11) -> float:
    """SSIM averaged over pixels whose ``kernel_size`` window lies entirely inside ``mask_hw``.

    Pixels outside the image are treated as valid, so an all-valid mask reproduces the
    unmasked SSIM exactly.
    """
    pred = _fill_invalid(pred_bchw, gt_bchw, mask_hw)
    _, ssim_map = ssim(pred, gt_bchw, data_range=1.0, kernel_size=kernel_size, return_full_image=True)
    invalid = (~mask_hw.bool()).float().to(ssim_map.device)[None, None]
    r = kernel_size // 2
    touched = F.max_pool2d(F.pad(invalid, (r, r, r, r), value=0.0), kernel_size, stride=1)[0, 0] > 0
    keep = ~touched
    if not bool(keep.any()):
        raise ValueError("no fully valid SSIM windows")
    return float(ssim_map.mean(dim=1)[0][keep].mean())


def _center_crop_bchw(
    *tensors: torch.Tensor, max_side: int
) -> tuple[torch.Tensor, ...]:
    """Center-crop BCHW tensors so max(H, W) ≤ ``max_side`` (shared window)."""
    max_side = int(max_side)
    h, w = int(tensors[0].shape[-2]), int(tensors[0].shape[-1])
    if max(h, w) <= max_side:
        return tensors
    th, tw = min(h, max_side), min(w, max_side)
    y0 = max(0, (h - th) // 2)
    x0 = max(0, (w - tw) // 2)
    return tuple(t[:, :, y0 : y0 + th, x0 : x0 + tw] for t in tensors)


def compute_masked_image_metrics(
    pred_bchw: torch.Tensor,
    gt_bchw: torch.Tensor,
    bilinear_bchw: torch.Tensor,
    mask_hw: torch.Tensor,
    *,
    device: torch.device,
    lpips_fn,
    perceptual_max_side: int = 2048,
) -> dict[str, Any]:
    """Full-frame metrics restricted to ``mask_hw`` (H, W bool).

    PSNR/MAE/MSE use the full mask. SSIM/LPIPS are computed on the mask bbox, center-cropped
    to ``perceptual_max_side`` when the bbox is larger (avoids OOM on huge AOIs), and averaged
    only over windows/feature cells that lie entirely inside the mask.
    """
    mask_hw = mask_hw.to(device)
    valid_fraction = float(mask_hw.float().mean().item())

    model_psnr = masked_psnr(pred_bchw, gt_bchw, mask_hw)
    bilinear_psnr = masked_psnr(bilinear_bchw, gt_bchw, mask_hw)
    model_mse = masked_mse(pred_bchw, gt_bchw, mask_hw)
    bilinear_mse = masked_mse(bilinear_bchw, gt_bchw, mask_hw)
    model_mae = masked_mae(pred_bchw, gt_bchw, mask_hw)
    bilinear_mae = masked_mae(bilinear_bchw, gt_bchw, mask_hw)

    y0, y1, x0, x1 = mask_bbox_slices(mask_hw)
    pred_crop = pred_bchw[:, :, y0:y1, x0:x1]
    gt_crop = gt_bchw[:, :, y0:y1, x0:x1]
    bil_crop = bilinear_bchw[:, :, y0:y1, x0:x1]
    mask_crop = mask_hw[y0:y1, x0:x1][None, None]
    pred_crop, gt_crop, bil_crop, mask_crop = _center_crop_bchw(
        pred_crop, gt_crop, bil_crop, mask_crop, max_side=perceptual_max_side
    )
    mask_crop = mask_crop[0, 0]

    with torch.no_grad():
        model_ssim = masked_ssim(pred_crop, gt_crop, mask_crop)
        bilinear_ssim = masked_ssim(bil_crop, gt_crop, mask_crop)
        model_lpips = masked_lpips(lpips_fn, pred_crop, gt_crop, mask_crop)
        bilinear_lpips = masked_lpips(lpips_fn, bil_crop, gt_crop, mask_crop)

    return {
        "masked": True,
        "valid_fraction": valid_fraction,
        "bbox_slices_hr": [y0, y1, x0, x1],
        "perceptual_hw": [int(pred_crop.shape[-2]), int(pred_crop.shape[-1])],
        "model_psnr": model_psnr,
        "bilinear_psnr": bilinear_psnr,
        "psnr_improvement": model_psnr - bilinear_psnr,
        "model_ssim": model_ssim,
        "bilinear_ssim": bilinear_ssim,
        "ssim_improvement": model_ssim - bilinear_ssim,
        "model_lpips": model_lpips,
        "bilinear_lpips": bilinear_lpips,
        "lpips_improvement": bilinear_lpips - model_lpips,
        "model_mse": model_mse,
        "bilinear_mse": bilinear_mse,
        "mse_improvement": bilinear_mse - model_mse,
        "model_mae": model_mae,
        "bilinear_mae": bilinear_mae,
        "mae_improvement": bilinear_mae - model_mae,
        "test_loss": model_mse,
        "test_psnr": model_psnr,
    }


def spot_fully_inside_mask(
    mask_hw: torch.Tensor | np.ndarray,
    slices: tuple[int, int, int, int],
    *,
    min_valid_frac: float = 0.95,
) -> bool:
    y0, y1, x0, x1 = slices
    if isinstance(mask_hw, torch.Tensor):
        patch = mask_hw[y0:y1, x0:x1]
        return float(patch.float().mean().item()) >= min_valid_frac
    patch = np.asarray(mask_hw, dtype=bool)[y0:y1, x0:x1]
    return float(patch.mean()) >= min_valid_frac
