"""Score reconstructions on a shared geographic HR window.

Nested and size-ladder comparisons must not let the scoring window grow with
training-field size. Crop every prediction, reference, bilinear baseline, and
mask to the same center window, then score. The bilinear score is then an
invariant of the comparison, not of the crop.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
import torch

from eval.masked_metrics import masked_mse, masked_psnr
from eval.spot_metrics import crop_bchw


def center_crop_slices(
    height: int, width: int, target_h: int, target_w: int
) -> tuple[int, int, int, int]:
    """Return ``(y0, y1, x0, x1)`` for a centered ``target_h×target_w`` window."""
    h, w = int(height), int(width)
    th = min(max(1, int(target_h)), h)
    tw = min(max(1, int(target_w)), w)
    y0 = max(0, (h - th) // 2)
    x0 = max(0, (w - tw) // 2)
    return y0, y0 + th, x0, x0 + tw


def common_center_hw(shapes: Sequence[tuple[int, int]]) -> tuple[int, int]:
    """Intersection of center-aligned HR shapes: min height × min width."""
    if not shapes:
        raise ValueError("common_center_hw requires at least one shape")
    height = min(int(h) for h, _ in shapes)
    width = min(int(w) for _, w in shapes)
    if height < 1 or width < 1:
        raise ValueError(f"empty common footprint from shapes {list(shapes)}")
    return height, width


def nest_window_frac(
    child_iy: int,
    child_ix: int,
    child_side: int,
    parent_side: int,
) -> tuple[float, float, float, float]:
    """Return ``(fy, fx, fh, fw)`` of a nested child inside its parent field."""
    child_side = int(child_side)
    parent_side = int(parent_side)
    if child_side < 1 or parent_side < child_side or parent_side % child_side != 0:
        raise ValueError(f"child_side {child_side} must divide parent_side {parent_side}")
    n = parent_side // child_side
    if not (0 <= int(child_iy) < n and 0 <= int(child_ix) < n):
        raise ValueError(f"nest index ({child_iy}, {child_ix}) outside 0..{n - 1}")
    return int(child_iy) / n, int(child_ix) / n, 1.0 / n, 1.0 / n


def crop_hwc_frac(
    image: np.ndarray, fy: float, fx: float, fh: float, fw: float
) -> np.ndarray:
    """Crop an HWC array to a fractional window."""
    height, width = int(image.shape[0]), int(image.shape[1])
    y0 = int(round(float(fy) * height))
    x0 = int(round(float(fx) * width))
    y1 = max(y0 + 1, int(round((float(fy) + float(fh)) * height)))
    x1 = max(x0 + 1, int(round((float(fx) + float(fw)) * width)))
    return image[y0:y1, x0:x1]


def crop_bchw_to_hw(tensor: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """Center-crop a BCHW tensor to ``target_h×target_w``."""
    _, _, height, width = tensor.shape
    return crop_bchw(tensor, center_crop_slices(height, width, target_h, target_w))


def crop_mask_to_hw(mask_hw: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """Center-crop an HW mask to ``target_h×target_w``."""
    height, width = int(mask_hw.shape[-2]), int(mask_hw.shape[-1])
    y0, y1, x0, x1 = center_crop_slices(height, width, target_h, target_w)
    return mask_hw[..., y0:y1, x0:x1]


def crop_pair_to_common(
    pred_bchw: torch.Tensor,
    gt_bchw: torch.Tensor,
    bilinear_bchw: torch.Tensor,
    mask_hw: torch.Tensor | None,
    *,
    target_hw: tuple[int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, tuple[int, int]]:
    """Crop pred/gt/bilinear/(mask) onto one shared center window."""
    shapes = [
        (int(pred_bchw.shape[-2]), int(pred_bchw.shape[-1])),
        (int(gt_bchw.shape[-2]), int(gt_bchw.shape[-1])),
        (int(bilinear_bchw.shape[-2]), int(bilinear_bchw.shape[-1])),
    ]
    if mask_hw is not None:
        shapes.append((int(mask_hw.shape[-2]), int(mask_hw.shape[-1])))
    common = common_center_hw(shapes)
    height, width = target_hw if target_hw is not None else common
    height, width = common_center_hw([(height, width), common])
    pred = crop_bchw_to_hw(pred_bchw, height, width)
    gt = crop_bchw_to_hw(gt_bchw, height, width)
    bilinear = crop_bchw_to_hw(bilinear_bchw, height, width)
    mask = crop_mask_to_hw(mask_hw, height, width) if mask_hw is not None else None
    return pred, gt, bilinear, mask, (height, width)


def score_common_footprint(
    pred_bchw: torch.Tensor,
    gt_bchw: torch.Tensor,
    bilinear_bchw: torch.Tensor,
    mask_hw: torch.Tensor | None = None,
    *,
    target_hw: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Masked PSNR/MSE on the shared center window.

    SSIM/LPIPS stay with the full evaluator; this function is the crop-invariant
    structural check used before publishing a field-size comparison.
    """
    pred, gt, bilinear, mask, hw = crop_pair_to_common(
        pred_bchw, gt_bchw, bilinear_bchw, mask_hw, target_hw=target_hw
    )
    if mask is None:
        mask = torch.ones(hw, dtype=torch.bool, device=pred.device)
    if not bool(mask.any()):
        raise ValueError("common-footprint mask is empty")
    model_psnr = masked_psnr(pred, gt, mask)
    bilinear_psnr = masked_psnr(bilinear, gt, mask)
    model_mse = masked_mse(pred, gt, mask)
    bilinear_mse = masked_mse(bilinear, gt, mask)
    return {
        "common_footprint": True,
        "hr_hw": [int(hw[0]), int(hw[1])],
        "valid_fraction": float(mask.float().mean().item()),
        "model_psnr": model_psnr,
        "bilinear_psnr": bilinear_psnr,
        "psnr_improvement": model_psnr - bilinear_psnr,
        "model_mse": model_mse,
        "bilinear_mse": bilinear_mse,
        "mse_improvement": bilinear_mse - model_mse,
    }


def bilinear_scores_are_invariant(
    scores: Iterable[dict[str, Any]],
    *,
    keys: Sequence[str] = ("bilinear_psnr", "bilinear_mse"),
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> None:
    """Fail if bilinear scores differ across field-size variants.

    The required nested/size-ladder invariant: identical baseline pixels,
    reference, transform, windows, and masks ⇒ identical bilinear scores.
    """
    rows = list(scores)
    if len(rows) < 2:
        raise ValueError("bilinear invariant needs at least two scored variants")
    ref = rows[0]
    for idx, row in enumerate(rows[1:], start=1):
        for key in keys:
            a = float(ref[key])
            b = float(row[key])
            if not np.isclose(a, b, atol=atol, rtol=rtol, equal_nan=True):
                raise ValueError(
                    f"bilinear invariant failed on {key}: variant 0={a!r} vs variant {idx}={b!r}"
                )
