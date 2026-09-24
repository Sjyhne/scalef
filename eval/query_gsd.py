"""Query-time scoring of a fitted field on a different output GSD.

Training stays on the native ScaleF grid (usually 2.5 m, df=4). After restore,
the continuous INR is decoded on a denser or coarser coordinate grid covering
the same AOI and scored against NIB warped to that GSD.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch
from rasterio.transform import Affine

from data import _make_coord_grid
from eval.hr_render import render_hr_rgb_tiled, resolve_hr_render_tile
from eval.masked_metrics import compute_masked_image_metrics
from s2_dataset import (
    _apply_hr_spatial_shift,
    _build_hr_eval_mask,
    _city_id_from_s2_dir,
    _harmonize_hr_histogram_match,
    _warp_rgb_to_grid,
    load_hr_eval_shift,
)


def parse_query_gsd_m(raw: str | None) -> list[float]:
    """Parse ``--query_gsd_m 5,1`` into metres. Empty / None → no extra evals."""
    if raw is None:
        return []
    values: list[float] = []
    for part in str(raw).split(","):
        text = part.strip()
        if not text:
            continue
        gsd = float(text)
        if gsd <= 0:
            raise ValueError(f"query GSD must be positive, got {gsd}")
        values.append(gsd)
    return values


def query_grid_hw(lr_height: int, lr_width: int, native_gsd_m: float, query_gsd_m: float) -> tuple[int, int, int]:
    """Return ``(query_h, query_w, query_df)`` for an integer scale factor."""
    ratio = float(native_gsd_m) / float(query_gsd_m)
    query_df = int(round(ratio))
    if query_df < 1 or abs(ratio - query_df) > 1e-6:
        raise ValueError(
            f"query GSD {query_gsd_m:g} m does not divide native {native_gsd_m:g} m "
            f"(ratio={ratio:g})"
        )
    return int(lr_height) * query_df, int(lr_width) * query_df, query_df


def load_query_reference(dataset, query_gsd_m: float, args) -> dict[str, Any]:
    """Warp NIB to ``query_gsd_m`` on the dataset AOI, with the usual eval prep."""
    if not getattr(dataset, "has_hr_gt", False) or dataset.hr_path is None:
        raise ValueError("query-time GSD scoring requires HR GT")
    height, width, query_df = query_grid_hw(
        dataset.lr_height, dataset.lr_width, dataset.native_gsd_m, query_gsd_m
    )
    query_transform = Affine(
        float(dataset.lr_transform.a) / query_df,
        0.0,
        float(dataset.lr_transform.c),
        0.0,
        float(dataset.lr_transform.e) / query_df,
        float(dataset.lr_transform.f),
    )
    hr_rgb, hr_cover, _, _ = _warp_rgb_to_grid(
        dataset.hr_path, dataset.crs, query_transform, height, width
    )
    s2_valid_lr = (dataset.lr_rgb[0].detach().cpu().numpy() > 0).any(axis=-1)
    mask = _build_hr_eval_mask(hr_cover, s2_valid_lr, hr_rgb, query_df)
    if not bool(getattr(args, "no_hr_harmonize", False)):
        base = dataset.lr_rgb[int(dataset.base_frame_index)].detach().cpu().numpy()
        hr_rgb = _harmonize_hr_histogram_match(hr_rgb, base)
    shift = None
    if not bool(getattr(args, "no_hr_spatial_align", False)):
        city = _city_id_from_s2_dir(dataset.s2_dir)
        shift = load_hr_eval_shift(
            city,
            path=getattr(args, "spatial_alignment_path", None),
            current_df=query_df,
            s2_dir=dataset.s2_dir,
            parent_tile_id=(dataset.meta.get("patch_grid") or {}).get("parent_tile_id"),
        )
        if shift is not None:
            hr_rgb, mask = _apply_hr_spatial_shift(hr_rgb, mask, shift[0], shift[1])
    return {
        "gsd_m": float(query_gsd_m),
        "df": int(query_df),
        "hr_hwc": np.clip(hr_rgb.astype(np.float32), 0.0, 1.0),
        "mask_hw": mask.astype(bool),
        "shift_yx": list(shift) if shift is not None else None,
        "height": int(height),
        "width": int(width),
    }


def _metrics_payload(frame_metrics: dict[str, Any], *, extra: dict[str, Any]) -> dict[str, Any]:
    return {
        **extra,
        "lpips": float(frame_metrics["model_lpips"]),
        "lpips_bilinear": float(frame_metrics["bilinear_lpips"]),
        "lpips_improvement": float(
            frame_metrics["bilinear_lpips"] - frame_metrics["model_lpips"]
        ),
        "psnr": float(frame_metrics["model_psnr"]),
        "psnr_bilinear": float(frame_metrics["bilinear_psnr"]),
        "ssim": float(frame_metrics["model_ssim"]),
        "ssim_bilinear": float(frame_metrics["bilinear_ssim"]),
        "valid_fraction": float(frame_metrics.get("valid_fraction") or 0.0),
        "masked": bool(frame_metrics.get("masked", False)),
    }


@torch.no_grad()
def eval_one_query_gsd(
    model,
    dataset,
    query_gsd_m: float,
    *,
    device: torch.device,
    args,
    fwd_kwargs: dict,
    lpips_fn,
    train_pred_hwc: np.ndarray | None = None,
) -> dict[str, Any]:
    """Decode the fitted field at ``query_gsd_m`` and score against NIB."""
    import time

    t0 = time.perf_counter()
    ref = load_query_reference(dataset, query_gsd_m, args)
    height, width = ref["height"], ref["width"]
    coords = _make_coord_grid(height, width, device=device)
    sample_id = torch.zeros(1, device=device, dtype=torch.long)
    tile = resolve_hr_render_tile(height, width, int(getattr(args, "hr_render_tile", 0) or 0))
    render_kwargs = dict(fwd_kwargs)
    eval_dtype = render_kwargs.pop("eval_autocast_dtype", None)
    pred = render_hr_rgb_tiled(
        model,
        coords,
        sample_id,
        device=device,
        tile=tile,
        eval_autocast_dtype=eval_dtype,
        **render_kwargs,
    )
    mean = dataset.get_lr_mean(0).to(device=device, dtype=pred.dtype).view(1, 1, 1, -1)
    std = dataset.get_lr_std(0).to(device=device, dtype=pred.dtype).view(1, 1, 1, -1)
    pred = torch.clamp(pred * std + mean, 0.0, 1.0)
    pred_bchw = pred.permute(0, 3, 1, 2)
    gt_bchw = torch.from_numpy(ref["hr_hwc"]).permute(2, 0, 1).unsqueeze(0).to(device)
    lr = dataset.get_lr_sample(0).detach().cpu().numpy()
    if lr.ndim == 3 and lr.shape[0] in (1, 3, 4):
        lr = np.transpose(lr, (1, 2, 0))
    bilinear = cv2.resize(np.clip(lr[..., :3], 0.0, 1.0), (width, height), interpolation=cv2.INTER_LINEAR)
    bilinear_bchw = (
        torch.from_numpy(np.clip(bilinear, 0.0, 1.0).astype(np.float32))
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )
    mask = torch.from_numpy(ref["mask_hw"])
    queried = _metrics_payload(
        compute_masked_image_metrics(
            pred_bchw, gt_bchw, bilinear_bchw, mask, device=device, lpips_fn=lpips_fn
        ),
        extra={
            "mode": "query_field",
            "gsd_m": float(query_gsd_m),
            "df": ref["df"],
            "height": height,
            "width": width,
            "shift_yx": ref["shift_yx"],
            "render_s": float(time.perf_counter() - t0),
        },
    )
    payload: dict[str, Any] = {"query_field": queried}
    if train_pred_hwc is not None and abs(float(query_gsd_m) - float(dataset.hr_gsd_m)) > 1e-6:
        resampled = cv2.resize(
            np.clip(train_pred_hwc, 0.0, 1.0),
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )
        resampled_bchw = (
            torch.from_numpy(resampled.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(device)
        )
        payload["resampled_from_train_gsd"] = _metrics_payload(
            compute_masked_image_metrics(
                resampled_bchw, gt_bchw, bilinear_bchw, mask, device=device, lpips_fn=lpips_fn
            ),
            extra={
                "mode": "resampled_from_train_gsd",
                "gsd_m": float(query_gsd_m),
                "train_gsd_m": float(dataset.hr_gsd_m),
                "df": ref["df"],
                "height": height,
                "width": width,
            },
        )
    return payload


@torch.no_grad()
def eval_query_gsds(
    model,
    dataset,
    query_gsds: list[float],
    *,
    device: torch.device,
    args,
    fwd_kwargs: dict,
    lpips_fn,
    train_pred_hwc: np.ndarray | None = None,
) -> dict[str, Any]:
    """Score each requested GSD. Skips the training GSD (already in the main eval)."""
    train_gsd = float(getattr(dataset, "hr_gsd_m", 0.0) or 0.0)
    out: dict[str, Any] = {"train_gsd_m": train_gsd, "grids": {}}
    for gsd in query_gsds:
        if train_gsd > 0 and abs(float(gsd) - train_gsd) < 1e-6:
            continue
        key = f"{gsd:g}m"
        print(f"Query-time eval at {gsd:g} m ...", flush=True)
        out["grids"][key] = eval_one_query_gsd(
            model,
            dataset,
            float(gsd),
            device=device,
            args=args,
            fwd_kwargs=fwd_kwargs,
            lpips_fn=lpips_fn,
            train_pred_hwc=train_pred_hwc,
        )
        q = out["grids"][key]["query_field"]
        print(
            f"  {gsd:g} m query LPIPS {q['lpips']:.4f} (bil {q['lpips_bilinear']:.4f}, "
            f"{q['render_s']:.1f}s)",
            flush=True,
        )
        resampled = out["grids"][key].get("resampled_from_train_gsd")
        if resampled is not None:
            print(
                f"  {gsd:g} m resampled-from-{train_gsd:g}m LPIPS {resampled['lpips']:.4f}",
                flush=True,
            )
    return out
