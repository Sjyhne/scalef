"""Shared Sentinel-2 reflectance → display PNG preview helpers."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from s2_reflectance import (
    S2_RGB_DISPLAY_BANDS,
    stack_bands_hwc,
)

DEFAULT_PREVIEW_DOWNSAMPLE = 8
DEFAULT_PREVIEW_P_LOW = 1.0
DEFAULT_PREVIEW_P_HIGH = 99.0


def select_rgb_hwc(multiband_hwc: np.ndarray, band_names: tuple[str, ...]) -> np.ndarray:
    rgb = stack_bands_hwc(multiband_hwc, band_names, S2_RGB_DISPLAY_BANDS)
    if rgb is None:
        raise ValueError(f"Cannot select RGB from bands {band_names}")
    return rgb


def _finite_mask_hwc(rgb_hwc: np.ndarray) -> np.ndarray:
    return np.isfinite(np.asarray(rgb_hwc, dtype=np.float32)).all(axis=-1)


def _sample_pixels(
    rgb_hwc: np.ndarray,
    sample_mask: np.ndarray | None,
    *,
    max_samples: int = 2_000_000,
) -> np.ndarray:
    x = np.asarray(rgb_hwc, dtype=np.float32)[..., :3]
    if sample_mask is None:
        flat = x.reshape(-1, 3)
    else:
        flat = x[np.asarray(sample_mask, dtype=bool)]
    if flat.size == 0:
        return flat.reshape(0, 3)
    if flat.shape[0] > max_samples:
        idx = np.random.default_rng(0).choice(flat.shape[0], max_samples, replace=False)
        flat = flat[idx]
    return flat


def stretch_limits_scalar_legacy(
    rgb_list: list[np.ndarray],
    *,
    p_low: float = DEFAULT_PREVIEW_P_LOW,
    p_high: float = DEFAULT_PREVIEW_P_HIGH,
) -> tuple[np.ndarray, np.ndarray]:
    """Scalar p_low–p_high on all finite RGB values (matches legacy Atacama PNGs)."""
    stacked = np.stack([np.asarray(rgb, dtype=np.float32)[..., :3] for rgb in rgb_list], axis=0)
    lo = float(np.nanpercentile(stacked, p_low))
    hi = float(np.nanpercentile(stacked, p_high))
    lo3 = np.full(3, lo, dtype=np.float32)
    hi3 = np.full(3, hi, dtype=np.float32)
    return lo3, hi3


def stretch_limits_scalar(
    rgb_list: list[np.ndarray],
    sample_masks: list[np.ndarray | None],
    *,
    p_low: float = DEFAULT_PREVIEW_P_LOW,
    p_high: float = DEFAULT_PREVIEW_P_HIGH,
) -> tuple[np.ndarray, np.ndarray]:
    flats = [_sample_pixels(rgb, mask) for rgb, mask in zip(rgb_list, sample_masks)]
    flat = np.concatenate([f for f in flats if f.size], axis=0)
    if flat.size == 0:
        return stretch_limits_scalar_legacy(rgb_list, p_low=p_low, p_high=p_high)
    lo = float(np.percentile(flat, p_low))
    hi = float(np.percentile(flat, p_high))
    lo3 = np.full(3, lo, dtype=np.float32)
    hi3 = np.full(3, hi, dtype=np.float32)
    return lo3, hi3


def stretch_limits_per_channel(
    rgb_list: list[np.ndarray],
    sample_masks: list[np.ndarray | None],
    *,
    p_low: float = DEFAULT_PREVIEW_P_LOW,
    p_high: float = DEFAULT_PREVIEW_P_HIGH,
) -> tuple[np.ndarray, np.ndarray]:
    flats = [_sample_pixels(rgb, mask) for rgb, mask in zip(rgb_list, sample_masks)]
    flat = np.concatenate([f for f in flats if f.size], axis=0)
    if flat.size == 0:
        return np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32)
    lo = np.percentile(flat, p_low, axis=0).astype(np.float32)
    hi = np.percentile(flat, p_high, axis=0).astype(np.float32)
    return lo, hi


def scene_stretch_limits_from_arrays(
    rgb_list: list[np.ndarray],
    valid_list: list[np.ndarray | None],
    *,
    downsample: int = DEFAULT_PREVIEW_DOWNSAMPLE,
    p_low: float = DEFAULT_PREVIEW_P_LOW,
    p_high: float = DEFAULT_PREVIEW_P_HIGH,
    stretch: str = "scalar",
) -> tuple[np.ndarray, np.ndarray]:
    del downsample  # percentiles always from full-res samples
    if stretch == "scalar":
        return stretch_limits_scalar_legacy(rgb_list, p_low=p_low, p_high=p_high)
    sample_masks = [
        (valid & _finite_mask_hwc(rgb)) if valid is not None else _finite_mask_hwc(rgb)
        for rgb, valid in zip(rgb_list, valid_list)
    ]
    return stretch_limits_per_channel(rgb_list, sample_masks, p_low=p_low, p_high=p_high)


def downsample_nanmean_hwc(
    arr_hwc: np.ndarray,
    factor: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Area downsample using nanmean; returns (image, coverage mask)."""
    if factor <= 1:
        finite = np.isfinite(arr_hwc).all(axis=-1)
        return arr_hwc, finite

    x = np.asarray(arr_hwc, dtype=np.float32)
    h, w = x.shape[:2]
    out_h, out_w = max(1, h // factor), max(1, w // factor)
    h_trim, w_trim = out_h * factor, out_w * factor
    x = x[:h_trim, :w_trim]
    finite = np.isfinite(x).all(axis=-1)
    x = x.copy()
    x[~finite] = np.nan

    blocks = x.reshape(out_h, factor, out_w, factor, x.shape[2])
    finite_blocks = finite.reshape(out_h, factor, out_w, factor)
    counts = finite_blocks.sum(axis=(1, 3))
    sums = np.nansum(blocks, axis=(1, 3))
    out = np.zeros((out_h, out_w, x.shape[2]), dtype=np.float32)
    nz = counts > 0
    out[nz] = (sums[nz] / counts[nz, None]).astype(np.float32)
    valid_out = nz
    out[~valid_out] = np.nan
    return out, valid_out


def write_reflectance_preview_png(
    reflectance_hwc: np.ndarray,
    out_path: Path,
    *,
    band_names: tuple[str, ...] | None = None,
    valid_hw: np.ndarray | None = None,
    downsample: int = DEFAULT_PREVIEW_DOWNSAMPLE,
    lo: np.ndarray | None = None,
    hi: np.ndarray | None = None,
    p_low: float = DEFAULT_PREVIEW_P_LOW,
    p_high: float = DEFAULT_PREVIEW_P_HIGH,
    gamma: float = 1.0,
    stretch: str = "scalar",
) -> Path:
    del valid_hw  # clouds stay visible; only true nodata is suppressed
    if band_names is not None and reflectance_hwc.shape[-1] > 3:
        rgb = select_rgb_hwc(reflectance_hwc, band_names)
    else:
        rgb = np.asarray(reflectance_hwc[..., :3], dtype=np.float32)

    finite = _finite_mask_hwc(rgb)

    if lo is None or hi is None:
        if stretch == "per_channel":
            sample_mask = finite
            lo, hi = stretch_limits_per_channel([rgb], [sample_mask], p_low=p_low, p_high=p_high)
        else:
            lo, hi = stretch_limits_scalar_legacy([rgb], p_low=p_low, p_high=p_high)

    span = np.maximum(hi - lo, 1e-6)
    display = (np.asarray(rgb, dtype=np.float32) - lo) / span
    display = np.clip(display, 0.0, 1.0)
    if gamma != 1.0 and gamma > 0:
        display = np.power(display, float(gamma))
    display[~finite] = np.nan

    small, valid_small = downsample_nanmean_hwc(display, downsample)
    u8 = np.zeros(small.shape, dtype=np.uint8)
    show = valid_small & np.isfinite(small).all(axis=-1)
    u8[show] = (small[show] * 255.0).round().astype(np.uint8)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(u8, cv2.COLOR_RGB2BGR))
    return out_path
