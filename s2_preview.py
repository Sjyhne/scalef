"""Shared Sentinel-2 reflectance → display PNG preview helpers."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from s2_reflectance import (
    S2_RGB_DISPLAY_BANDS,
    _percentile_lo_hi_per_channel,
    _stretch_reflectance_to_uint8,
    stack_bands_hwc,
)

DEFAULT_PREVIEW_DOWNSAMPLE = 8
DEFAULT_PREVIEW_P_LOW = 2.0
DEFAULT_PREVIEW_P_HIGH = 98.0


def select_rgb_hwc(multiband_hwc: np.ndarray, band_names: tuple[str, ...]) -> np.ndarray:
    rgb = stack_bands_hwc(multiband_hwc, band_names, S2_RGB_DISPLAY_BANDS)
    if rgb is None:
        raise ValueError(f"Cannot select RGB from bands {band_names}")
    return rgb


def downsample_hwc(arr: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return arr
    h, w = arr.shape[:2]
    out_w, out_h = max(1, w // factor), max(1, h // factor)
    return cv2.resize(arr, (out_w, out_h), interpolation=cv2.INTER_AREA)


def downsample_valid(mask: np.ndarray | None, factor: int) -> np.ndarray | None:
    if mask is None or factor <= 1:
        return mask
    h, w = mask.shape
    out_w, out_h = max(1, w // factor), max(1, h // factor)
    small = cv2.resize(mask.astype(np.uint8), (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    return small.astype(bool)


def reflectance_to_preview_uint8(
    rgb_hwc: np.ndarray,
    valid_hw: np.ndarray | None,
    *,
    lo: np.ndarray | None = None,
    hi: np.ndarray | None = None,
    p_low: float = DEFAULT_PREVIEW_P_LOW,
    p_high: float = DEFAULT_PREVIEW_P_HIGH,
    gamma: float = 1.0,
) -> np.ndarray:
    if lo is None or hi is None:
        lo, hi = _percentile_lo_hi_per_channel(
            rgb_hwc, valid_hw, p_low=p_low, p_high=p_high
        )
    return _stretch_reflectance_to_uint8(rgb_hwc, lo, hi, gamma=gamma)


def scene_stretch_limits_from_arrays(
    rgb_list: list[np.ndarray],
    valid_list: list[np.ndarray | None],
    *,
    downsample: int = DEFAULT_PREVIEW_DOWNSAMPLE,
    p_low: float = DEFAULT_PREVIEW_P_LOW,
    p_high: float = DEFAULT_PREVIEW_P_HIGH,
) -> tuple[np.ndarray, np.ndarray]:
    stacks: list[np.ndarray] = []
    for rgb, valid in zip(rgb_list, valid_list):
        rgb_small = downsample_hwc(rgb, downsample)
        valid_small = downsample_valid(valid, downsample)
        x = rgb_small.astype(np.float32)
        if valid_small is not None:
            for c in range(3):
                x[..., c] = np.where(valid_small, x[..., c], np.nan)
        stacks.append(x)
    stacked = np.stack(stacks, axis=0)
    lo = np.nanpercentile(stacked, p_low, axis=(0, 1, 2)).astype(np.float32)
    hi = np.nanpercentile(stacked, p_high, axis=(0, 1, 2)).astype(np.float32)
    return lo, hi


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
) -> Path:
    if band_names is not None and reflectance_hwc.shape[-1] > 3:
        rgb = select_rgb_hwc(reflectance_hwc, band_names)
    else:
        rgb = reflectance_hwc[..., :3]
    rgb_small = downsample_hwc(rgb, downsample)
    valid_small = downsample_valid(valid_hw, downsample)
    u8 = reflectance_to_preview_uint8(
        rgb_small,
        valid_small,
        lo=lo,
        hi=hi,
        p_low=p_low,
        p_high=p_high,
        gamma=gamma,
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(u8, cv2.COLOR_RGB2BGR))
    return out_path
