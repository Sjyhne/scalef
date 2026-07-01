"""Sentinel-2 reflectance helpers (FORCE BOA conventions + scene contrast stretch)."""

from __future__ import annotations

import numpy as np

S2_RGB_DISPLAY_BANDS: tuple[str, ...] = ("B04", "B03", "B02")
FORCE_BOA_SCALE = 10000.0
FORCE_NODATA = -9999


def bands_chw_to_reflectance_hwc(
    chw: np.ndarray,
    band_indices: tuple[int, ...] | list[int],
    *,
    scale: float = FORCE_BOA_SCALE,
    nodata: float | None = 0.0,
) -> np.ndarray:
    planes = [np.asarray(chw[b], dtype=np.float32) for b in band_indices]
    rgb = np.stack(planes, axis=-1)
    rgb = np.clip(rgb, 0.0, None)
    if nodata is not None:
        rgb = np.where(rgb == nodata, 0.0, rgb)
    if scale > 0:
        rgb = rgb / float(scale)
    return np.clip(rgb, 0.0, 1.0)


def reflectance_to_uint8_linear(rgb: np.ndarray) -> np.ndarray:
    return (np.clip(rgb, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def _stretch_reflectance_to_uint8(
    rgb: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    gamma: float = 1.0,
) -> np.ndarray:
    span = np.maximum(hi - lo, 1e-6)
    x = (rgb - lo) / span
    x = np.clip(x, 0.0, 1.0)
    if gamma != 1.0 and gamma > 0:
        x = np.power(x, float(gamma))
    return reflectance_to_uint8_linear(x)


def reflectance_list_to_uint8_scene_percentile(
    reflectance_hwc_list: list[np.ndarray],
    *,
    p_low: float = 2.0,
    p_high: float = 98.0,
    gamma: float = 1.0,
) -> list[np.ndarray]:
    if not reflectance_hwc_list:
        return []
    stacked = np.stack(reflectance_hwc_list, axis=0)
    lo = np.nanpercentile(stacked, p_low, axis=(0, 1, 2))
    hi = np.nanpercentile(stacked, p_high, axis=(0, 1, 2))
    return [_stretch_reflectance_to_uint8(rgb, lo, hi, gamma=gamma) for rgb in reflectance_hwc_list]


def reflectance_list_to_uint8_per_frame_percentile(
    reflectance_hwc_list: list[np.ndarray],
    *,
    p_low: float = 1.0,
    p_high: float = 99.0,
    gamma: float = 1.0,
) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for rgb in reflectance_hwc_list:
        lo = float(np.nanpercentile(rgb, p_low))
        hi = float(np.nanpercentile(rgb, p_high))
        out.append(_stretch_reflectance_to_uint8(rgb, lo, hi, gamma=gamma))
    return out


def _rgb_hwc_for_stretch(rgb_hwc: np.ndarray) -> np.ndarray:
    x = np.asarray(rgb_hwc, dtype=np.float32)
    if x.ndim == 2:
        x = np.stack([x, x, x], axis=-1)
    if x.shape[-1] < 3:
        x = np.repeat(x[..., :1], 3, axis=-1)
    return x[..., :3]


def _percentile_lo_hi_per_channel(
    rgb_hwc: np.ndarray,
    valid_hw: np.ndarray | None,
    *,
    p_low: float,
    p_high: float,
) -> tuple[np.ndarray, np.ndarray]:
    x = _rgb_hwc_for_stretch(rgb_hwc)
    if valid_hw is not None:
        mask = np.asarray(valid_hw, dtype=bool)
        if mask.shape != x.shape[:2]:
            raise ValueError(f"valid mask {mask.shape} does not match image {x.shape[:2]}")
        lo = np.array(
            [float(np.nanpercentile(x[..., c][mask], p_low)) for c in range(3)],
            dtype=np.float32,
        )
        hi = np.array(
            [float(np.nanpercentile(x[..., c][mask], p_high)) for c in range(3)],
            dtype=np.float32,
        )
    else:
        lo = np.nanpercentile(x, p_low, axis=(0, 1)).astype(np.float32)
        hi = np.nanpercentile(x, p_high, axis=(0, 1)).astype(np.float32)
    return lo, hi


def reflectance_hwc_rgb_to_uint8(
    rgb_hwc: np.ndarray,
    *,
    p_low: float = 2.0,
    p_high: float = 98.0,
    gamma: float = 1.0,
    valid_hw: np.ndarray | None = None,
) -> np.ndarray:
    x = _rgb_hwc_for_stretch(rgb_hwc)
    lo, hi = _percentile_lo_hi_per_channel(x, valid_hw, p_low=p_low, p_high=p_high)
    return _stretch_reflectance_to_uint8(x, lo, hi, gamma=gamma)


def stack_bands_hwc(
    multiband_hwc: np.ndarray,
    band_names: tuple[str, ...] | list[str],
    order: tuple[str, ...],
) -> np.ndarray | None:
    name_to_idx = {str(n).upper(): i for i, n in enumerate(band_names)}
    planes = []
    for band in order:
        idx = name_to_idx.get(str(band).upper())
        if idx is None:
            return None
        planes.append(multiband_hwc[..., idx])
    return np.stack(planes, axis=-1)
