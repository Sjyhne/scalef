"""
Sentinel-2 BOA reflectance (B4, B3, B2) helpers and *display* RGB conversion.

Use reflectance tensors (approximately 0–1, or 0–10000 DN-style) for SR / INR.
Apply `s2_to_rgb` **after** super-resolution for visualization (uint8 RGB),
so per-scene contrast is not baked in before upsampling.
"""
from __future__ import annotations

import numpy as np

def s2_to_rgb(
    data: np.ndarray,
    smooth_quantiles: bool = True,
    gamma: float = 0.7,
) -> np.ndarray:
    """
    Sentinel-2 style true-color preview (uint8 HWC).

    Parameters
    ----------
    data :
        Channel-first array (C, H, W). Either:
        - C == 3: already B4, B3, B2 (same order as ESA band names; R, G, B).
        - C in (4..16): full S2 stack; uses indices [3,2,1] as B4,B3,B2.
    """
    if len(data.shape) == 4:
        data = data[0]

    if data.shape[0] == 3:
        rgb = np.transpose(data, (1, 2, 0))
    elif data.shape[0] > 13:
        rgb = data[:, :, [3, 2, 1]]
    else:
        rgb = data[[3, 2, 1]].transpose((1, 2, 0))

    if smooth_quantiles:
        min_value, q99_value = np.quantile(rgb, q=[0.0, 0.99])
        min_value = np.clip(min_value, 0, 1000)
        q99_value = np.clip(q99_value, 2000, 20000)
        rgb = (rgb - min_value) / (q99_value - min_value + 1e-6)
    else:
        rgb = rgb / 2000.0

    rgb = np.clip(rgb, 0, 1)
    if gamma is not None:
        rgb = np.power(rgb, gamma)

    return (rgb * 255.0).round().astype(np.uint8)


def reflectance_b432_chw_to_dn_style(chw: np.ndarray) -> np.ndarray:
    """Scale (3,H,W) float reflectance ~[0,1] to DN-style range expected by ``s2_to_rgb``."""
    x = np.clip(chw.astype(np.float32), 0.0, None) * 10000.0
    return x


def post_sr_reflectance_to_rgb_u8(
    sr_b432_chw: np.ndarray,
    smooth_quantiles: bool = True,
    gamma: float = 0.7,
) -> np.ndarray:
    """
    Convert SR output B4,B3,B2 reflectance (C,H,W, ~0–1) to uint8 HWC RGB.

    Internally rescales to DN-style before ``s2_to_rgb`` so quantile stretch matches
    the training / benchmark pipelines.
    """
    dn = reflectance_b432_chw_to_dn_style(sr_b432_chw)
    return s2_to_rgb(dn, smooth_quantiles=smooth_quantiles, gamma=gamma)


def chw_to_hwc_reflectance(chw: np.ndarray) -> np.ndarray:
    """(3,H,W) float → (H,W,3)."""
    return np.transpose(np.asarray(chw, dtype=np.float32), (1, 2, 0))


def hwc_to_chw_reflectance(hwc: np.ndarray) -> np.ndarray:
    """(H,W,3) float → (3,H,W)."""
    return np.transpose(np.asarray(hwc, dtype=np.float32), (2, 0, 1))
