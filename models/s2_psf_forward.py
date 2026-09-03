"""HR→LR degradation operators for supervision (DSen2 + legacy meter PSF)."""

from __future__ import annotations

import math
from typing import Mapping

import torch
import torch.nn.functional as F

S2_RGB_BAND_ORDER = ("B04", "B03", "B02")
DSEN2_PSF_TRUNCATE_DEFAULT = 4.0

# Approximate native PSF σ in meters at 10 m GSD (legacy ScaleF calibration).
DEFAULT_S2_PSF_SIGMA_M_BY_BAND: dict[str, float] = {
    "B02": 2.8,
    "B03": 3.25,
    "B04": 4.2,
    "B08": 3.5,
}


def dsen2_sigma_px(scale: int) -> float:
    scale = max(1, int(scale))
    return 1.0 / float(scale)


def _gaussian_kernel_1d(sigma_px: float, truncate: float, device, dtype) -> torch.Tensor:
    sigma_px = max(float(sigma_px), 1e-6)
    radius = max(1, int(math.ceil(truncate * sigma_px)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (x / sigma_px) ** 2)
    k = k / k.sum()
    return k


_BLUR_KERNEL_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor, int]] = {}


def _separable_blur_weights(sigmas, truncate: float, device, dtype):
    """Grouped separable Gaussian weights, cached across calls.

    Returns ``(w_x, w_y, radius)`` shaped [C,1,1,K] and [C,1,K,1]. Bands with a
    smaller sigma are zero-padded up to the widest kernel so every channel can
    run in one grouped convolution; the zero taps contribute nothing, so the
    result matches per-band kernels of their natural radius.
    """
    key = (tuple(round(float(s), 6) for s in sigmas), float(truncate), str(device), dtype)
    cached = _BLUR_KERNEL_CACHE.get(key)
    if cached is not None:
        return cached

    ks = [_gaussian_kernel_1d(float(s), truncate, device, dtype) for s in sigmas]
    radius = max(k.numel() // 2 for k in ks)
    size = 2 * radius + 1
    stacked = torch.zeros(len(ks), size, device=device, dtype=dtype)
    for i, k in enumerate(ks):
        off = radius - k.numel() // 2
        stacked[i, off : off + k.numel()] = k
    out = (stacked.view(-1, 1, 1, size), stacked.view(-1, 1, size, 1), radius)
    _BLUR_KERNEL_CACHE[key] = out
    return out


def gaussian_blur_per_band(x: torch.Tensor, sigma_px, truncate: float = DSEN2_PSF_TRUNCATE_DEFAULT) -> torch.Tensor:
    """``x``: [B,C,H,W]; ``sigma_px`` scalar or length-C sequence."""
    c = x.shape[1]
    if isinstance(sigma_px, (list, tuple)):
        sigmas = list(sigma_px)
    elif hasattr(sigma_px, "__len__") and not isinstance(sigma_px, (str, bytes)):
        sigmas = list(sigma_px)
    else:
        sigmas = [float(sigma_px)] * c

    # A Gaussian is separable, so two 1-D passes replace a K x K convolution.
    # Reflection padding commutes with the per-axis convolution, so padding and
    # filtering one axis at a time matches the 2-D reflect-pad-then-convolve form.
    w_x, w_y, pad = _separable_blur_weights(sigmas, truncate, x.device, x.dtype)
    y = F.pad(x, (pad, pad, 0, 0), mode="reflect")
    y = F.conv2d(y, w_x, groups=c)
    y = F.pad(y, (0, 0, pad, pad), mode="reflect")
    return F.conv2d(y, w_y, groups=c)


def area_downsample(x: torch.Tensor, scale: int) -> torch.Tensor:
    scale = max(1, int(scale))
    return F.avg_pool2d(x, kernel_size=scale, stride=scale)


def degrade_sr_dsen2(x: torch.Tensor, scale: int, *, truncate: float = DSEN2_PSF_TRUNCATE_DEFAULT) -> torch.Tensor:
    scale = max(1, int(scale))
    y = gaussian_blur_per_band(x, dsen2_sigma_px(scale), truncate=truncate)
    return area_downsample(y, scale)


def degrade_sr_to_s2_lr_meter_psf(
    x: torch.Tensor,
    scale: int,
    sigma_m_by_band: Mapping[str, float],
    band_order: tuple[str, ...] = S2_RGB_BAND_ORDER,
    native_gsd_m: float = 10.0,
    *,
    truncate: float = DSEN2_PSF_TRUNCATE_DEFAULT,
) -> torch.Tensor:
    scale = max(1, int(scale))
    gsd_hr = float(native_gsd_m) / float(scale)
    sigmas_px = [float(sigma_m_by_band.get(b, 4.0)) / gsd_hr for b in band_order]
    c = x.shape[1]
    if len(sigmas_px) < c:
        sigmas_px = sigmas_px + [sigmas_px[-1]] * (c - len(sigmas_px))
    else:
        sigmas_px = sigmas_px[:c]
    y = gaussian_blur_per_band(x, sigmas_px, truncate=truncate)
    return area_downsample(y, scale)


def degrade_hr_bchw(
    x: torch.Tensor,
    scale: int,
    degradation: str,
    *,
    truncate: float = DSEN2_PSF_TRUNCATE_DEFAULT,
    band_order: tuple[str, ...] = S2_RGB_BAND_ORDER,
    native_gsd_m: float = 10.0,
    sigma_m_by_band: Mapping[str, float] | None = None,
    psf_sigma_scale: float = 1.0,
) -> torch.Tensor:
    mode = str(degradation).lower().strip()
    sigma_scale = max(0.0, float(psf_sigma_scale))
    if mode == "s2_psf":
        if sigma_scale <= 0.0:
            return area_downsample(x, scale)
        sigma = dsen2_sigma_px(scale) * sigma_scale
        if sigma_scale < 1.0:
            y = gaussian_blur_per_band(x, sigma, truncate=truncate)
            return area_downsample(y, scale)
        return degrade_sr_dsen2(x, scale, truncate=truncate)
    if mode == "s2_psf_m":
        if sigma_scale <= 0.0:
            return area_downsample(x, scale)
        sigmas = dict(sigma_m_by_band) if sigma_m_by_band is not None else dict(DEFAULT_S2_PSF_SIGMA_M_BY_BAND)
        if sigma_scale != 1.0:
            sigmas = {k: v * sigma_scale for k, v in sigmas.items()}
        return degrade_sr_to_s2_lr_meter_psf(
            x, scale, sigmas, band_order, native_gsd_m, truncate=truncate
        )
    if mode == "area":
        return area_downsample(x, scale)
    raise ValueError(f"Unknown degradation: {degradation!r}")
