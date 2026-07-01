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


def gaussian_blur_per_band(x: torch.Tensor, sigma_px, truncate: float = DSEN2_PSF_TRUNCATE_DEFAULT) -> torch.Tensor:
    """``x``: [B,C,H,W]; ``sigma_px`` scalar or length-C sequence."""
    b, c, h, w = x.shape
    if isinstance(sigma_px, (list, tuple)):
        sigmas = list(sigma_px)
    elif hasattr(sigma_px, "__len__") and not isinstance(sigma_px, (str, bytes)):
        sigmas = list(sigma_px)
    else:
        sigmas = [float(sigma_px)] * c
    out = []
    for ch in range(c):
        k1 = _gaussian_kernel_1d(float(sigmas[ch]), truncate, x.device, x.dtype)
        k2 = k1[:, None] * k1[None, :]
        k2 = k2 / k2.sum()
        weight = k2.view(1, 1, k2.shape[0], k2.shape[1]).expand(1, 1, -1, -1)
        plane = x[:, ch : ch + 1]
        pad = k2.shape[0] // 2
        plane = F.pad(plane, (pad, pad, pad, pad), mode="reflect")
        out.append(F.conv2d(plane, weight))
    return torch.cat(out, dim=1)


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
) -> torch.Tensor:
    mode = str(degradation).lower().strip()
    if mode == "s2_psf":
        return degrade_sr_dsen2(x, scale, truncate=truncate)
    if mode == "s2_psf_m":
        sigmas = dict(sigma_m_by_band) if sigma_m_by_band is not None else dict(DEFAULT_S2_PSF_SIGMA_M_BY_BAND)
        return degrade_sr_to_s2_lr_meter_psf(
            x, scale, sigmas, band_order, native_gsd_m, truncate=truncate
        )
    if mode == "area":
        return area_downsample(x, scale)
    raise ValueError(f"Unknown degradation: {degradation!r}")
