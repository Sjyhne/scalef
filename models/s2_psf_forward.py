"""Sentinel-2–style differentiable blur + average-pool forward model (HR → LR)."""

from __future__ import annotations

import math
from typing import Dict, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_kernel1d(
    sigma_px: float,
    truncate: float = 4.0,
    device=None,
    dtype=None,
) -> torch.Tensor:
    if sigma_px <= 0:
        return torch.tensor([1.0], device=device, dtype=dtype)

    radius = int(math.ceil(truncate * sigma_px))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (x / sigma_px) ** 2)
    return k / k.sum()


def gaussian_blur_per_band(
    x: torch.Tensor,
    sigma_px_by_band: list[float],
    *,
    truncate: float = 4.0,
) -> torch.Tensor:
    """
    Args:
        x: [B, C, H, W]
        sigma_px_by_band: one sigma value per band, in SR (HR) pixels

    Returns:
        Blurred tensor with shape [B, C, H, W]
    """
    outputs = []

    for c, sigma_px in enumerate(sigma_px_by_band):
        xc = x[:, c : c + 1]

        k = gaussian_kernel1d(
            sigma_px,
            truncate=truncate,
            device=x.device,
            dtype=x.dtype,
        )

        if k.numel() > 1:
            r = k.numel() // 2

            kx = k.view(1, 1, 1, -1)
            ky = k.view(1, 1, -1, 1)

            xc = F.pad(xc, (r, r, 0, 0), mode="reflect")
            xc = F.conv2d(xc, kx)

            xc = F.pad(xc, (0, 0, r, r), mode="reflect")
            xc = F.conv2d(xc, ky)

        outputs.append(xc)

    return torch.cat(outputs, dim=1)


def degrade_sr_to_s2_lr(
    sr: torch.Tensor,
    scale: int,
    sigma_m_by_band: dict[str, float] | Mapping[str, float] | None = None,
    band_order: tuple[str, ...] = ("B02", "B03", "B04", "B08"),
    native_gsd_m: float = 10.0,
    *,
    truncate: float = 4.0,
) -> torch.Tensor:
    """
    Degrade SR image to Sentinel-2–like LR: optional per-band Gaussian blur in HR pixels,
    then block average pool (integration to native GSD).

    Args:
        sr: [C,H,W] or [B,C,H,W] high-resolution image
        scale: e.g. 4 for 2.5 m → 10 m when native_gsd_m is 10
        sigma_m_by_band: Gaussian PSF sigma in meters per band key. If None, only area-pools.
        band_order: band order in ``sr`` channels (length must match C)
        native_gsd_m: Sentinel-2 native GSD in meters (normally 10.0)
        truncate: Gaussian radius in sigma units for kernel support

    Returns:
        LR-sized tensor, same rank as input ([C,H',W'] or [B,C,H',W']).
    """
    squeeze_batch = False

    if sr.ndim == 3:
        sr = sr.unsqueeze(0)
        squeeze_batch = True

    if sr.ndim != 4:
        raise ValueError("Expected sr with shape [C,H,W] or [B,C,H,W].")

    _, c, h, w = sr.shape

    if h % scale != 0 or w % scale != 0:
        raise ValueError(
            f"SR size {(h, w)} is not divisible by scale={scale}.",
        )

    if sigma_m_by_band is not None:
        hr_gsd_m = native_gsd_m / float(scale)

        sigma_px_by_band = [
            float(sigma_m_by_band[band]) / hr_gsd_m
            for band in band_order
        ]

        if len(sigma_px_by_band) != c:
            raise ValueError(
                f"Got {c} SR bands, but {len(sigma_px_by_band)} PSF sigmas (band_order).",
            )

        sr = gaussian_blur_per_band(sr, sigma_px_by_band, truncate=truncate)

    lr = F.avg_pool2d(sr, kernel_size=scale, stride=scale)

    if squeeze_batch:
        lr = lr.squeeze(0)

    return lr


class S2PSFForward(nn.Module):
    """
    Differentiable Sentinel-2–like forward model: separable Gaussian per band on the HR
    grid, then block average pooling to native GSD (e.g. 10 m).

    x_hr: [B, C, H_hr, W_hr] — C must match ``len(band_order)``.
    y_lr: [B, C, H_lr, W_lr] with H_lr = H_hr / scale_factor (exact division required).

    Implementation is delegated to :func:`degrade_sr_to_s2_lr`.
    """

    def __init__(
        self,
        scale_factor: int,
        sigma_m_by_band: Mapping[str, float],
        band_order: Sequence[str] = ("B02", "B03", "B04"),
        native_gsd_m: float = 10.0,
        truncate: float = 4.0,
    ):
        super().__init__()
        if scale_factor < 1:
            raise ValueError("scale_factor must be >= 1")
        self.scale_factor = int(scale_factor)
        self.band_order = tuple(band_order)
        self.native_gsd_m = float(native_gsd_m)
        self.truncate = float(truncate)
        self._sigma_m_by_band: Dict[str, float] = {k: float(v) for k, v in sigma_m_by_band.items()}

    def forward(self, x_hr: torch.Tensor) -> torch.Tensor:
        return degrade_sr_to_s2_lr(
            x_hr,
            self.scale_factor,
            self._sigma_m_by_band,
            band_order=self.band_order,
            native_gsd_m=self.native_gsd_m,
            truncate=self.truncate,
        )


_CACHE: Dict[Tuple, S2PSFForward] = {}


def get_s2_psf_forward(
    scale_factor: int,
    sigma_m_by_band: Mapping[str, float],
    band_order: Sequence[str],
    native_gsd_m: float,
    truncate: float,
    device: torch.device,
) -> S2PSFForward:
    """Return a cached ``S2PSFForward`` moved to ``device`` (small cache keyed by hyperparameters)."""
    key = (
        str(device),
        int(scale_factor),
        tuple((b, float(sigma_m_by_band[b])) for b in band_order),
        float(native_gsd_m),
        float(truncate),
    )
    if key not in _CACHE:
        _CACHE[key] = S2PSFForward(
            scale_factor=int(scale_factor),
            sigma_m_by_band=sigma_m_by_band,
            band_order=band_order,
            native_gsd_m=native_gsd_m,
            truncate=truncate,
        )
    m = _CACHE[key]
    return m.to(device)
