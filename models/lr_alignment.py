"""HR prediction → LR supervision alignment (area downsample or S2 PSF)."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F

from models.s2_psf_forward import (
    DEFAULT_S2_PSF_SIGMA_M_BY_BAND,
    DSEN2_PSF_TRUNCATE_DEFAULT,
    degrade_hr_bchw,
)


def default_lr_align_args(*, lr_degradation: str = "area") -> SimpleNamespace:
    """Minimal ``args`` namespace for ``align_prediction_hwc_to_target``."""
    sigmas = dict(DEFAULT_S2_PSF_SIGMA_M_BY_BAND)
    return SimpleNamespace(
        lr_degradation=str(lr_degradation),
        s2_native_gsd_m=10.0,
        s2_psf_truncate=DSEN2_PSF_TRUNCATE_DEFAULT,
        s2_psf_sigma_b02_m=sigmas["B02"],
        s2_psf_sigma_b03_m=sigmas["B03"],
        s2_psf_sigma_b04_m=sigmas["B04"],
        s2_psf_sigma_b08_m=sigmas["B08"],
    )


def _df_from_shapes(hp: int, wp: int, ht: int, wt: int) -> int:
    if min(hp, wp, ht, wt) <= 0:
        raise ValueError(
            f"lr_alignment: invalid spatial sizes HR=({hp},{wp}) LR=({ht},{wt}).",
        )
    df_h = hp // ht
    df_w = wp // wt
    if df_h != df_w or hp != df_h * ht or wp != df_w * wt:
        raise ValueError(
            f"lr_alignment: HR ({hp}x{wp}) must be an integer tile multiple of "
            f"LR ({ht}x{wt}); got df_h={df_h}, df_w={df_w}.",
        )
    return int(df_h)


def align_prediction_hwc_to_target(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    args,
    device: torch.device,
) -> torch.Tensor:
    """Resize ``pred`` [B,H,W,C] to match ``target`` [B,Ht,Wt,C]."""
    del device
    if pred.shape[1:3] == target.shape[1:3]:
        return pred

    _, ht, wt, _ = target.shape
    x = pred.permute(0, 3, 1, 2).contiguous()
    degradation = str(getattr(args, "lr_degradation", "area")).lower().strip()

    if degradation == "area":
        x = F.interpolate(x, size=(ht, wt), mode="area")
        return x.permute(0, 2, 3, 1).contiguous()

    hp, wp = x.shape[2], x.shape[3]
    df = _df_from_shapes(hp, wp, ht, wt)
    sigmas = dict(DEFAULT_S2_PSF_SIGMA_M_BY_BAND)
    x = degrade_hr_bchw(
        x,
        df,
        degradation,
        truncate=float(getattr(args, "s2_psf_truncate", DSEN2_PSF_TRUNCATE_DEFAULT)),
        native_gsd_m=float(getattr(args, "s2_native_gsd_m", 10.0)),
        sigma_m_by_band={
            "B02": float(getattr(args, "s2_psf_sigma_b02_m", sigmas["B02"])),
            "B03": float(getattr(args, "s2_psf_sigma_b03_m", sigmas["B03"])),
            "B04": float(getattr(args, "s2_psf_sigma_b04_m", sigmas["B04"])),
            "B08": float(getattr(args, "s2_psf_sigma_b08_m", sigmas["B08"])),
        },
    )
    if x.shape[2] != ht or x.shape[3] != wt:
        raise RuntimeError(
            f"lr_alignment [{degradation}]: output spatial {tuple(x.shape)} != target LR ({ht}, {wt}).",
        )
    return x.permute(0, 2, 3, 1).contiguous()
