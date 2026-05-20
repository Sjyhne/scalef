"""HR prediction → LR supervision alignment (area downsample or S2 PSF stack)."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F

from models.s2_psf_forward import S2_RGB_BAND_ORDER, get_s2_psf_forward


def default_lr_align_args() -> SimpleNamespace:
    """Minimal ``args`` namespace for area-only alignment (no S2 PSF)."""
    return SimpleNamespace(lr_degradation="area")


def align_prediction_hwc_to_target(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    args,
    device: torch.device,
) -> torch.Tensor:
    """Resize ``pred`` [B,H,W,C] to match ``target`` [B,Ht,Wt,C] (HR render vs LR supervision)."""
    if pred.shape[1:3] == target.shape[1:3]:
        return pred

    _, ht, wt, _ = target.shape
    x = pred.permute(0, 3, 1, 2).contiguous()

    if getattr(args, "lr_degradation", "area") != "s2_psf":
        x = F.interpolate(x, size=(ht, wt), mode="area")
        return x.permute(0, 2, 3, 1).contiguous()

    _, _, hp, wp = x.shape
    if min(hp, wp, ht, wt) <= 0:
        raise ValueError(
            f"lr_alignment [s2_psf]: invalid spatial sizes HR=({hp},{wp}) LR=({ht},{wt}).",
        )

    df_h = hp // ht
    df_w = wp // wt
    if df_h != df_w or hp != df_h * ht or wp != df_w * wt:
        raise ValueError(
            f"lr_alignment [s2_psf]: HR ({hp}x{wp}) must be an integer tile multiple of "
            f"LR ({ht}x{wt}); got df_h={df_h}, df_w={df_w}.",
        )

    df = int(df_h)
    c = x.shape[1]
    if c != 3:
        raise ValueError(
            f"lr_alignment [s2_psf]: expected 3 RGB channels, got C={c}.",
        )

    band_order = S2_RGB_BAND_ORDER
    mod = get_s2_psf_forward(
        scale_factor=df,
        sigma_m_by_band={
            "B02": float(args.s2_psf_sigma_b02_m),
            "B03": float(args.s2_psf_sigma_b03_m),
            "B04": float(args.s2_psf_sigma_b04_m),
            "B08": float(args.s2_psf_sigma_b08_m),
        },
        band_order=band_order,
        native_gsd_m=float(args.s2_native_gsd_m),
        truncate=float(args.s2_psf_truncate),
        device=device,
    )
    x = mod(x)
    if x.shape[2] != ht or x.shape[3] != wt:
        raise RuntimeError(
            f"lr_alignment [s2_psf]: PSF output spatial {tuple(x.shape)} != target LR ({ht}, {wt}).",
        )
    return x.permute(0, 2, 3, 1).contiguous()
