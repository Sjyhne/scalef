"""Per-frame Sentinel-2 cloud masks for the reconstruction loss.

SCL classes match the national planner (``map_lr512_clear_counts``):
medium/high cloud and cirrus. Shadow is optional and off by default.
On-disk ``cloud_mask`` GeoTIFFs use ``1 = cloudy``.
"""

from __future__ import annotations

import numpy as np
import torch

# Sentinel-2 L2A SCL class ids.
SCL_NODATA = 0
SCL_SHADOW = 3
SCL_CLOUD_MED = 8
SCL_CLOUD_HIGH = 9
SCL_CIRRUS = 10
SCL_SNOW = 11


def score_scl_cell(
    scl: np.ndarray,
    *,
    max_cloud_frac: float,
    min_valid_frac: float,
    include_shadow: bool = False,
    max_snow_frac: float | None = 0.05,
) -> tuple[bool, float, float, float]:
    """Admit one LR cell from an on-disk SCL window.

    Returns ``(passed, cloud_frac, snow_frac, valid_frac)``. Fractions are
    NaN when the window has no valid pixels.
    """
    arr = np.asarray(scl)
    valid = arr != SCL_NODATA
    valid_frac = float(valid.mean()) if arr.size else 0.0
    if not valid.any():
        return False, float("nan"), float("nan"), 0.0
    cloudy = cloudy_from_scl(arr, include_shadow=include_shadow)
    snow = arr == SCL_SNOW
    cloud_frac = float(cloudy[valid].mean())
    snow_frac = float(snow[valid].mean())
    if valid_frac < float(min_valid_frac):
        return False, cloud_frac, snow_frac, valid_frac
    if cloud_frac > float(max_cloud_frac):
        return False, cloud_frac, snow_frac, valid_frac
    if max_snow_frac is not None and snow_frac > float(max_snow_frac):
        return False, cloud_frac, snow_frac, valid_frac
    return True, cloud_frac, snow_frac, valid_frac


def cloudy_from_scl(scl: np.ndarray, *, include_shadow: bool = False) -> np.ndarray:
    """Boolean cloudy mask from an SCL array (any shape)."""
    ids = {SCL_CLOUD_MED, SCL_CLOUD_HIGH, SCL_CIRRUS}
    if include_shadow:
        ids.add(SCL_SHADOW)
    return np.isin(np.asarray(scl), list(ids))


def clear_from_cloud_mask(mask: np.ndarray) -> np.ndarray:
    """``cloud_mask`` GeoTIFF convention: nonzero = cloudy."""
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[0]
    return arr <= 0


def apply_clear_to_holdout_masks(
    holdout_masks: list[torch.Tensor],
    clear: torch.Tensor,
) -> list[torch.Tensor]:
    """AND per-frame clear maps onto holdout train masks.

    ``holdout_masks[i]`` is ``[1,H,W,1]`` bool (True = train).
    ``clear`` is ``[N,H,W]`` bool (True = clear).
    """
    if clear.ndim != 3:
        raise ValueError(f"clear must be [N,H,W], got {tuple(clear.shape)}")
    if int(clear.shape[0]) != len(holdout_masks):
        raise ValueError(
            f"clear n_frames {int(clear.shape[0])} != holdout {len(holdout_masks)}"
        )
    out: list[torch.Tensor] = []
    for i, hold in enumerate(holdout_masks):
        c = clear[i].to(device=hold.device, dtype=torch.bool)
        while c.ndim < hold.ndim:
            if c.ndim == 2:
                c = c.unsqueeze(0)
            else:
                c = c.unsqueeze(-1)
        out.append(hold & c)
    return out


def read_scl_10m(item) -> tuple[np.ndarray, dict]:
    """Read SCL and nearest-neighbor upsample to the item's 10 m B04 grid."""
    import planetary_computer as pc
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    signed = pc.sign(item)
    if "SCL" not in signed.assets or "B04" not in signed.assets:
        raise RuntimeError(f"{item.id}: missing SCL or B04")

    with rasterio.open(signed.assets["B04"].href) as ref:
        dst_h, dst_w = int(ref.height), int(ref.width)
        dst_transform = ref.transform
        dst_crs = ref.crs
        profile = {
            "crs": dst_crs,
            "transform": dst_transform,
            "height": dst_h,
            "width": dst_w,
            "resolution_m": float(dst_transform.a),
        }

    with rasterio.open(signed.assets["SCL"].href) as src:
        scl_native = src.read(1)
        out = np.zeros((dst_h, dst_w), dtype=np.uint8)
        reproject(
            source=scl_native,
            destination=out,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
        )
    return out, profile


def reproject_scl_to_grid(
    scl: np.ndarray,
    src_profile: dict,
    *,
    dst_transform,
    dst_crs,
    height: int,
    width: int,
) -> np.ndarray:
    """Nearest-neighbor SCL onto an existing RGB grid."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    if (
        int(src_profile["height"]) == int(height)
        and int(src_profile["width"]) == int(width)
        and str(src_profile["crs"]) == str(dst_crs)
    ):
        return np.asarray(scl, dtype=np.uint8)
    out = np.zeros((height, width), dtype=np.uint8)
    reproject(
        source=np.asarray(scl, dtype=np.uint8),
        destination=out,
        src_transform=src_profile["transform"],
        src_crs=src_profile["crs"],
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.nearest,
    )
    return out


def frame_clear_fractions(clear: torch.Tensor | np.ndarray) -> list[float]:
    arr = clear.detach().cpu().numpy() if isinstance(clear, torch.Tensor) else np.asarray(clear)
    if arr.ndim != 3:
        raise ValueError(f"clear must be [N,H,W], got {tuple(arr.shape)}")
    return [float(arr[i].mean()) for i in range(arr.shape[0])]
