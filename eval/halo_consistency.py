"""Train-time halo: expand a cell AOI into a frozen neighbor SR and match it.

The neighbor is a GeoTIFF (already-fitted ``sr_pred.tif``), not a second INR.
Coordinates stay in the current field's unit square; only the AOI grows.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from eval.lr_holdout import elementwise_recon, masked_mean


def expand_aoi_for_halo(
    aoi: dict,
    *,
    west_px: int = 0,
    north_px: int = 0,
) -> dict:
    """Grow an LR ``aoi_window`` westward / northward. Clamps at the raster origin."""
    col0 = int(aoi["col_off"])
    row0 = int(aoi["row_off"])
    width = int(aoi["width"])
    height = int(aoi["height"])
    west = max(0, int(west_px))
    north = max(0, int(north_px))
    take_w = min(west, col0)
    take_n = min(north, row0)
    return {
        "col_off": col0 - take_w,
        "row_off": row0 - take_n,
        "width": width + take_w,
        "height": height + take_n,
    }


def halo_masks_lr(
    lr_h: int,
    lr_w: int,
    *,
    west_px: int = 0,
    north_px: int = 0,
) -> np.ndarray:
    """True on the expanded west/north LR strips."""
    mask = np.zeros((lr_h, lr_w), dtype=bool)
    west = max(0, min(int(west_px), lr_w))
    north = max(0, min(int(north_px), lr_h))
    if west:
        mask[:, :west] = True
    if north:
        mask[:north, :] = True
    return mask


def load_neighbor_sr_on_grid(
    sr_path: Path | str,
    *,
    dst_transform,
    dst_crs,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Warp neighbor SR onto the current HR grid. Returns ``(rgb[H,W,3], valid[H,W])``."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    path = Path(sr_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    rgb = np.zeros((3, height, width), dtype=np.float32)
    valid = np.zeros((height, width), dtype=np.float32)
    with rasterio.open(path) as src:
        src_valid = np.any(src.read() > 0, axis=0).astype(np.float32)
        for i in range(min(3, src.count)):
            reproject(
                source=src.read(i + 1),
                destination=rgb[i],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
            )
        reproject(
            source=src_valid,
            destination=valid,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
        )
    return np.clip(np.transpose(rgb, (1, 2, 0)), 0.0, 1.0), valid > 0.5


def attach_halo_to_dataset(dataset, args) -> None:
    """Load neighbor SR and store halo target/mask on ``dataset`` (no-op if unset)."""
    sr_path = getattr(args, "halo_sr", None)
    west = int(getattr(args, "halo_lr_px_west", 0) or 0)
    north = int(getattr(args, "halo_lr_px_north", 0) or 0)
    dataset.halo_target = None
    dataset.halo_mask_hr = None
    dataset.halo_lr_px_west = west
    dataset.halo_lr_px_north = north
    if not sr_path or (west <= 0 and north <= 0):
        return
    geo = dataset.get_geo_meta()
    hr_h = int(dataset.original_hr.shape[0])
    hr_w = int(dataset.original_hr.shape[1])
    rgb, valid = load_neighbor_sr_on_grid(
        sr_path,
        dst_transform=geo["hr_transform"],
        dst_crs=geo["crs"],
        height=hr_h,
        width=hr_w,
    )
    df = int(getattr(dataset, "df", 4) or 4)
    lr_mask = halo_masks_lr(
        int(dataset.lr_height),
        int(dataset.lr_width),
        west_px=west,
        north_px=north,
    )
    hr_mask = np.repeat(np.repeat(lr_mask, df, axis=0), df, axis=1)[:hr_h, :hr_w]
    hr_mask = hr_mask & valid
    dataset.halo_target = torch.as_tensor(
        np.ascontiguousarray(rgb), dtype=torch.float32, device=dataset.device
    )
    dataset.halo_mask_hr = torch.as_tensor(hr_mask, dtype=torch.bool, device=dataset.device)
    n = int(hr_mask.sum())
    print(
        f"halo: neighbor {sr_path}  west={west} north={north}  "
        f"HR overlap {n} px ({100.0 * n / max(hr_h * hr_w, 1):.1f}% of field)",
        flush=True,
    )
    if n < 16:
        print("WARN: halo overlap is almost empty; check --halo_sr footprint", flush=True)


def sample_halo_batch(
    dataset,
    *,
    device: torch.device,
    chunk_lr: int = 128,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Random halo window: HR coords ``[1,h,w,2]`` and reflectance target ``[1,h,w,3]``."""
    mask = getattr(dataset, "halo_mask_hr", None)
    target = getattr(dataset, "halo_target", None)
    if mask is None or target is None or not bool(mask.any()):
        return None
    coords = dataset.get_hr_coordinates().to(device)
    hr_h, hr_w = int(coords.shape[0]), int(coords.shape[1])
    df = max(1, int(getattr(dataset, "df", 4) or 4))
    chunk = max(df, int(chunk_lr) * df)
    west = int(getattr(dataset, "halo_lr_px_west", 0) or 0) * df
    north = int(getattr(dataset, "halo_lr_px_north", 0) or 0) * df

    strips: list[tuple[int, int, int, int]] = []
    if west > 0:
        strips.append((0, hr_h, 0, min(west, hr_w)))
    if north > 0:
        strips.append((0, min(north, hr_h), 0, hr_w))
    if not strips:
        return None

    g = generator
    pick = int(torch.randint(0, len(strips), (1,), generator=g).item()) if len(strips) > 1 else 0
    r0, r1, c0, c1 = strips[pick]
    if r1 - r0 > chunk:
        extra = r1 - r0 - chunk
        off = int(torch.randint(0, extra + 1, (1,), generator=g).item()) if extra > 0 else 0
        r0, r1 = r0 + off, r0 + off + chunk
    if c1 - c0 > chunk:
        extra = c1 - c0 - chunk
        off = int(torch.randint(0, extra + 1, (1,), generator=g).item()) if extra > 0 else 0
        c0, c1 = c0 + off, c0 + off + chunk

    m = mask[r0:r1, c0:c1]
    if not bool(m.any()):
        return None
    return (
        coords[r0:r1, c0:c1].unsqueeze(0).to(device),
        target[r0:r1, c0:c1].unsqueeze(0).to(device),
        m.to(device),
    )


def _halo_mask_nchw(mask_hw: torch.Tensor, pred_hwc: torch.Tensor) -> torch.Tensor:
    hold = mask_hw.to(device=pred_hwc.device, dtype=torch.bool)
    if hold.ndim == 2:
        hold = hold.view(1, hold.shape[0], hold.shape[1], 1)
    return hold


def halo_consistency_loss(
    pred_hwc: torch.Tensor,
    target_hwc: torch.Tensor,
    mask_hw: torch.Tensor,
    *,
    recon_loss: str = "charbonnier",
    charbonnier_eps: float = 0.01,
    mode: str = "full",
    lf_hr_px: int = 16,
) -> torch.Tensor:
    """Compare destandardized RGB to neighbor reflectance on the halo mask.

    ``full``: per-pixel recon (legacy). ``lowfreq``: masked average-pool then
    recon, so the neighbor only donates gauge/texture at ~lf_hr_px. ``mean``:
    one RGB mean over the overlap (pure DC / colour lock).
    """
    hold = _halo_mask_nchw(mask_hw, pred_hwc)
    mode_n = str(mode or "full").strip().lower()
    if mode_n == "mean":
        w = hold.to(dtype=pred_hwc.dtype)
        denom = w.sum().clamp_min(1.0)
        pred_mu = (pred_hwc * w).sum(dim=(1, 2)) / denom
        tgt_mu = (target_hwc * w).sum(dim=(1, 2)) / denom
        return (pred_mu - tgt_mu).abs().mean()
    if mode_n == "lowfreq":
        import torch.nn.functional as F

        k = max(2, int(lf_hr_px))
        pred_nchw = pred_hwc.permute(0, 3, 1, 2)
        tgt_nchw = target_hwc.permute(0, 3, 1, 2)
        w = hold.permute(0, 3, 1, 2).to(dtype=pred_hwc.dtype)
        h, ww = pred_nchw.shape[-2], pred_nchw.shape[-1]
        h0, w0 = (h // k) * k, (ww // k) * k
        if h0 < k or w0 < k:
            mode_n = "full"
        else:
            pred_nchw = pred_nchw[..., :h0, :w0]
            tgt_nchw = tgt_nchw[..., :h0, :w0]
            w = w[..., :h0, :w0]
            w_pool = F.avg_pool2d(w, k, k)
            pred_pool = F.avg_pool2d(pred_nchw * w, k, k) / w_pool.clamp_min(1e-6)
            tgt_pool = F.avg_pool2d(tgt_nchw * w, k, k) / w_pool.clamp_min(1e-6)
            valid = w_pool > 0.25
            pred_h = pred_pool.permute(0, 2, 3, 1)
            tgt_h = tgt_pool.permute(0, 2, 3, 1)
            elem = elementwise_recon(
                recon_loss, pred_h, tgt_h, charbonnier_eps=charbonnier_eps
            )
            return masked_mean(elem, valid.permute(0, 2, 3, 1))
    elem = elementwise_recon(
        recon_loss, pred_hwc, target_hwc, charbonnier_eps=charbonnier_eps
    )
    return masked_mean(elem, hold)


def parse_halo_sides(sides: str | None, halo_lr_px: int) -> tuple[int, int]:
    """Return ``(west_px, north_px)`` from ``west``, ``north``, or ``west,north``."""
    px = max(0, int(halo_lr_px))
    if not sides or px <= 0:
        return 0, 0
    tokens = {t.strip().lower() for t in str(sides).split(",") if t.strip()}
    unknown = tokens - {"west", "north"}
    if unknown:
        raise ValueError(f"unknown halo sides {sorted(unknown)}; use west and/or north")
    return (px if "west" in tokens else 0, px if "north" in tokens else 0)
