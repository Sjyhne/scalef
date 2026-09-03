"""Write georeferenced GeoTIFFs for QGIS inspection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _as_hwc01(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)
    if x.ndim == 3 and x.shape[0] in (1, 3) and x.shape[-1] not in (1, 3):
        x = np.transpose(x, (1, 2, 0))
    if x.ndim != 3 or x.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Expected HWC image, got shape {tuple(x.shape)}")
    return np.clip(x, 0.0, 1.0)


def write_rgb_geotiff(
    path: Path | str,
    rgb_hwc: np.ndarray,
    *,
    transform,
    crs,
    nodata: float | None = 0.0,
) -> Path:
    """Write float32 reflectance RGB GeoTIFF (CHW) with CRS + affine."""
    import rasterio
    from rasterio.crs import CRS

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = _as_hwc01(rgb_hwc)
    h, w, c = rgb.shape
    data = np.transpose(rgb, (2, 0, 1))
    profile = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": c,
        "dtype": "float32",
        "crs": CRS.from_user_input(crs) if crs is not None else None,
        "transform": transform,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    if nodata is not None:
        profile["nodata"] = float(nodata)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
        for i in range(1, c + 1):
            dst.set_band_description(i, ("R", "G", "B", "A")[i - 1] if i <= 4 else f"B{i}")
    return path


def export_qgis_layers(
    out_dir: Path | str,
    *,
    hr_gt_hwc: np.ndarray,
    sr_pred_hwc: np.ndarray,
    s2_bilinear_hwc: np.ndarray,
    dataset=None,
    geo_meta: dict[str, Any] | None = None,
    lr_hwc: np.ndarray | None = None,
) -> dict[str, str]:
    """Write HR GT / SR / S2-bilinear (+ optional LR) GeoTIFFs for QGIS.

    Uses the dataset S2 CRS and HR affine so layers stack correctly.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if geo_meta is None and dataset is not None and hasattr(dataset, "get_geo_meta"):
        geo_meta = dataset.get_geo_meta()
    geo_meta = geo_meta or {}
    crs = geo_meta.get("crs")
    hr_transform = geo_meta.get("hr_transform")
    lr_transform = geo_meta.get("lr_transform")
    if crs is None or hr_transform is None:
        raise ValueError(
            "Missing CRS/hr_transform on dataset; cannot write georeferenced GeoTIFFs."
        )

    written: dict[str, str] = {}
    pairs = [
        ("hr_gt.tif", hr_gt_hwc),
        ("sr_pred.tif", sr_pred_hwc),
        ("s2_bilinear.tif", s2_bilinear_hwc),
    ]
    for name, arr in pairs:
        p = write_rgb_geotiff(out_dir / name, arr, transform=hr_transform, crs=crs)
        written[name] = str(p)

    if lr_hwc is not None and lr_transform is not None:
        p = write_rgb_geotiff(out_dir / "s2_lr.tif", lr_hwc, transform=lr_transform, crs=crs)
        written["s2_lr.tif"] = str(p)

    meta = {
        "crs": str(crs),
        "hr_gsd_m": geo_meta.get("hr_gsd_m"),
        "native_gsd_m": geo_meta.get("native_gsd_m"),
        "df": geo_meta.get("df"),
        "files": written,
        "notes": (
            "Float32 reflectance ~[0,1]. hr_gt is harmonized (+ spatially aligned) NIB. "
            "Open in QGIS and set the same CRS; layers should overlay."
        ),
    }
    import json

    (out_dir / "qgis_layers.json").write_text(json.dumps(meta, indent=2) + "\n")
    written["qgis_layers.json"] = str(out_dir / "qgis_layers.json")
    return written
