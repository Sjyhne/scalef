#!/usr/bin/env python3
"""Stitch per-MGRS LR512 clear-count GeoTIFFs into one Norway WGS84 map.

Reprojects each ``*_lr512_clear_counts.tif`` to EPSG:4326 and merges with
``method=max`` in overlaps (MGRS tiles overlap at edges).

Example
-------
    python scripts/stitch_lr512_clear_counts.py \\
      --in-dir production/cloud_availability/lr512_norway_july2025_pm45 \\
      --out production/cloud_availability/lr512_norway_july2025_pm45/norway_clear_counts.tif
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject

ROOT = Path(__file__).resolve().parent.parent


def _reproject_to_wgs84(src_path: Path, dest_path: Path) -> Path:
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, "EPSG:4326", src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update(
            {
                "crs": "EPSG:4326",
                "transform": transform,
                "width": width,
                "height": height,
                "compress": "deflate",
                "nodata": 65535,
            }
        )
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(dest_path, "w", **kwargs) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs="EPSG:4326",
                resampling=Resampling.nearest,
                src_nodata=src.nodata,
                dst_nodata=65535,
            )
    return dest_path


def stitch(
    in_dir: Path,
    out_tif: Path,
    *,
    min_clear: int,
    mgrs_allow: set[str] | None = None,
) -> dict:
    sources = sorted(in_dir.glob("*_lr512_clear_counts.tif"))
    if mgrs_allow is not None:
        filtered = []
        for p in sources:
            tid = p.name.split("_")[0]
            if tid in mgrs_allow:
                filtered.append(p)
        skipped = len(sources) - len(filtered)
        if skipped:
            print(f"  stitch filter: using {len(filtered)}/{len(sources)} tiles (mainland list)", flush=True)
        sources = filtered
    if not sources:
        raise SystemExit(f"no *_lr512_clear_counts.tif in {in_dir}")

    tmp_dir = in_dir / "_wgs84_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    warped = []
    for i, src in enumerate(sources):
        dest = tmp_dir / src.name
        if not dest.is_file() or dest.stat().st_mtime < src.stat().st_mtime:
            print(f"  reproject [{i + 1}/{len(sources)}] {src.name}", flush=True)
            _reproject_to_wgs84(src, dest)
        warped.append(dest)

    datasets = [rasterio.open(p) for p in warped]
    try:
        mosaic, transform = merge(datasets, nodata=65535, method="max")
        crs = datasets[0].crs
    finally:
        for ds in datasets:
            ds.close()

    # merge max with nodata can leave 65535; clamp display nodata
    arr = mosaic[0]
    arr = np.where(arr == 65535, 0, arr).astype(np.uint16)

    out_tif.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": arr.shape[0],
        "width": arr.shape[1],
        "count": 1,
        "dtype": "uint16",
        "crs": crs,
        "transform": transform,
        "compress": "deflate",
        "nodata": 0,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(arr, 1)
        dst.set_band_description(1, "n_clear_days")

    # PNG overview: lon/lat with geographic aspect (1° lon ≠ 1° lon ground distance).
    # Equal-aspect lon/lat made Norway look too wide / overly diagonal. Metric CRS
    # (UTM33) is fine for one zone but spans 31–36 here; lon/lat + cos(φ) is clearer.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    png = out_tif.with_suffix(".png")
    west = float(transform.c)
    east = float(transform.c + transform.a * arr.shape[1])
    north = float(transform.f)
    south = float(transform.f + transform.e * arr.shape[0])
    mid_lat = 0.5 * (south + north)

    fig, ax = plt.subplots(figsize=(5.5, 11))
    show = np.ma.masked_where(arr == 0, arr)
    im = ax.imshow(
        show,
        cmap="viridis",
        vmin=0,
        vmax=max(int(arr.max()), min_clear),
        interpolation="nearest",
        extent=[west, east, south, north],
        origin="upper",
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.55, pad=0.02)
    cbar.set_label("clear LR512 days")
    n_ok = int((arr >= min_clear).sum())
    n_land = int((arr > 0).sum())
    ax.set_title(
        f"Norway LR512 clear days (SCL rules)\n"
        f"≥{min_clear}: {n_ok}/{n_land} cells with data  |  {len(sources)} MGRS\n"
        f"lon/lat, aspect=1/cos({mid_lat:.0f}°)  (MGRS tiles = rectangles)"
    )
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_aspect(1.0 / np.cos(np.deg2rad(mid_lat)), adjustable="box")

    # Sanity markers (georef check)
    for name, lon, lat in (
        ("Bergen", 5.32, 60.39),
        ("Oslo", 10.75, 59.91),
        ("Tromsø", 18.96, 69.65),
        ("Kirkenes", 30.04, 69.73),
    ):
        ax.plot(lon, lat, "r.", markersize=6)
        ax.annotate(name, (lon, lat), color="white", fontsize=7, xytext=(3, 3), textcoords="offset points")

    fig.tight_layout()
    fig.savefig(png, dpi=180)
    plt.close(fig)

    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_sources": len(sources),
        "sources": [p.name for p in sources],
        "out_tif": str(out_tif),
        "out_png": str(png),
        "crs": "EPSG:4326",
        "merge_method": "max",
        "min_clear": min_clear,
        "cells_ge_min_clear": n_ok,
        "cells_with_data": n_land,
        "count_max": int(arr.max()),
        "count_mean_nonzero": float(arr[arr > 0].mean()) if n_land else 0.0,
    }
    out_tif.with_suffix(out_tif.suffix + ".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Wrote {out_tif} and {png}", flush=True)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--in-dir",
        type=Path,
        default=ROOT / "production" / "cloud_availability" / "lr512_norway_july2025_pm45",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output GeoTIFF (default: <in-dir>/norway_clear_counts.tif)",
    )
    ap.add_argument("--min-clear", type=int, default=6)
    ap.add_argument(
        "--mgrs-file",
        type=Path,
        default=None,
        help="Optional JSON with mgrs:[] — only stitch those mainland tiles",
    )
    args = ap.parse_args()
    in_dir = args.in_dir if args.in_dir.is_absolute() else ROOT / args.in_dir
    out = args.out
    if out is None:
        out = in_dir / "norway_clear_counts.tif"
    elif not out.is_absolute():
        out = ROOT / out
    allow = None
    if args.mgrs_file is not None:
        mf = args.mgrs_file if args.mgrs_file.is_absolute() else ROOT / args.mgrs_file
        allow = set(json.loads(mf.read_text())["mgrs"])
    stitch(in_dir, out, min_clear=int(args.min_clear), mgrs_allow=allow)


if __name__ == "__main__":
    main()
