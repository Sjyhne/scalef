#!/usr/bin/env python3
"""Per-LR512 clear-revisit counts under production cloud rules (pilot heatmap).

For each MGRS tile and date window, score every non-overlapping LR512 window:
how many unique days pass ``max_cloud_frac`` / ``min_valid_frac``.

Default cloud source is L2A **SCL** (Planetary Computer COG, windowed) with
classes matching a practical stand-in for OmniCloudMask::

    cloudy = SCL in {8,9,10}   # medium/high cloud + thin cirrus
    (+ 3 cloud shadow if --include-shadow)
    invalid = SCL == 0

This is the planning map you want: a color grid over the 512 tiling, not
scene-level STAC eo:cloud_cover. OmniCloudMask mode can be added later for
exact fetch parity (heavier).

Example
-------
    python scripts/map_lr512_clear_counts.py \\
      --mgrs 32VML 32VNM 32VLL \\
      --date 2025-07-15 --days-before 45 --days-after 45 \\
      --max-cloud-frac 0.15 --min-valid-frac 0.85 \\
      --out-dir production/cloud_availability/lr512_july2025_pm45
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fetch_s2_revisits import (  # noqa: E402
    L2A,
    catalog,
    item_mgrs,
)

# Sentinel-2 L2A SCL class ids.
SCL_NODATA = 0
SCL_SHADOW = 3
SCL_CLOUD_MED = 8
SCL_CLOUD_HIGH = 9
SCL_CIRRUS = 10
SCL_SNOW = 11

DEFAULT_SIDE = 512


def _yyyy_mm_dd(s: str) -> str:
    return datetime.fromisoformat(str(s)[:10]).strftime("%Y-%m-%d")


def date_window(center: str, days_before: int, days_after: int) -> str:
    c = datetime.fromisoformat(_yyyy_mm_dd(center))
    a = (c - timedelta(days=int(days_before))).strftime("%Y-%m-%d")
    b = (c + timedelta(days=int(days_after))).strftime("%Y-%m-%d")
    return f"{a}/{b}"


def _search_mgrs(mgrs: str, datetime_range: str, *, max_items: int) -> list:
    # Query a loose bbox from tile id is hard without mgrs lib; use intersects
    # via PC filter on s2:mgrs_tile when supported, else post-filter.
    kwargs = {
        "collections": [L2A],
        "datetime": datetime_range,
        "max_items": int(max_items),
        "query": {"s2:mgrs_tile": {"eq": mgrs}},
    }
    last_err = None
    for attempt in range(5):
        try:
            items = list(catalog().search(**kwargs).items())
            # Belt-and-suspenders post-filter.
            return [it for it in items if item_mgrs(it) == mgrs]
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            wait = 5 * (attempt + 1)
            print(f"  STAC {mgrs} fail: {exc}; retry {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"STAC search failed for {mgrs}: {last_err}")


def _one_per_day(items: list) -> list:
    best: dict[str, object] = {}
    for it in items:
        if it.datetime is None:
            continue
        day = it.datetime.astimezone(timezone.utc).strftime("%Y-%m-%d")
        cc = float(it.properties.get("eo:cloud_cover", 100.0))
        prev = best.get(day)
        if prev is None or cc < float(prev.properties.get("eo:cloud_cover", 100.0)):
            best[day] = it
    return [best[d] for d in sorted(best)]


def _sign(item):
    import planetary_computer as pc

    return pc.sign(item)


def _read_scl_10m(item) -> tuple[np.ndarray, dict]:
    """Read SCL and nearest-neighbor upsample to the 10 m grid of B04."""
    import planetary_computer as pc

    signed = _sign(item)
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


def _cloudy_mask(scl: np.ndarray, *, include_shadow: bool) -> np.ndarray:
    ids = {SCL_CLOUD_MED, SCL_CLOUD_HIGH, SCL_CIRRUS}
    if include_shadow:
        ids.add(SCL_SHADOW)
    return np.isin(scl, list(ids))


def score_grid_stats(
    scl: np.ndarray,
    *,
    side: int,
    max_cloud_frac: float,
    min_valid_frac: float,
    include_shadow: bool,
    max_snow_frac: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-LR cell pass mask plus cloud/snow fractions on valid pixels.

    Returns ``(passed, cloud_frac, snow_frac)`` each shaped ``(n_y, n_x)``.
    Fractions are NaN when the cell has no valid pixels.
    """
    h, w = scl.shape
    n_y, n_x = h // side, w // side
    cloudy = _cloudy_mask(scl, include_shadow=include_shadow)
    snow = scl == SCL_SNOW
    valid = scl != SCL_NODATA
    passed = np.zeros((n_y, n_x), dtype=bool)
    cloud_frac = np.full((n_y, n_x), np.nan, dtype=np.float32)
    snow_frac = np.full((n_y, n_x), np.nan, dtype=np.float32)
    for iy in range(n_y):
        for ix in range(n_x):
            r0, c0 = iy * side, ix * side
            v = valid[r0 : r0 + side, c0 : c0 + side]
            c = cloudy[r0 : r0 + side, c0 : c0 + side]
            s = snow[r0 : r0 + side, c0 : c0 + side]
            if v.any():
                cloud_frac[iy, ix] = float(c[v].mean())
                snow_frac[iy, ix] = float(s[v].mean())
            if float(v.mean()) < min_valid_frac:
                continue
            frac = float(cloud_frac[iy, ix]) if np.isfinite(cloud_frac[iy, ix]) else 1.0
            if frac > max_cloud_frac:
                continue
            if max_snow_frac is not None:
                sfrac = float(snow_frac[iy, ix]) if np.isfinite(snow_frac[iy, ix]) else 1.0
                if sfrac > max_snow_frac:
                    continue
            passed[iy, ix] = True
    return passed, cloud_frac, snow_frac


def score_grid(
    scl: np.ndarray,
    *,
    side: int,
    max_cloud_frac: float,
    min_valid_frac: float,
    include_shadow: bool,
    max_snow_frac: float | None = None,
) -> np.ndarray:
    """Return bool grid (n_y, n_x): True if this 512 window passes rules.

    When ``max_snow_frac`` is set, SCL class 11 (snow/ice) above that fraction
    fails the cell — needed for green-season national maps (default clear maps
    historically ignored snow).
    """
    passed, _, _ = score_grid_stats(
        scl,
        side=side,
        max_cloud_frac=max_cloud_frac,
        min_valid_frac=min_valid_frac,
        include_shadow=include_shadow,
        max_snow_frac=max_snow_frac,
    )
    return passed


def write_count_geotiff(
    counts: np.ndarray,
    profile: dict,
    side: int,
    dest: Path,
) -> None:
    """Write counts on a grid where each output pixel = one LR512 cell (side m)."""
    n_y, n_x = counts.shape
    res = float(profile["resolution_m"]) * side
    transform = profile["transform"]
    # Cell (0,0) covers [0:side, 0:side] in native 10 m pixels.
    west = transform.c
    north = transform.f
    out_transform = from_origin(west, north, res, res)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        dest,
        "w",
        driver="GTiff",
        height=n_y,
        width=n_x,
        count=1,
        dtype="uint16",
        crs=profile["crs"],
        transform=out_transform,
        compress="deflate",
        nodata=65535,
    ) as dst:
        dst.write(counts.astype(np.uint16), 1)
        dst.set_band_description(1, "n_clear_days")


def write_heatmap_png(counts: np.ndarray, dest: Path, *, title: str, min_clear: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 7))
    vmax = max(int(counts.max()), min_clear)
    im = ax.imshow(counts, cmap="viridis", vmin=0, vmax=vmax, interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("clear LR days (pass rules)")
    n_ok = int((counts >= min_clear).sum())
    ax.set_title(f"{title}\n≥{min_clear} clear: {n_ok}/{counts.size} cells")
    ax.set_xlabel("tile ix")
    ax.set_ylabel("tile iy")
    fig.tight_layout()
    fig.savefig(dest, dpi=160)
    plt.close(fig)


def process_mgrs(
    mgrs: str,
    *,
    datetime_range: str,
    side: int,
    max_cloud_frac: float,
    min_valid_frac: float,
    include_shadow: bool,
    max_snow_frac: float | None,
    max_stac_items: int,
    max_stac_cloud: float,
    min_clear: int,
    out_dir: Path,
    device: str,  # reserved for future OCM
) -> dict:
    print(f"\n===== {mgrs} {datetime_range} =====", flush=True)
    items = _search_mgrs(mgrs, datetime_range, max_items=max_stac_items)
    if max_stac_cloud < 100:
        items = [
            it
            for it in items
            if float(it.properties.get("eo:cloud_cover", 100.0)) < max_stac_cloud
        ]
    days = _one_per_day(items)
    print(f"{len(items)} STAC items → {len(days)} unique days", flush=True)

    counts = None
    profile = None
    used_days: list[str] = []
    for i, it in enumerate(days):
        day = it.datetime.astimezone(timezone.utc).strftime("%Y-%m-%d")
        try:
            scl, profile = _read_scl_10m(it)
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {day} {it.id}: {exc}", flush=True)
            continue
        passed = score_grid(
            scl,
            side=side,
            max_cloud_frac=max_cloud_frac,
            min_valid_frac=min_valid_frac,
            include_shadow=include_shadow,
            max_snow_frac=max_snow_frac,
        )
        if counts is None:
            counts = np.zeros(passed.shape, dtype=np.uint16)
        counts += passed.astype(np.uint16)
        n_pass = int(passed.sum())
        used_days.append(day)
        print(
            f"  [{i + 1}/{len(days)}] {day} eo:cloud={it.properties.get('eo:cloud_cover')} "
            f"cells_pass={n_pass}/{passed.size}",
            flush=True,
        )

    if counts is None or profile is None:
        return {"mgrs_tile": mgrs, "error": "no_scorable_days"}

    tag = f"{mgrs}_lr{side}"
    tif = out_dir / f"{tag}_clear_counts.tif"
    png = out_dir / f"{tag}_clear_counts.png"
    write_count_geotiff(counts, profile, side, tif)
    write_heatmap_png(
        counts,
        png,
        title=f"{mgrs}  {datetime_range}",
        min_clear=min_clear,
    )
    summary = {
        "mgrs_tile": mgrs,
        "date_range": datetime_range,
        "side": side,
        "max_cloud_frac": max_cloud_frac,
        "min_valid_frac": min_valid_frac,
        "include_shadow": include_shadow,
        "max_snow_frac": max_snow_frac,
        "cloud_source": "SCL (8/9/10[+3]; snow gated if max_snow_frac set)",
        "n_days_scored": len(used_days),
        "days": used_days,
        "grid": {"n_y": int(counts.shape[0]), "n_x": int(counts.shape[1])},
        "clear_count_min": int(counts.min()),
        "clear_count_max": int(counts.max()),
        "clear_count_mean": float(counts.mean()),
        "cells_ge_min_clear": int((counts >= min_clear).sum()),
        "cells_total": int(counts.size),
        "geotiff": str(tif.relative_to(ROOT)) if tif.is_relative_to(ROOT) else str(tif),
        "png": str(png.relative_to(ROOT)) if png.is_relative_to(ROOT) else str(png),
    }
    (out_dir / f"{tag}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"Wrote {tif.name}: mean={summary['clear_count_mean']:.1f} "
        f"≥{min_clear}:{summary['cells_ge_min_clear']}/{summary['cells_total']}",
        flush=True,
    )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--mgrs",
        nargs="+",
        default=None,
        help="MGRS tiles to score (default: July 3×3 pilot if --mgrs-file unset).",
    )
    ap.add_argument(
        "--mgrs-file",
        type=Path,
        default=None,
        help="JSON with {\"mgrs\": [\"32VNM\", ...]} (overrides --mgrs).",
    )
    ap.add_argument("--date", type=str, default="2025-07-15")
    ap.add_argument("--days-before", type=int, default=45)
    ap.add_argument("--days-after", type=int, default=45)
    ap.add_argument("--side", type=int, default=DEFAULT_SIDE)
    ap.add_argument("--max-cloud-frac", type=float, default=0.15)
    ap.add_argument("--min-valid-frac", type=float, default=0.85)
    ap.add_argument("--include-shadow", action="store_true")
    ap.add_argument(
        "--max-snow-frac",
        type=float,
        default=-1.0,
        help=(
            "Max SCL snow/ice (class 11) fraction inside an LR512 cell "
            "(-1=ignore snow, legacy; 0.05 recommended for green-season maps)."
        ),
    )
    ap.add_argument("--max-stac-items", type=int, default=200)
    ap.add_argument(
        "--max-stac-cloud",
        type=float,
        default=100.0,
        help="Optional STAC eo:cloud_cover prefilter before SCL scoring.",
    )
    ap.add_argument("--min-clear", type=int, default=6)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip MGRS that already have *_lr{side}_summary.json in --out-dir.",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "production" / "cloud_availability" / "lr512_july2025_pm45",
    )
    args = ap.parse_args()

    if args.mgrs_file is not None:
        mf = args.mgrs_file if args.mgrs_file.is_absolute() else ROOT / args.mgrs_file
        mgrs_list = list(json.loads(mf.read_text())["mgrs"])
    elif args.mgrs:
        mgrs_list = list(args.mgrs)
    else:
        mgrs_list = [
            "32VLM",
            "32VMM",
            "32VNM",
            "32VLL",
            "32VML",
            "32VNL",
            "32VLK",
            "32VMK",
            "32VNK",
        ]

    center = _yyyy_mm_dd(args.date)
    window = date_window(center, args.days_before, args.days_after)
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    max_snow = None if float(args.max_snow_frac) < 0 else float(args.max_snow_frac)

    results = []
    for mgrs in mgrs_list:
        mgrs_u = str(mgrs).upper().lstrip("T")
        summary_path = out_dir / f"{mgrs_u}_lr{int(args.side)}_summary.json"
        if args.skip_existing and summary_path.is_file():
            print(f"skip existing {mgrs_u}", flush=True)
            results.append(json.loads(summary_path.read_text()))
            continue
        try:
            results.append(
                process_mgrs(
                    mgrs_u,
                    datetime_range=window,
                    side=int(args.side),
                    max_cloud_frac=float(args.max_cloud_frac),
                    min_valid_frac=float(args.min_valid_frac),
                    include_shadow=bool(args.include_shadow),
                    max_snow_frac=max_snow,
                    max_stac_items=int(args.max_stac_items),
                    max_stac_cloud=float(args.max_stac_cloud),
                    min_clear=int(args.min_clear),
                    out_dir=out_dir,
                    device=args.device,
                )
            )
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {mgrs}: {exc}", flush=True)
            results.append({"mgrs_tile": mgrs, "error": str(exc)})

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "center_date": center,
        "date_range": window,
        "rules": {
            "side": int(args.side),
            "max_cloud_frac": float(args.max_cloud_frac),
            "min_valid_frac": float(args.min_valid_frac),
            "include_shadow": bool(args.include_shadow),
            "max_snow_frac": max_snow,
            "cloud_source": "SCL",
            "one_per_day": True,
            "min_clear": int(args.min_clear),
        },
        "results": results,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nBatch summary → {out_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
