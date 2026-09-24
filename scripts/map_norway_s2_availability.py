#!/usr/bin/env python3
"""Precompute Sentinel-2 cloud-free availability over Norway (MGRS × date window).

Fast layer: Planetary Computer STAC ``eo:cloud_cover`` per MGRS tile (no download).
This is a planning map — scene-level cloud is coarser than OmniCloudMask AOI
fractions, but enough to see where a shared season can supply ≥6–8 clear dates.

Outputs under ``--out-dir`` (default ``production/cloud_availability/``):

* ``mgrs_availability.json`` — per-tile counts / dates
* ``mgrs_availability.csv``
* ``mgrs_availability.geojson`` — tile footprints (from STAC geometries)
* ``mgrs_availability_map.png`` — choropleth of clear-day counts

Example
-------
    python scripts/map_norway_s2_availability.py \\
      --date 2025-07-15 --days-before 45 --days-after 45 \\
      --cloud-lt 15 --min-clear 6
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fetch_s2_revisits import (  # noqa: E402
    L2A,
    catalog,
    item_mgrs,
)

# Mainland Norway (+ small buffer). UTM 32–35.
NORWAY_BBOX = [4.3, 57.8, 31.5, 71.4]


def _yyyy_mm_dd(s: str) -> str:
    return datetime.fromisoformat(str(s)[:10]).strftime("%Y-%m-%d")


def date_window(center: str, days_before: int, days_after: int) -> str:
    c = datetime.fromisoformat(_yyyy_mm_dd(center))
    a = (c - timedelta(days=int(days_before))).strftime("%Y-%m-%d")
    b = (c + timedelta(days=int(days_after))).strftime("%Y-%m-%d")
    return f"{a}/{b}"


def _bbox_tiles(
    west: float, south: float, east: float, north: float, *, step: float
) -> list[list[float]]:
    tiles = []
    lon = west
    while lon < east:
        lat = south
        lon2 = min(lon + step, east)
        while lat < north:
            lat2 = min(lat + step, north)
            tiles.append([lon, lat, lon2, lat2])
            lat = lat2
        lon = lon2
    return tiles


def _search_bbox(bbox: list[float], datetime_range: str, *, max_items: int) -> list:
    kwargs = {
        "collections": [L2A],
        "bbox": list(bbox),
        "datetime": datetime_range,
        "max_items": int(max_items),
    }
    last_err = None
    for attempt in range(5):
        try:
            return list(catalog().search(**kwargs).items())
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            wait = 5 * (attempt + 1)
            print(f"  STAC fail {bbox}: {exc}; retry {wait}s", flush=True)
            time.sleep(wait)
    print(f"  WARN giving up on {bbox}: {last_err}", flush=True)
    return []


def _item_day(it) -> str | None:
    if it.datetime is None:
        return None
    return it.datetime.astimezone(timezone.utc).strftime("%Y-%m-%d")


def _geom_union_bbox(geoms: list[dict]) -> list[float] | None:
    xs, ys = [], []
    for g in geoms:
        if not g:
            continue
        coords = g.get("coordinates")
        if g.get("type") == "Polygon":
            ring = coords[0]
        elif g.get("type") == "MultiPolygon":
            ring = coords[0][0]
        else:
            continue
        for x, y, *_ in ring:
            xs.append(x)
            ys.append(y)
    if not xs:
        return None
    return [min(xs), min(ys), max(xs), max(ys)]


def collect_items(
    bbox: list[float],
    datetime_range: str,
    *,
    step: float,
    max_items: int,
) -> dict[str, list]:
    """Return mgrs -> list of unique STAC items."""
    tiles = _bbox_tiles(*bbox, step=step)
    print(f"Querying {len(tiles)} bbox tiles over {bbox}  {datetime_range}", flush=True)
    by_id: dict[str, object] = {}
    for i, tb in enumerate(tiles):
        items = _search_bbox(tb, datetime_range, max_items=max_items)
        for it in items:
            by_id[it.id] = it
        print(f"  [{i + 1}/{len(tiles)}] {tb} → {len(items)} items (unique {len(by_id)})", flush=True)

    by_mgrs: dict[str, list] = defaultdict(list)
    for it in by_id.values():
        mgrs = item_mgrs(it)
        if mgrs:
            by_mgrs[mgrs].append(it)
    return dict(by_mgrs)


def summarize_mgrs(
    by_mgrs: dict[str, list],
    *,
    cloud_thresholds: list[float],
    one_per_day: bool,
) -> list[dict]:
    rows = []
    for mgrs, items in sorted(by_mgrs.items()):
        clouds = []
        days_by_thr: dict[float, set[str]] = {t: set() for t in cloud_thresholds}
        for it in items:
            cc = float(it.properties.get("eo:cloud_cover", 100.0))
            clouds.append(cc)
            day = _item_day(it)
            for thr in cloud_thresholds:
                if cc < thr and day:
                    days_by_thr[thr].add(day)

        # Prefer geometry from lowest-cloud item for footprint display.
        best = min(items, key=lambda it: float(it.properties.get("eo:cloud_cover", 100.0)))
        footprint = best.geometry

        row = {
            "mgrs_tile": mgrs,
            "n_scenes": len(items),
            "cloud_min": min(clouds) if clouds else None,
            "cloud_median": float(sorted(clouds)[len(clouds) // 2]) if clouds else None,
            "footprint": footprint,
            "bbox": _geom_union_bbox([footprint] if footprint else []),
        }
        for thr in cloud_thresholds:
            key = f"clear_days_lt{int(thr)}"
            if one_per_day:
                row[key] = len(days_by_thr[thr])
            else:
                row[key] = sum(
                    1
                    for it in items
                    if float(it.properties.get("eo:cloud_cover", 100.0)) < thr
                )
            row[f"clear_dates_lt{int(thr)}"] = sorted(days_by_thr[thr])
        rows.append(row)
    return rows


def write_outputs(
    rows: list[dict],
    out_dir: Path,
    *,
    meta: dict,
    color_key: str,
    min_clear: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    slim = []
    for r in rows:
        d = {k: v for k, v in r.items() if k not in ("footprint",)}
        slim.append(d)

    payload = {"meta": meta, "tiles": slim}
    (out_dir / "mgrs_availability.json").write_text(json.dumps(payload, indent=2) + "\n")

    # CSV
    clear_day_keys = sorted({k for r in slim for k in r if k.startswith("clear_days_")})
    fieldnames = [
        "mgrs_tile",
        "n_scenes",
        "cloud_min",
        "cloud_median",
        *clear_day_keys,
        "meets_min_clear",
        "bbox_west",
        "bbox_south",
        "bbox_east",
        "bbox_north",
    ]
    with (out_dir / "mgrs_availability.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in slim:
            bb = r.get("bbox") or [None, None, None, None]
            w.writerow(
                {
                    **r,
                    "meets_min_clear": int(r.get(color_key, 0) >= min_clear),
                    "bbox_west": bb[0],
                    "bbox_south": bb[1],
                    "bbox_east": bb[2],
                    "bbox_north": bb[3],
                }
            )

    # GeoJSON
    features = []
    for r in rows:
        if not r.get("footprint"):
            continue
        props = {k: v for k, v in r.items() if k not in ("footprint",) and not k.startswith("clear_dates_")}
        props["meets_min_clear"] = bool(r.get(color_key, 0) >= min_clear)
        features.append({"type": "Feature", "geometry": r["footprint"], "properties": props})
    gj = {"type": "FeatureCollection", "features": features}
    (out_dir / "mgrs_availability.geojson").write_text(json.dumps(gj) + "\n")

    _plot_map(rows, out_dir / "mgrs_availability_map.png", color_key=color_key, min_clear=min_clear, meta=meta)
    print(f"Wrote {out_dir}", flush=True)


def _plot_map(
    rows: list[dict],
    dest: Path,
    *,
    color_key: str,
    min_clear: int,
    meta: dict,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Polygon

    fig, ax = plt.subplots(figsize=(8, 10))
    patches, values = [], []
    for r in rows:
        g = r.get("footprint")
        if not g:
            continue
        if g["type"] == "Polygon":
            rings = [g["coordinates"][0]]
        elif g["type"] == "MultiPolygon":
            rings = [p[0] for p in g["coordinates"]]
        else:
            continue
        val = float(r.get(color_key) or 0)
        for ring in rings:
            patches.append(Polygon([(x, y) for x, y, *_ in ring], closed=True))
            values.append(val)

    if not patches:
        ax.text(0.5, 0.5, "no footprints", ha="center", transform=ax.transAxes)
    else:
        coll = PatchCollection(patches, cmap="viridis", edgecolor="k", linewidth=0.3)
        import numpy as np

        coll.set_array(np.asarray(values, dtype=float))
        ax.add_collection(coll)
        ax.autoscale_view()
        cbar = fig.colorbar(coll, ax=ax, shrink=0.7)
        cbar.set_label(f"{color_key} (min clear={min_clear})")
        n_ok = sum(1 for r in rows if (r.get(color_key) or 0) >= min_clear)
        ax.set_title(
            f"Norway S2 clear days  {meta.get('date_range')}\n"
            f"{color_key} ≥ {min_clear}: {n_ok}/{len(rows)} MGRS"
        )
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(dest, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", type=str, default="2025-07-15")
    ap.add_argument("--days-before", type=int, default=45)
    ap.add_argument("--days-after", type=int, default=45)
    ap.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("W", "S", "E", "N"),
        default=NORWAY_BBOX,
    )
    ap.add_argument("--bbox-step", type=float, default=3.0, help="Degrees per STAC subquery.")
    ap.add_argument("--max-items", type=int, default=400)
    ap.add_argument(
        "--cloud-thresholds",
        type=float,
        nargs="+",
        default=[10, 15, 20, 30, 50],
        help="eo:cloud_cover upper bounds for clear-day counts.",
    )
    ap.add_argument(
        "--cloud-lt",
        type=float,
        default=15.0,
        help="Primary threshold for the map color / min-clear check.",
    )
    ap.add_argument("--min-clear", type=int, default=6, help="Target clear days (floor).")
    ap.add_argument("--allow-same-day", action="store_true", help="Count scenes not unique days.")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "production" / "cloud_availability",
    )
    args = ap.parse_args()

    center = _yyyy_mm_dd(args.date)
    window = date_window(center, args.days_before, args.days_after)
    thresholds = sorted(set(float(t) for t in args.cloud_thresholds) | {float(args.cloud_lt)})

    by_mgrs = collect_items(
        list(args.bbox),
        window,
        step=float(args.bbox_step),
        max_items=int(args.max_items),
    )
    rows = summarize_mgrs(
        by_mgrs,
        cloud_thresholds=thresholds,
        one_per_day=not args.allow_same_day,
    )
    color_key = f"clear_days_lt{int(args.cloud_lt)}"
    meta = {
        "center_date": center,
        "date_range": window,
        "bbox": list(args.bbox),
        "cloud_metric": "eo:cloud_cover (STAC scene-level)",
        "cloud_lt": float(args.cloud_lt),
        "min_clear": int(args.min_clear),
        "one_per_day": not args.allow_same_day,
        "n_mgrs": len(rows),
        "note": (
            "Planning layer only. Scene eo:cloud_cover ≠ OmniCloudMask AOI fraction. "
            "Use to choose MGRS / widen windows before fetch; refine per-512 after masks exist."
        ),
    }
    write_outputs(
        rows,
        args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir,
        meta=meta,
        color_key=color_key,
        min_clear=int(args.min_clear),
    )

    n_ok = sum(1 for r in rows if (r.get(color_key) or 0) >= int(args.min_clear))
    print(f"{color_key} ≥ {args.min_clear}: {n_ok}/{len(rows)} MGRS tiles", flush=True)


if __name__ == "__main__":
    main()
