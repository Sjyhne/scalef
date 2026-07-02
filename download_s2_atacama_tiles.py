#!/usr/bin/env python3
"""Download Sentinel-2 L2A time series for all MGRS tiles covering the Atacama AOI.

Wraps :mod:`download_s2_tile_series` once per discovered tile. Each tile is written
under ``<output>/<MGRS>/`` with the same NPZ / mask / preview layout as the
single-tile downloader.

Examples::

    # Discover tiles in the default Atacama bbox and print the plan
    python download_s2_atacama_tiles.py \\
        --start-date 2025-06-01 --end-date 2025-09-30 \\
        --dry-run

    # Download one tile first (smoke test)
    python download_s2_atacama_tiles.py \\
        --start-date 2025-06-01 --end-date 2025-09-30 \\
        --tiles 19KDQ --max-scenes 1

    # Full Atacama batch (large: many tiles × many dates)
    python download_s2_atacama_tiles.py \\
        --start-date 2025-06-01 --end-date 2025-09-30 \\
        --max-cloud-cover 10 \\
        --output data/s2_atacama
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

from download_s2_earth_search import (
    _iso_range,
    _parse_bbox,
    _validate_yyyy_mm_dd,
    discover_mgrs_tiles_in_bbox,
    normalize_mgrs_tile,
)
from download_s2_tile_series import run_download

# west, south, east, north — northern Chile / core Atacama Desert
ATACAMA_BBOX: tuple[float, float, float, float] = (-70.75, -26.5, -67.25, -20.5)


def add_download_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--start-date", type=str, required=True)
    p.add_argument("--end-date", type=str, required=True)
    p.add_argument("--band-preset", type=str, default="rgb_nir")
    p.add_argument("--epsg", type=int, default=None)
    p.add_argument("--resolution-m", type=float, default=10.0)
    p.add_argument("--max-cloud-cover", type=float, default=10.0)
    p.add_argument(
        "--max-aoi-nodata-pct",
        type=float,
        default=None,
        help="See download_s2_tile_series.py (disabled by default for full tiles).",
    )
    p.add_argument("--max-scenes", type=int, default=None)
    p.add_argument("--auto-crop-size", type=int, default=0)
    p.add_argument("--coverage-gate-pct", type=float, default=20.0)
    p.add_argument("--bbox-margin", type=float, default=1.25)
    p.add_argument("--fill-max-days", type=float, default=21.0)
    p.add_argument("--preview-downsample", type=int, default=8)
    p.add_argument("--preview-gamma", type=float, default=1.0)
    p.add_argument(
        "--preview-stretch",
        choices=("scalar", "per_channel"),
        default="scalar",
    )


def build_tile_args(base: argparse.Namespace, tile: str, meta: dict) -> Namespace:
    out_dir = Path(base.output) / tile
    return Namespace(
        mgrs_tile=tile,
        center_lon=float(meta["center_lon"]),
        center_lat=float(meta["center_lat"]),
        start_date=base.start_date,
        end_date=base.end_date,
        output=out_dir,
        band_preset=base.band_preset,
        epsg=base.epsg,
        resolution_m=base.resolution_m,
        max_cloud_cover=base.max_cloud_cover,
        max_aoi_nodata_pct=base.max_aoi_nodata_pct,
        max_scenes=base.max_scenes,
        auto_crop_size=base.auto_crop_size,
        coverage_gate_pct=base.coverage_gate_pct,
        bbox_margin=base.bbox_margin,
        fill_max_days=base.fill_max_days,
        preview_downsample=base.preview_downsample,
        preview_gamma=base.preview_gamma,
        preview_stretch=base.preview_stretch,
        dry_run=False,
    )


def resolve_tiles(args: argparse.Namespace) -> dict[str, dict]:
    if args.tiles:
        tiles = {normalize_mgrs_tile(t): {"mgrs_tile": normalize_mgrs_tile(t)} for t in args.tiles}
        datetime_range = _iso_range(args.start_date, args.end_date)
        discovered = discover_mgrs_tiles_in_bbox(
            args.bbox,
            datetime_range=datetime_range,
            max_cloud_cover=float(args.max_cloud_cover),
        )
        for tile in list(tiles.keys()):
            if tile in discovered:
                tiles[tile] = discovered[tile]
            else:
                tiles[tile] = {
                    "mgrs_tile": tile,
                    "center_lon": float(args.default_center_lon),
                    "center_lat": float(args.default_center_lat),
                    "scene_count": None,
                    "sample_item_id": None,
                }
        return dict(sorted(tiles.items()))

    return discover_mgrs_tiles_in_bbox(
        args.bbox,
        datetime_range=_iso_range(args.start_date, args.end_date),
        max_cloud_cover=float(args.max_cloud_cover),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    batch = p.add_argument_group("batch")
    batch.add_argument(
        "--output",
        type=Path,
        default=Path("data/s2_atacama"),
        help="Root output directory; each tile -> <output>/<MGRS>/",
    )
    batch.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        default=ATACAMA_BBOX,
        help=f"WGS84 AOI for tile discovery (default: {ATACAMA_BBOX})",
    )
    batch.add_argument(
        "--tiles",
        nargs="*",
        default=None,
        help="Download only these MGRS tiles (skip discovery). Example: 19KDQ 19KCR",
    )
    batch.add_argument(
        "--default-center-lon",
        type=float,
        default=-69.3,
        help="Fallback center when --tiles is set but STAC discovery misses a tile.",
    )
    batch.add_argument(
        "--default-center-lat",
        type=float,
        default=-23.2,
        help="Fallback center when --tiles is set but STAC discovery misses a tile.",
    )
    batch.add_argument("--max-tiles", type=int, default=None, help="Limit number of tiles (testing).")
    batch.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip tiles that already have stac_download_manifest.json.",
    )
    batch.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep downloading remaining tiles if one fails.",
    )
    batch.add_argument("--dry-run", action="store_true")
    add_download_arguments(p)
    args = p.parse_args(argv)
    args.bbox = tuple(float(v) for v in args.bbox)
    _parse_bbox(args.bbox)
    _validate_yyyy_mm_dd(args.start_date, label="start")
    _validate_yyyy_mm_dd(args.end_date, label="end")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    tiles = resolve_tiles(args)
    if not tiles:
        raise SystemExit(f"No MGRS tiles found in bbox {args.bbox} for the requested date/cloud filters.")

    tile_items = list(tiles.items())
    if args.max_tiles is not None and args.max_tiles > 0:
        tile_items = tile_items[: int(args.max_tiles)]

    print(f"AOI bbox (W,S,E,N): {args.bbox}")
    print(f"Date range: {args.start_date} → {args.end_date}")
    print(f"Output root: {args.output}")
    print(f"Tiles to process: {len(tile_items)}")
    for tile, meta in tile_items:
        scenes = meta.get("scene_count")
        scene_s = str(scenes) if scenes is not None else "?"
        print(
            f"  {tile}  center=({meta['center_lon']:.3f}, {meta['center_lat']:.3f})  "
            f"stac_scenes≈{scene_s}"
        )

    plan = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "bbox_wgs84": list(args.bbox),
        "datetime_range": _iso_range(args.start_date, args.end_date),
        "max_cloud_cover": float(args.max_cloud_cover),
        "output_root": str(args.output),
        "tiles": [
            {
                "mgrs_tile": tile,
                "center_wgs84": [meta["center_lon"], meta["center_lat"]],
                "scene_count_stac": meta.get("scene_count"),
                "output_dir": str(Path(args.output) / tile),
            }
            for tile, meta in tile_items
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    plan_path = args.output / "batch_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(f"Wrote {plan_path}")

    if args.dry_run:
        return 0

    results: list[dict] = []
    failures = 0
    for i, (tile, meta) in enumerate(tile_items, start=1):
        out_dir = Path(args.output) / tile
        manifest = out_dir / "stac_download_manifest.json"
        if args.skip_existing and manifest.is_file():
            print(f"\n[{i}/{len(tile_items)}] {tile} — skip (existing {manifest})")
            results.append({"mgrs_tile": tile, "status": "skipped", "output_dir": str(out_dir)})
            continue

        print(f"\n[{i}/{len(tile_items)}] {tile} → {out_dir}")
        t0 = time.perf_counter()
        tile_args = build_tile_args(args, tile, meta)
        try:
            run_download(tile_args)
            status = "ok"
        except Exception as exc:
            failures += 1
            status = "error"
            print(f"  ERROR: {exc}", file=sys.stderr)
            if not args.continue_on_error:
                traceback.print_exc()
                break
            traceback.print_exc()
        wall = time.perf_counter() - t0
        row = {
            "mgrs_tile": tile,
            "status": status,
            "output_dir": str(out_dir),
            "wall_seconds": wall,
        }
        if status == "ok" and manifest.is_file():
            with manifest.open("r", encoding="utf-8") as f:
                man = json.load(f)
            row["num_scenes"] = man.get("num_scenes")
        results.append(row)
        print(f"  {status} ({wall:.1f}s)")

    summary = {
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "tiles_requested": len(tile_items),
        "tiles_ok": sum(1 for r in results if r["status"] == "ok"),
        "tiles_skipped": sum(1 for r in results if r["status"] == "skipped"),
        "tiles_failed": sum(1 for r in results if r["status"] == "error"),
        "results": results,
    }
    summary_path = args.output / "batch_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {summary_path}")
    print(
        f"Done: ok={summary['tiles_ok']} skipped={summary['tiles_skipped']} "
        f"failed={summary['tiles_failed']}"
    )
    return 1 if failures and not args.continue_on_error else (1 if failures else 0)


if __name__ == "__main__":
    raise SystemExit(main())
