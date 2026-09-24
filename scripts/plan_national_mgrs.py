#!/usr/bin/env python3
"""Plan per-LR512 fetch dates for one MGRS granule (national 2025 recipe).

Scores SCL on every unique day in the window (default jul_pm45), then for each
mainland LR512 cell ranks passing days by distance to 15 July and keeps up to
16. Thin stacks (1–5 days) are kept. QA rasters record observation depth, not
a certainty score.

Example::

    python scripts/plan_national_mgrs.py --mgrs 32VNM
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.country_config import configured_path, load_country_config  # noqa: E402
from scripts.land_mask_lr512 import DEFAULT_LAND_MASK, load_land_paths  # noqa: E402
from scripts.map_lr512_clear_counts import (  # noqa: E402
    DEFAULT_SIDE,
    _one_per_day,
    _read_scl_10m,
    _search_mgrs,
    date_window,
    score_grid_stats,
)
from scripts.national_cell_queue import (  # noqa: E402
    DEFAULT_MAX_FRAMES,
    JUL_PM45,
    NATIONAL_CENTER,
    bbox_wgs84_from_profile,
    build_cell_plan,
    land_cell_mask,
    write_plan_artifacts,
)


def _yyyy_mm_dd(s: str) -> str:
    return datetime.fromisoformat(str(s)[:10]).strftime("%Y-%m-%d")


def _save_cube(path: Path, *, days, passed, cloud, snow, profile: dict, stac_ids=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = ["" if v is None else str(v) for v in (stac_ids or [""] * len(days))]
    np.savez_compressed(
        path,
        days=np.asarray(days),
        passed=passed.astype(np.bool_),
        cloud=cloud.astype(np.float32),
        snow=snow.astype(np.float32),
        stac_ids=np.asarray(ids),
        profile_json=np.asarray(json.dumps(_jsonable_profile(profile))),
    )


def _jsonable_profile(profile: dict) -> dict:
    transform = profile["transform"]
    return {
        "crs": str(profile["crs"]),
        "transform": list(transform)[:6],
        "height": int(profile["height"]),
        "width": int(profile["width"]),
        "resolution_m": float(profile["resolution_m"]),
    }


def _profile_from_cube(z) -> dict:
    import rasterio
    from rasterio.transform import Affine

    raw = json.loads(str(np.asarray(z["profile_json"]).reshape(-1)[0]))
    return {
        "crs": rasterio.crs.CRS.from_string(raw["crs"]),
        "transform": Affine(*raw["transform"]),
        "height": int(raw["height"]),
        "width": int(raw["width"]),
        "resolution_m": float(raw["resolution_m"]),
    }


def plan_mgrs(
    mgrs: str,
    *,
    datetime_range: str,
    center: str,
    side: int,
    max_cloud_frac: float,
    min_valid_frac: float,
    include_shadow: bool,
    max_snow_frac: float | None,
    max_stac_items: int,
    max_stac_cloud: float,
    max_frames: int,
    out_dir: Path,
    mainland_only: bool,
    land_mask: Path | None,
    min_land_frac: float,
) -> dict:
    mgrs_u = str(mgrs).upper().lstrip("T")
    cube_path = out_dir / f"{mgrs_u}_score_cube.npz"
    print(f"\n===== plan {mgrs_u} {datetime_range} =====", flush=True)

    scored_days: list[str] = []
    scored_ids: list[str] = []
    passed_list: list[np.ndarray] = []
    cloud_list: list[np.ndarray] = []
    snow_list: list[np.ndarray] = []
    profile = None

    if cube_path.is_file():
        z = np.load(cube_path, allow_pickle=True)
        scored_days = [str(d) for d in z["days"].tolist()]
        if "stac_ids" in z.files:
            scored_ids = [str(s) for s in z["stac_ids"].tolist()]
        else:
            scored_ids = [""] * len(scored_days)
        passed_arr = z["passed"]
        cloud_arr = z["cloud"]
        snow_arr = z["snow"]
        passed_list = [passed_arr[i] for i in range(passed_arr.shape[0])]
        cloud_list = [cloud_arr[i] for i in range(cloud_arr.shape[0])]
        snow_list = [snow_arr[i] for i in range(snow_arr.shape[0])]
        if "profile_json" in z.files:
            profile = _profile_from_cube(z)
        print(f"resume cube: {len(scored_days)} days from {cube_path}", flush=True)

    items = _search_mgrs(mgrs_u, datetime_range, max_items=max_stac_items)
    if max_stac_cloud < 100:
        items = [
            it
            for it in items
            if float(it.properties.get("eo:cloud_cover", 100.0)) < max_stac_cloud
        ]
    days_items = _one_per_day(items)
    print(f"{len(items)} STAC items → {len(days_items)} unique days", flush=True)

    scored_set = set(scored_days)
    for i, it in enumerate(days_items):
        day = it.datetime.astimezone(timezone.utc).strftime("%Y-%m-%d")
        if day in scored_set:
            continue
        try:
            scl, profile = _read_scl_10m(it)
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {day} {it.id}: {exc}", flush=True)
            continue
        passed, cloud, snow = score_grid_stats(
            scl,
            side=side,
            max_cloud_frac=max_cloud_frac,
            min_valid_frac=min_valid_frac,
            include_shadow=include_shadow,
            max_snow_frac=max_snow_frac,
        )
        scored_days.append(day)
        scored_ids.append(str(it.id))
        scored_set.add(day)
        passed_list.append(passed)
        cloud_list.append(cloud)
        snow_list.append(snow)
        print(
            f"  {i + 1}/{len(days_items)} {day} {it.id} pass_cells={int(passed.sum())}",
            flush=True,
        )
        _save_cube(
            cube_path,
            days=scored_days,
            passed=np.stack(passed_list, axis=0),
            cloud=np.stack(cloud_list, axis=0),
            snow=np.stack(snow_list, axis=0),
            profile=profile,
            stac_ids=scored_ids,
        )

    if not passed_list:
        raise SystemExit(f"{mgrs_u}: no SCL days scored")

    if profile is None:
        raise SystemExit(f"{mgrs_u}: scored days but no raster profile (corrupt cube?)")

    passed = np.stack(passed_list, axis=0)
    cloud = np.stack(cloud_list, axis=0)
    snow = np.stack(snow_list, axis=0)
    n_y, n_x = passed.shape[1], passed.shape[2]

    land_paths = None
    if mainland_only or land_mask is not None:
        mask_path = land_mask if land_mask is not None else DEFAULT_LAND_MASK
        if not Path(mask_path).is_absolute():
            mask_path = ROOT / mask_path
        land_paths = load_land_paths(mask_path)
    land = land_cell_mask(
        profile["transform"],
        profile["crs"],
        side=side,
        n_y=n_y,
        n_x=n_x,
        land_paths=land_paths,
        min_land_frac=min_land_frac,
    )
    print(f"land cells {int(land.sum())}/{land.size}", flush=True)

    plan = build_cell_plan(
        scored_days,
        passed,
        cloud,
        snow,
        land,
        center=center,
        max_frames=max_frames,
        side=side,
    )
    plan["mgrs_tile"] = mgrs_u
    plan["date_range"] = datetime_range
    plan["rules"] = {
        "max_cloud_frac": float(max_cloud_frac),
        "min_valid_frac": float(min_valid_frac),
        "include_shadow": bool(include_shadow),
        "max_snow_frac": None if max_snow_frac is None else float(max_snow_frac),
        "mainland_only": bool(mainland_only or land_mask is not None),
        "min_land_frac": float(min_land_frac),
        "cloud_source": "SCL",
        "ranking": "closest_to_center_cap_max_frames_keep_thin",
    }
    plan["bbox_wgs84"] = bbox_wgs84_from_profile(profile)
    plan["n_days_scored"] = len(scored_days)
    plan["days_scored"] = scored_days
    plan["days_scored_items"] = [
        {"date": d, "stac_id": sid}
        for d, sid in zip(scored_days, scored_ids)
        if sid
    ]
    json_plan = write_plan_artifacts(plan, profile, out_dir, mgrs_u)
    print(
        f"{mgrs_u}: land={plan['n_land_cells']} with_frames={plan['n_cells_with_frames']} "
        f"no_pass={plan['n_cells_no_pass']} union_dates={plan['n_union_dates']} "
        f"n_frames={plan['n_frames_min']}–{plan['n_frames_max']} "
        f"→ {out_dir / 'plan.json'}",
        flush=True,
    )
    return json_plan


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mgrs", type=str, required=True)
    p.add_argument(
        "--country-config",
        type=Path,
        default=None,
        help="Country config controlling default plan root and land outline.",
    )
    p.add_argument("--date", type=str, default=NATIONAL_CENTER.isoformat())
    p.add_argument("--start-date", type=str, default=JUL_PM45[0])
    p.add_argument("--end-date", type=str, default=JUL_PM45[1])
    p.add_argument("--days-before", type=int, default=None)
    p.add_argument("--days-after", type=int, default=None)
    p.add_argument("--side", type=int, default=DEFAULT_SIDE)
    p.add_argument("--max-cloud-frac", type=float, default=0.15)
    p.add_argument("--min-valid-frac", type=float, default=0.85)
    p.add_argument("--include-shadow", action="store_true")
    p.add_argument("--max-snow-frac", type=float, default=0.05)
    p.add_argument("--max-stac-items", type=int, default=200)
    p.add_argument("--max-stac-cloud", type=float, default=100.0)
    p.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: production/national_2025/plans/{MGRS}",
    )
    p.add_argument("--mainland-only", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--land-mask", type=Path, default=None)
    p.add_argument("--min-land-frac", type=float, default=0.0)
    return p


def main() -> None:
    args = build_parser().parse_args()
    mgrs = str(args.mgrs).upper().lstrip("T")
    country_config = (
        load_country_config(args.country_config) if args.country_config is not None else None
    )
    center = _yyyy_mm_dd(args.date)
    if args.days_before is not None or args.days_after is not None:
        before = 45 if args.days_before is None else int(args.days_before)
        after = 45 if args.days_after is None else int(args.days_after)
        window = date_window(center, before, after)
    else:
        start = _yyyy_mm_dd(args.start_date)
        end = _yyyy_mm_dd(args.end_date)
        window = f"{start}/{end}"
    out_dir = args.out_dir
    if out_dir is None:
        if country_config is not None:
            out_dir = configured_path(country_config, "paths", "plan_root") / mgrs
        else:
            out_dir = ROOT / "production" / "national_2025" / "plans" / mgrs
    elif not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    max_snow = None if float(args.max_snow_frac) < 0 else float(args.max_snow_frac)
    t0 = time.monotonic()
    land_mask = args.land_mask
    if land_mask is None and country_config is not None:
        land_mask = configured_path(country_config, "inputs", "land_outline_geojson")
    plan_mgrs(
        mgrs,
        datetime_range=window,
        center=center,
        side=int(args.side),
        max_cloud_frac=float(args.max_cloud_frac),
        min_valid_frac=float(args.min_valid_frac),
        include_shadow=bool(args.include_shadow),
        max_snow_frac=max_snow,
        max_stac_items=int(args.max_stac_items),
        max_stac_cloud=float(args.max_stac_cloud),
        max_frames=int(args.max_frames),
        out_dir=out_dir,
        mainland_only=bool(args.mainland_only),
        land_mask=land_mask,
        min_land_frac=float(args.min_land_frac),
    )
    print(f"elapsed_s={time.monotonic() - t0:.1f}", flush=True)


if __name__ == "__main__":
    main()
