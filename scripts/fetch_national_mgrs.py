#!/usr/bin/env python3
"""Plan + fetch one MGRS granule for the 2025 national recipe.

Does not overwrite named-site stacks under data/s2_revisits/<city>/. Writes to
data/s2_revisits/national_2025/{MGRS}/.

    python scripts/fetch_national_mgrs.py --mgrs 32VNM --plan-only
    python scripts/fetch_national_mgrs.py --mgrs 32VNM
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.country_config import configured_path, load_country_config  # noqa: E402
from scripts.national_cell_queue import JUL_PM45, NATIONAL_CENTER  # noqa: E402
from scripts.plan_national_mgrs import plan_mgrs  # noqa: E402


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    env = dict(**__import__("os").environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    subprocess.run(cmd, cwd=ROOT, check=True, env=env)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mgrs", type=str, required=True)
    p.add_argument(
        "--country-config",
        type=Path,
        default=None,
        help="Country config controlling default plan, data, and land-outline paths.",
    )
    p.add_argument("--date", type=str, default=NATIONAL_CENTER.isoformat())
    p.add_argument("--start-date", type=str, default=JUL_PM45[0])
    p.add_argument("--end-date", type=str, default=JUL_PM45[1])
    p.add_argument("--max-frames", type=int, default=16)
    p.add_argument("--side", type=int, default=512)
    p.add_argument("--max-snow-frac", type=float, default=0.05)
    p.add_argument("--max-cloud-frac", type=float, default=0.15)
    p.add_argument("--min-valid-frac", type=float, default=0.85)
    p.add_argument("--max-stac-items", type=int, default=400)
    p.add_argument("--plan-only", action="store_true")
    p.add_argument("--fetch-only", action="store_true", help="Reuse existing plan.json")
    p.add_argument("--tile", action="store_true", help="Also write LR512 tile dirs from the plan")
    p.add_argument(
        "--plan-dir",
        type=Path,
        default=None,
        help="Default: production/national_2025/plans/{MGRS}",
    )
    p.add_argument(
        "--s2-out",
        type=Path,
        default=None,
        help="Default: data/s2_revisits/national_2025/{MGRS}",
    )
    p.add_argument("--dry-run-fetch", action="store_true")
    args = p.parse_args()

    mgrs = str(args.mgrs).upper().lstrip("T")
    country_config = (
        load_country_config(args.country_config) if args.country_config is not None else None
    )
    plan_dir = args.plan_dir
    if plan_dir is None:
        if country_config is not None:
            plan_dir = configured_path(country_config, "paths", "plan_root") / mgrs
        else:
            plan_dir = ROOT / "production" / "national_2025" / "plans" / mgrs
    elif not plan_dir.is_absolute():
        plan_dir = ROOT / plan_dir
    s2_out = args.s2_out
    if s2_out is None:
        if country_config is not None:
            s2_out = configured_path(country_config, "paths", "s2_root") / mgrs
        else:
            s2_out = ROOT / "data" / "s2_revisits" / "national_2025" / mgrs
    elif not s2_out.is_absolute():
        s2_out = ROOT / s2_out

    window = f"{args.start_date}/{args.end_date}"
    plan_path = plan_dir / "plan.json"
    if args.fetch_only:
        if not plan_path.is_file():
            raise SystemExit(f"missing {plan_path}; run without --fetch-only first")
        plan = json.loads(plan_path.read_text())
    else:
        max_snow = None if float(args.max_snow_frac) < 0 else float(args.max_snow_frac)
        plan = plan_mgrs(
            mgrs,
            datetime_range=window,
            center=args.date,
            side=int(args.side),
            max_cloud_frac=float(args.max_cloud_frac),
            min_valid_frac=float(args.min_valid_frac),
            include_shadow=False,
            max_snow_frac=max_snow,
            max_stac_items=int(args.max_stac_items),
            max_stac_cloud=100.0,
            max_frames=int(args.max_frames),
            out_dir=plan_dir,
            mainland_only=True,
            land_mask=(
                configured_path(country_config, "inputs", "land_outline_geojson")
                if country_config is not None
                else None
            ),
            min_land_frac=0.0,
        )
        plan_path = Path(plan.get("plan_path") or plan_path)

    print(
        f"plan {mgrs}: union_dates={plan.get('n_union_dates')} "
        f"land_cells={plan.get('n_cells_with_frames')}/"
        f"{plan.get('n_land_cells')} "
        f"n_frames={plan.get('n_frames_min')}–{plan.get('n_frames_max')}",
        flush=True,
    )
    if args.plan_only:
        return

    bbox = plan.get("bbox_wgs84")
    if not bbox or len(bbox) != 4:
        raise SystemExit(f"{plan_path} missing bbox_wgs84")
    fetch_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "fetch_s2_revisits.py"),
        "--out",
        str(s2_out),
        "--start-date",
        args.start_date,
        "--end-date",
        args.end_date,
        "--date",
        args.date,
        "--bbox",
        *(str(v) for v in bbox),
        "--mgrs-tile",
        mgrs,
        "--dates-file",
        str(plan_path),
        "--cloud-method",
        "none",
        "--num-samples",
        "0",
        "--max-stac-cloud",
        "100",
        "--max-stac-items",
        str(args.max_stac_items),
        "--no-preview",
    ]
    if args.dry_run_fetch:
        fetch_cmd.append("--dry-run")
    _run(fetch_cmd)

    ondisk_plan = plan_dir / "plan_ondisk.json"
    if not args.dry_run_fetch:
        _run(
            [
                sys.executable,
                str(ROOT / "scripts" / "rescore_cell_plan.py"),
                "--plan",
                str(plan_path),
                "--s2-dir",
                str(s2_out),
                "--out",
                str(ondisk_plan),
            ]
        )
        tile_plan = ondisk_plan if ondisk_plan.is_file() else plan_path
    else:
        tile_plan = plan_path

    if args.tile:
        man = s2_out / f"granule_tiles_lr{int(args.side)}_manifest.json"
        _run(
            [
                sys.executable,
                str(ROOT / "scripts" / "make_granule_tiles.py"),
                "--src",
                str(s2_out),
                "--out-root",
                str(s2_out.parent),
                "--side",
                str(args.side),
                "--manifest",
                str(man),
                "--mainland-only",
                "--cell-plan",
                str(tile_plan),
            ]
        )


if __name__ == "__main__":
    main()
