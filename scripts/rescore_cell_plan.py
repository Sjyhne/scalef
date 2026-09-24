#!/usr/bin/env python3
"""Re-admit national LR512 cells from SCL that actually landed on disk.

Does not re-download RGB. Reads each frame's ``scl_path`` / ``cloud_mask`` in
the cell window, applies the plan rules, re-ranks passing days by |date −
center|, and writes ``plan_ondisk.json``. Optionally rewrites existing tile
``meta.json`` frame lists.

Example::

    python scripts/rescore_cell_plan.py \\
      --plan production/national_2025/plans/32VNM/plan.json \\
      --s2-dir data/s2_revisits/national_2025/32VNM \\
      --rewrite-tiles
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.s2_cloud_mask import clear_from_cloud_mask, score_scl_cell  # noqa: E402
from scripts.national_cell_queue import (  # noqa: E402
    NATIONAL_CENTER,
    build_cell_plan,
    cell_lookup,
    filter_frames_for_cell,
    frame_day,
)


def _rules(plan: dict) -> dict:
    rules = dict(plan.get("rules") or {})
    return {
        "max_cloud_frac": float(rules.get("max_cloud_frac", 0.15)),
        "min_valid_frac": float(rules.get("min_valid_frac", 0.85)),
        "include_shadow": bool(rules.get("include_shadow", False)),
        "max_snow_frac": rules.get("max_snow_frac", 0.05),
        "max_frames": int(plan.get("max_frames") or 16),
        "side": int(plan.get("side") or 512),
        "center": plan.get("center_date") or NATIONAL_CENTER.isoformat(),
    }


def _read_window(path: Path, row0: int, col0: int, side: int) -> np.ndarray:
    import rasterio

    with rasterio.open(path) as src:
        if int(src.width) == int(side) and int(src.height) == int(side):
            return src.read(1)
        return src.read(1, window=((row0, row0 + side), (col0, col0 + side)))


def _score_frame(
    s2_dir: Path,
    frame: dict,
    *,
    row0: int,
    col0: int,
    side: int,
    rules: dict,
) -> tuple[bool, float, float]:
    scl_name = frame.get("scl_path")
    mask_name = frame.get("cloud_mask")
    if scl_name and (s2_dir / str(scl_name)).is_file():
        scl = _read_window(s2_dir / str(scl_name), row0, col0, side)
        passed, cloud, snow, _valid = score_scl_cell(
            scl,
            max_cloud_frac=rules["max_cloud_frac"],
            min_valid_frac=rules["min_valid_frac"],
            include_shadow=rules["include_shadow"],
            max_snow_frac=rules["max_snow_frac"],
        )
        return passed, cloud, snow
    if mask_name and (s2_dir / str(mask_name)).is_file():
        mask = _read_window(s2_dir / str(mask_name), row0, col0, side)
        cloudy = ~clear_from_cloud_mask(mask)
        cloud = float(cloudy.mean()) if cloudy.size else 1.0
        passed = cloud <= float(rules["max_cloud_frac"])
        return passed, cloud, 0.0
    return False, float("nan"), float("nan")


def rescore_plan(plan: dict, s2_dir: Path) -> dict:
    """Rebuild the cell plan from parent-stack SCL windows."""
    rules = _rules(plan)
    side = rules["side"]
    meta = json.loads((s2_dir / "meta.json").read_text())
    frames = list(meta.get("frames") or [])
    if not frames:
        raise SystemExit(f"{s2_dir}/meta.json has no frames")

    grid = plan.get("grid") or {}
    n_y = int(grid.get("n_y") or 0)
    n_x = int(grid.get("n_x") or 0)
    if n_y < 1 or n_x < 1:
        raise SystemExit("plan.json missing grid.n_y / n_x")

    day_to_frame: dict[str, dict] = {}
    days: list[str] = []
    for fr in frames:
        try:
            day = frame_day(fr)
        except ValueError:
            continue
        if day not in day_to_frame:
            day_to_frame[day] = fr
            days.append(day)
    days = sorted(set(days))
    if not days:
        raise SystemExit(f"{s2_dir}: no parseable frame dates")

    n_days = len(days)
    passed = np.zeros((n_days, n_y, n_x), dtype=bool)
    cloud = np.full((n_days, n_y, n_x), np.nan, dtype=np.float32)
    snow = np.full((n_days, n_y, n_x), np.nan, dtype=np.float32)
    old = cell_lookup(plan)
    land = np.zeros((n_y, n_x), dtype=bool)
    for (iy, ix), rec in old.items():
        if 0 <= iy < n_y and 0 <= ix < n_x:
            land[iy, ix] = True

    for d_i, day in enumerate(days):
        fr = day_to_frame[day]
        for iy in range(n_y):
            for ix in range(n_x):
                if not land[iy, ix]:
                    continue
                rec = old.get((iy, ix)) or {}
                row0 = int(rec.get("row_off", iy * side))
                col0 = int(rec.get("col_off", ix * side))
                ok, c, s = _score_frame(
                    s2_dir, fr, row0=row0, col0=col0, side=side, rules=rules
                )
                passed[d_i, iy, ix] = ok
                cloud[d_i, iy, ix] = c
                snow[d_i, iy, ix] = s

    rebuilt = build_cell_plan(
        days,
        passed,
        cloud,
        snow,
        land,
        center=rules["center"],
        max_frames=rules["max_frames"],
        side=side,
    )
    rebuilt["mgrs_tile"] = plan.get("mgrs_tile")
    rebuilt["date_range"] = plan.get("date_range")
    rebuilt["rules"] = dict(plan.get("rules") or {})
    rebuilt["rules"]["cloud_source"] = "SCL_ondisk"
    rebuilt["rules"]["ranking"] = "closest_to_center_cap_max_frames_keep_thin"
    rebuilt["bbox_wgs84"] = plan.get("bbox_wgs84")
    rebuilt["source_plan"] = str(plan.get("plan_path") or "")
    rebuilt["s2_dir"] = str(s2_dir)
    rebuilt["days_scored"] = days
    rebuilt["n_days_scored"] = len(days)
    rebuilt["days_scored_items"] = [
        {"date": frame_day(fr), "stac_id": fr.get("stac_id")}
        for fr in frames
        if fr.get("stac_id")
    ]
    return rebuilt


def rewrite_tiles(plan: dict, s2_dir: Path, *, out_root: Path | None = None) -> dict:
    """Patch existing ``{parent}_t{side}_yYY_xXX/meta.json`` frame lists."""
    parent_meta = json.loads((s2_dir / "meta.json").read_text())
    side = int(plan.get("side") or 512)
    root = out_root if out_root is not None else s2_dir.parent
    n_rewritten = 0
    n_changed = 0
    changed: list[dict] = []
    for rec in plan.get("cells") or []:
        iy, ix = int(rec["iy"]), int(rec["ix"])
        dest = root / f"{s2_dir.name}_t{side}_y{iy:02d}_x{ix:02d}"
        meta_path = dest / "meta.json"
        if not meta_path.is_file():
            continue
        frames = filter_frames_for_cell(parent_meta.get("frames") or [], rec.get("dates") or [])
        meta = json.loads(meta_path.read_text())
        old_dates = list((meta.get("national_cell") or {}).get("dates") or [])
        new_dates = list(rec.get("dates") or [])
        meta["frames"] = frames
        meta["national_cell"] = {
            k: rec[k]
            for k in (
                "iy",
                "ix",
                "key",
                "dates",
                "n_frames",
                "date_span_days",
                "mean_snow_used",
                "mean_cloud_used",
            )
            if k in rec
        }
        meta["national_cell"]["rescore"] = "SCL_ondisk"
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        n_rewritten += 1
        if old_dates != new_dates:
            n_changed += 1
            dropped = sorted(set(old_dates) - set(new_dates))
            added = sorted(set(new_dates) - set(old_dates))
            changed.append(
                {
                    "key": rec.get("key"),
                    "dropped": dropped,
                    "added": added,
                    "n_frames": rec.get("n_frames"),
                }
            )
    return {"n_rewritten": n_rewritten, "n_changed": n_changed, "changed": changed}


def rescore_and_write(
    plan_path: Path,
    s2_dir: Path,
    *,
    out: Path,
    rewrite: bool = False,
    out_root: Path | None = None,
) -> dict:
    plan = json.loads(plan_path.read_text())
    rebuilt = rescore_plan(plan, s2_dir)
    # Avoid rewriting QA GeoTIFFs on top of the planner's originals.
    rebuilt.pop("qa", None)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rebuilt, indent=2) + "\n")
    rebuilt["plan_path"] = str(out)
    print(
        f"ondisk {rebuilt.get('mgrs_tile')}: "
        f"with_frames={rebuilt['n_cells_with_frames']}/"
        f"{rebuilt['n_land_cells']} "
        f"no_pass={rebuilt['n_cells_no_pass']} "
        f"union={rebuilt['n_union_dates']} "
        f"n_frames={rebuilt['n_frames_min']}–{rebuilt['n_frames_max']} "
        f"→ {out}",
        flush=True,
    )
    if rewrite:
        stats = rewrite_tiles(rebuilt, s2_dir, out_root=out_root)
        rebuilt["rewrite"] = {
            "n_rewritten": stats["n_rewritten"],
            "n_changed": stats["n_changed"],
            "changed_sample": stats["changed"][:20],
        }
        side = out.with_suffix(out.suffix + ".rewrite.json")
        side.write_text(json.dumps(stats, indent=2) + "\n")
        print(
            f"rewrote {stats['n_rewritten']} tiles ({stats['n_changed']} date sets changed)",
            flush=True,
        )
    return rebuilt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--s2-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--rewrite-tiles", action="store_true")
    ap.add_argument("--out-root", type=Path, default=None)
    args = ap.parse_args()
    plan_path = args.plan if args.plan.is_absolute() else ROOT / args.plan
    s2_dir = args.s2_dir if args.s2_dir.is_absolute() else ROOT / args.s2_dir
    out = args.out
    if out is None:
        out = plan_path.parent / "plan_ondisk.json"
    elif not out.is_absolute():
        out = ROOT / out
    out_root = args.out_root
    if out_root is not None and not out_root.is_absolute():
        out_root = ROOT / out_root
    rescore_and_write(
        plan_path,
        s2_dir,
        out=out,
        rewrite=bool(args.rewrite_tiles),
        out_root=out_root,
    )


if __name__ == "__main__":
    main()
