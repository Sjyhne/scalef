#!/usr/bin/env python3
"""Add Denmark windows the coarse land test dropped, without refitting finished cells.

A window is recovered when its footprint meets the 1:10 m Denmark outline and it
is missing from the overlap manifest. Existing tiles stay. New tiles inherit
dates from any planned cell they overlap; otherwise they use the parent stack
and OmniCloudMask admission. Identity dates for the new tiles are chosen locally
so the dates already assigned to fitted cells do not move.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.land_mask_lr512 import assess_window_land, load_land_paths  # noqa: E402
from scripts.make_granule_tiles import _plan_record_for_window, _write_tile  # noqa: E402
from scripts.national_cell_queue import filter_frames_for_cell, frame_day  # noqa: E402
from scripts.plan_identity_icm import independent_date  # noqa: E402

S2_ROOT = ROOT / "data/s2_revisits/denmark_national_2025"
PROD = ROOT / "production/denmark_national_2025"
LAND = ROOT / "production/country_inventory/denmark/denmark_land_outline_ne10m.geojson"
CENTER = date(2025, 7, 15)
ADMIT_CLOUD = 0.15
ADMIT_VALID = 0.85
HARD_CLOUD = 0.02
SLACK_CLOUD = 0.05
MAX_FRAMES = 16


def _write_json(path: Path, payload: dict, *, indent: int) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=indent) + "\n")
    tmp.replace(path)


def _origins(extent: int, side: int, stride: int) -> list[int]:
    last = extent - side
    out = list(range(0, last + 1, stride))
    if out[-1] != last:
        out.append(last)
    return out


def choose_shore_date(
    clouds: dict[str, float],
    *,
    primary: str | None,
    neighbor_dates: list[str],
    center: date = CENTER,
) -> str:
    """Pick a base date for one new cell. Existing cells are not candidates to edit."""
    if primary is not None:
        primary_cloud = clouds.get(primary)
        if primary_cloud is not None and primary_cloud <= SLACK_CLOUD:
            return primary
    votes: Counter[str] = Counter()
    for day in neighbor_dates:
        cloud = clouds.get(day)
        if cloud is not None and cloud <= SLACK_CLOUD:
            votes[day] += 1
    if votes:
        return votes.most_common(1)[0][0]
    return independent_date(clouds, center=center, max_cloud=HARD_CLOUD)


def _neighbor_dates(
    iy: int,
    ix: int,
    assigned: dict[tuple[int, int], str],
) -> list[str]:
    dates = []
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        day = assigned.get((iy + dy, ix + dx))
        if day:
            dates.append(day)
    return dates


def _score_cell(parent: Path, frames: list[dict], row_off: int, col_off: int, side: int) -> dict[str, dict]:
    """Per frame path: admission stats. Missing masks are omitted."""
    out: dict[str, dict] = {}
    for fr in frames:
        stem = Path(fr["path"]).stem
        ocm_path = parent / f"{stem}_ocm.tif"
        rgb_path = parent / fr["path"]
        if not ocm_path.is_file() or not rgb_path.is_file():
            continue
        window = Window(col_off, row_off, side, side)
        with rasterio.open(ocm_path) as ocm_src, rasterio.open(rgb_path) as rgb_src:
            classes = ocm_src.read(1, window=window)
            rgb = rgb_src.read((1, 2, 3), window=window)
        valid = np.any(rgb > 0, axis=0)
        out[fr["path"]] = {
            "date": frame_day(fr),
            "cloud": float(np.isin(classes, (1, 2)).mean()),
            "valid": float(valid.mean()),
            "masked": float(np.mean(classes > 0)),
        }
    return out


def _link_ocm(dest: Path, parent: Path, frames: list[dict]) -> None:
    for fr in frames:
        stem = Path(fr["path"]).stem
        src = parent / f"{stem}_ocm.tif"
        dst = dest / f"{stem}_ocm.tif"
        if not src.is_file():
            continue
        if dst.is_symlink() or dst.exists():
            continue
        dst.symlink_to(src)


def recover_granule(mgrs: str, *, land_paths, dry_run: bool) -> dict:
    man_path = S2_ROOT / mgrs / "granule_tiles_lr512_ovl5_manifest.json"
    prepared_path = PROD / "identity" / f"{mgrs}_ovl5" / "manifest.json"
    plan_path = PROD / "plans" / mgrs / "plan_ondisk.json"
    man = json.loads(man_path.read_text())
    parent = S2_ROOT / mgrs
    meta = json.loads((parent / "meta.json").read_text())
    frames = list(meta.get("frames") or [])
    first = parent / frames[0]["path"]
    with rasterio.open(first) as src:
        transform, crs = src.transform, src.crs
    aoi = man["aoi"]
    side = int(man["side"])
    stride = int(man["stride"])
    rows = [int(aoi["row_off"]) + o for o in _origins(int(aoi["height"]), side, stride)]
    cols = [int(aoi["col_off"]) + o for o in _origins(int(aoi["width"]), side, stride)]
    existing = {(int(t["iy"]), int(t["ix"])) for t in man["tiles"]}
    plan_cells = json.loads(plan_path.read_text()).get("cells") or []

    found = []
    for iy, row_off in enumerate(rows):
        for ix, col_off in enumerate(cols):
            if (iy, ix) in existing:
                continue
            frac, keep = assess_window_land(
                land_paths,
                transform,
                crs,
                row0=row_off,
                col0=col_off,
                side=side,
                sample_n=9,
                min_land_frac=0.0,
            )
            if not keep:
                continue
            found.append(
                {
                    "iy": iy,
                    "ix": ix,
                    "row_off": row_off,
                    "col_off": col_off,
                    "land_frac": round(float(frac), 4),
                }
            )

    if dry_run or not found:
        return {"mgrs": mgrs, "n_new": len(found), "admitted": 0, "dry_run": dry_run}

    prepared = json.loads(prepared_path.read_text())
    screen_path = prepared_path.parent / "frame_screen.json"
    identity_path = prepared_path.parent / "identity_plan.json"
    screen = json.loads(screen_path.read_text())
    identity = json.loads(identity_path.read_text())
    assigned = {
        (int(c["iy"]), int(c["ix"])): c.get("icm_date")
        for c in identity.get("cells") or []
        if c.get("icm_date")
    }
    prepared_ids = {t["tile_id"] for t in prepared.get("tiles") or []}
    admitted_rows = []
    dropped = list(prepared.get("admission_dropped") or [])

    for spec in found:
        iy, ix = int(spec["iy"]), int(spec["ix"])
        tile_id = f"{mgrs}_t512_ovl5_y{iy:02d}_x{ix:02d}"
        dest = S2_ROOT / tile_id
        cell_rec = _plan_record_for_window(
            plan_cells,
            row_off=int(spec["row_off"]),
            col_off=int(spec["col_off"]),
            side=side,
        )
        if cell_rec is None:
            dates = sorted({frame_day(fr) for fr in frames})
            cell_rec = {"dates": dates, "n_frames": len(dates), "source_plan_cells": []}
            tile_frames = list(frames)
        else:
            tile_frames = filter_frames_for_cell(frames, cell_rec.get("dates") or [])
        if not tile_frames:
            dropped.append({"tile_id": tile_id, "iy": iy, "ix": ix, "n_frames": 0})
            continue
        row = _write_tile(
            parent,
            meta,
            dest,
            row0=int(spec["row_off"]),
            col0=int(spec["col_off"]),
            side=side,
            iy=iy,
            ix=ix,
            force=False,
            stride=stride,
            overlap_frac=float(man["overlap_frac"]),
            frames=tile_frames,
            cell_rec=cell_rec,
        )
        row["land_frac"] = spec["land_frac"]
        _link_ocm(dest, parent, tile_frames)
        scores = _score_cell(parent, tile_frames, int(spec["row_off"]), int(spec["col_off"]), side)
        exclude = []
        admitted = []
        for fr in tile_frames:
            rec = scores.get(fr["path"])
            if rec is None or rec["cloud"] > ADMIT_CLOUD or rec["valid"] < ADMIT_VALID:
                exclude.append(fr["path"])
                continue
            admitted.append((fr, rec))
        admitted.sort(
            key=lambda item: (
                abs((date.fromisoformat(item[1]["date"]) - CENTER).days),
                item[1]["date"],
            )
        )
        for fr, _rec in admitted[MAX_FRAMES:]:
            exclude.append(fr["path"])
        admitted = admitted[:MAX_FRAMES]
        if not admitted:
            dropped.append(
                {"tile_id": tile_id, "iy": iy, "ix": ix, "n_frames": len(tile_frames)}
            )
            continue
        clouds = {rec["date"]: rec["masked"] for _fr, rec in admitted}
        primary = identity.get("primary")
        ind = independent_date(clouds, center=CENTER, max_cloud=HARD_CLOUD)
        day = choose_shore_date(
            clouds,
            primary=primary,
            neighbor_dates=_neighbor_dates(iy, ix, assigned),
        )
        assigned[(iy, ix)] = day
        screen["stacks"][tile_id] = {
            "stack": str(dest.relative_to(ROOT)),
            "n_frames": len(tile_frames),
            "n_excluded": len(exclude),
            "exclude": sorted(exclude),
            "flagged": {},
            "cloud_mask_suffix": "_ocm",
        }
        identity["cells"].append(
            {
                "iy": iy,
                "ix": ix,
                "tile_id": tile_id,
                "independent_date": ind,
                "icm_date": day,
                "joined_primary": bool(primary and day == primary and ind != primary),
                "joined_neighbour": bool(day != ind and day != primary),
                "leftover": bool(identity.get("primary") and day != identity.get("primary")),
                "primary_cloud": clouds.get(identity.get("primary")),
                "has_primary": identity.get("primary") in clouds,
                "shore_recovered": True,
            }
        )
        identity.setdefault("assignment", {})[tile_id] = day
        identity.setdefault("independent", {})[tile_id] = identity["cells"][-1]["independent_date"]
        if tile_id not in prepared_ids:
            prepared.setdefault("tiles", []).append(row)
            prepared_ids.add(tile_id)
        if not any(t["tile_id"] == tile_id for t in man["tiles"]):
            man["tiles"].append(row)
        admitted_rows.append(tile_id)

    recovered_ids = {t["tile_id"] for t in man["tiles"]}
    land_filter = man.setdefault("land_filter", {})
    land_filter["dropped"] = [
        row for row in land_filter.get("dropped") or [] if row.get("tile_id") not in recovered_ids
    ]
    land_filter["shore_outline"] = str(LAND.relative_to(ROOT))
    land_filter["n_shore_recovered"] = len(admitted_rows)
    man["grid"]["n_tiles"] = len(man["tiles"])
    prepared["grid"]["n_tiles"] = len(prepared["tiles"])
    prepared["grid"]["n_shore_recovered"] = len(admitted_rows)
    prepared["admission_dropped"] = dropped
    identity["n_cells"] = len(identity["cells"])
    identity["shore_recovered"] = len(admitted_rows)

    _write_json(man_path, man, indent=2)
    _write_json(prepared_path, prepared, indent=1)
    _write_json(screen_path, screen, indent=1)
    _write_json(identity_path, identity, indent=1)
    return {"mgrs": mgrs, "n_new": len(found), "admitted": len(admitted_rows), "dry_run": False}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mgrs", action="append", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    land_paths = load_land_paths(LAND)
    if args.mgrs:
        granules = args.mgrs
    else:
        granules = sorted(p.parent.name for p in S2_ROOT.glob("*/granule_tiles_lr512_ovl5_manifest.json"))
    total = 0
    for mgrs in granules:
        stats = recover_granule(mgrs, land_paths=land_paths, dry_run=bool(args.dry_run))
        total += int(stats["admitted"] if not args.dry_run else stats["n_new"])
        print(
            f"{mgrs}: candidates {stats['n_new']} admitted {stats['admitted']}",
            flush=True,
        )
    print(f"total {'candidates' if args.dry_run else 'admitted'} {total}", flush=True)


if __name__ == "__main__":
    main()
