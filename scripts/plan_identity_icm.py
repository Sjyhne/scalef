#!/usr/bin/env python3
"""Per-MGRS identity date ladder + neighbour ICM.

Phase 1: assign the ladder primary date wherever cell SCL ≤ slack.
Phase 2: leftovers may join a neighbour's date if that date is in-stack
and ≤ slack (grows weather islands instead of salt-and-pepper).
Never force a date that is missing from the cell stack.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.s2_cloud_mask import score_scl_cell  # noqa: E402
from scripts.national_cell_queue import cell_lookup, frame_day  # noqa: E402
from s2_dataset import _base_frame_index_for_nib  # noqa: E402

CENTER = date(2025, 7, 15)
SIDE = 512


def _parse_iso(d: str) -> date:
    return date.fromisoformat(str(d)[:10])


def independent_date(
    date_cloud: dict[str, float | None],
    *,
    center: date,
    max_cloud: float,
) -> str | None:
    """Same rule as s2_dataset._base_frame_index_for_nib, on cell SCL scores."""
    frames = [{"datetime": f"{d}T00:00:00+00:00", "path": d} for d in date_cloud]
    if not frames:
        return None
    clouds = []
    for d in date_cloud:
        c = date_cloud[d]
        clouds.append(1.0 if c is None or c != c else float(c))
    idx = _base_frame_index_for_nib(
        frames, center, cloud_fracs=clouds, max_cloud_frac=max_cloud
    )
    return list(date_cloud.keys())[idx]


def score_cell_dates(
    *,
    parent: Path,
    frames_by_day: dict[str, dict],
    cells: list[dict],
) -> dict[tuple[int, int], dict[str, float | None]]:
    import rasterio
    from rasterio.windows import Window

    wanted: dict[str, list[dict]] = {}
    for rec in cells:
        for d in rec.get("dates") or []:
            wanted.setdefault(d, []).append(rec)

    out: dict[tuple[int, int], dict[str, float | None]] = {
        (int(c["iy"]), int(c["ix"])): {} for c in cells
    }
    for day, recs in wanted.items():
        fr = frames_by_day.get(day)
        if fr is None or not fr.get("scl_path"):
            for rec in recs:
                out[int(rec["iy"]), int(rec["ix"])][day] = None
            continue
        path = parent / fr["scl_path"]
        if not path.is_file():
            for rec in recs:
                out[int(rec["iy"]), int(rec["ix"])][day] = None
            continue
        with rasterio.open(path) as src:
            for rec in recs:
                scl = src.read(
                    1,
                    window=Window(
                        int(rec["col_off"]),
                        int(rec["row_off"]),
                        int(rec.get("side") or SIDE),
                        int(rec.get("side") or SIDE),
                    ),
                )
                _ok, cloud, _snow, _valid = score_scl_cell(
                    scl, max_cloud_frac=1.0, min_valid_frac=0.0
                )
                out[int(rec["iy"]), int(rec["ix"])][day] = (
                    None if cloud != cloud else float(cloud)
                )
    return out


def date_ladder(
    date_clouds: dict[tuple[int, int], dict[str, float | None]],
    *,
    center: date,
    max_cloud: float,
    min_cover: float = 0.5,
) -> list[str]:
    """Widely clear dates first; among those covering ``min_cover``, closer to center wins.

    A distant day that is clear everywhere must not beat a nearer majority day
    (12 Aug vs 12 Jul on 32VNM). Leftovers are the point of later stack stages.
    """
    n = max(len(date_clouds), 1)
    counts: Counter[str] = Counter()
    for clouds in date_clouds.values():
        for d, c in clouds.items():
            if c is not None and c <= max_cloud:
                counts[d] += 1
    wide = [d for d, k in counts.items() if k / n >= min_cover]
    pool = wide if wide else list(counts)
    return sorted(
        pool,
        key=lambda d: (
            abs((_parse_iso(d) - center).days),
            -counts[d],
            d,
        ),
    )


def _neighbours(
    iy: int, ix: int, cells: set[tuple[int, int]]
) -> list[tuple[int, int]]:
    out = []
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        xy = (iy + dy, ix + dx)
        if xy in cells:
            out.append(xy)
    return out


def assign_icm(
    date_clouds: dict[tuple[int, int], dict[str, float | None]],
    *,
    center: date,
    hard_cloud: float,
    slack_cloud: float,
    parent: str = "32VNM",
    tile_ids: dict[tuple[int, int], str] | None = None,
    geometry: dict[tuple[int, int], dict] | None = None,
) -> dict:
    cells = set(date_clouds)
    independent = {
        xy: independent_date(clouds, center=center, max_cloud=hard_cloud)
        for xy, clouds in date_clouds.items()
    }
    ladder = date_ladder(date_clouds, center=center, max_cloud=hard_cloud, min_cover=0.5)
    primary = ladder[0] if ladder else None

    assignment = dict(independent)
    # Phase 1: join primary wherever slack allows.
    if primary is not None:
        for xy, clouds in date_clouds.items():
            c = clouds.get(primary)
            if c is not None and c <= slack_cloud:
                assignment[xy] = primary

    # Phase 2: leftovers join the modal slack-ok neighbour date.
    leftovers = {xy for xy, d in assignment.items() if d != primary}
    changed = True
    sweeps = 0
    while changed and sweeps < 32:
        changed = False
        sweeps += 1
        for xy in sorted(leftovers):
            clouds = date_clouds[xy]
            neigh = _neighbours(xy[0], xy[1], cells)
            if not neigh:
                continue
            votes: Counter[str] = Counter()
            for nxy in neigh:
                nd = assignment[nxy]
                if nd is None:
                    continue
                c = clouds.get(nd)
                if c is not None and c <= slack_cloud:
                    votes[nd] += 1
            if not votes:
                continue
            best, n = votes.most_common(1)[0]
            if best != assignment[xy] and n >= 1:
                assignment[xy] = best
                changed = True
        leftovers = {xy for xy, d in assignment.items() if d != primary}

    # Phase 3: polish every boundary, including cells already assigned primary.
    # A coordinate update is accepted only when it improves local cut count,
    # then independent-date fidelity, then distance to the requested center.
    polish_sweeps = 0
    n_polished = 0
    changed = True
    while changed and polish_sweeps < 32:
        changed = False
        polish_sweeps += 1
        for xy in sorted(cells):
            neighbours = _neighbours(xy[0], xy[1], cells)
            if not neighbours:
                continue
            current = assignment[xy]
            candidates = {current}
            candidates.update(assignment[item] for item in neighbours)
            candidates = {
                day
                for day in candidates
                if day is not None
                and date_clouds[xy].get(day) is not None
                and float(date_clouds[xy][day]) <= slack_cloud
            }
            if not candidates:
                continue

            def local_objective(day: str) -> tuple[int, int, int, str]:
                return (
                    sum(assignment[item] != day for item in neighbours),
                    int(day != independent[xy]),
                    abs((_parse_iso(day) - center).days),
                    day,
                )

            best = min(candidates, key=local_objective)
            if local_objective(best) < local_objective(current):
                assignment[xy] = best
                n_polished += 1
                changed = True
    leftovers = {xy for xy, d in assignment.items() if d != primary}

    records = []
    for xy in sorted(cells):
        ind = independent[xy]
        icm = assignment[xy]
        primary_cloud = date_clouds[xy].get(primary) if primary else None
        record = {
                "iy": xy[0],
                "ix": xy[1],
                "tile_id": (
                    tile_ids[xy]
                    if tile_ids is not None
                    else f"{parent}_t512_y{xy[0]:02d}_x{xy[1]:02d}"
                ),
                "independent_date": ind,
                "icm_date": icm,
                "joined_primary": bool(primary and icm == primary and ind != primary),
                "joined_neighbour": bool(icm != ind and icm != primary),
                "leftover": bool(primary and icm != primary),
                "primary_cloud": primary_cloud,
                "has_primary": primary in (date_clouds[xy] or {}),
            }
        if geometry is not None:
            record.update(geometry.get(xy) or {})
        records.append(record)

    n_ind = Counter(independent.values())
    n_icm = Counter(assignment.values())
    n_bound_ind = _n_boundaries(independent, cells)
    n_bound_icm = _n_boundaries(assignment, cells)
    return {
        "center": center.isoformat(),
        "hard_cloud": hard_cloud,
        "slack_cloud": slack_cloud,
        "ladder": ladder,
        "primary": primary,
        "n_cells": len(cells),
        "independent_counts": dict(n_ind),
        "icm_counts": dict(n_icm),
        "n_boundaries_independent": n_bound_ind,
        "n_boundaries_icm": n_bound_icm,
        "n_joined_primary": sum(1 for r in records if r["joined_primary"]),
        "n_joined_neighbour": sum(1 for r in records if r["joined_neighbour"]),
        "n_leftover": sum(1 for r in records if r["leftover"]),
        "icm_sweeps": sweeps,
        "boundary_polish_sweeps": polish_sweeps,
        "n_boundary_polished": n_polished,
        "cells": records,
        "assignment": {r["tile_id"]: r["icm_date"] for r in records},
        "independent": {r["tile_id"]: r["independent_date"] for r in records},
    }


def _n_boundaries(
    assignment: dict[tuple[int, int], str | None], cells: set[tuple[int, int]]
) -> int:
    n = 0
    for iy, ix in cells:
        d = assignment[(iy, ix)]
        if (iy, ix + 1) in cells and assignment[(iy, ix + 1)] != d:
            n += 1
        if (iy + 1, ix) in cells and assignment[(iy + 1, ix)] != d:
            n += 1
    return n


def write_ascii_grid(plan: dict, field: str = "icm_date") -> str:
    abbrev = {
        "2025-07-12": "A",
        "2025-07-18": "B",
        "2025-07-25": "C",
        "2025-08-12": "D",
        "2025-07-15": "E",
    }
    by_xy = {(c["iy"], c["ix"]): c[field] for c in plan["cells"]}
    ys = range(min(y for y, _ in by_xy), max(y for y, _ in by_xy) + 1)
    xs = range(min(x for _, x in by_xy), max(x for _, x in by_xy) + 1)
    lines = [f"legend A=07-12 B=07-18 C=07-25 D=08-12 E=07-15 *=other .=missing  field={field}"]
    for iy in ys:
        row = []
        for ix in xs:
            d = by_xy.get((iy, ix))
            if d is None:
                row.append(".")
            else:
                row.append(abbrev.get(d, "*"))
        lines.append(f"{iy:02d} " + "".join(row))
    return "\n".join(lines) + "\n"


def filter_plan_to_block(plan: dict, *, y0: int, y1: int, x0: int, x1: int) -> dict:
    cells = [
        c
        for c in plan["cells"]
        if y0 <= c["iy"] < y1 and x0 <= c["ix"] < x1
    ]
    assign = {c["tile_id"]: c["icm_date"] for c in cells}
    independent = {c["tile_id"]: c["independent_date"] for c in cells}
    by_xy = {(c["iy"], c["ix"]): c["icm_date"] for c in cells}
    keys = set(by_xy)
    out = dict(plan)
    out["block"] = {"y0": y0, "y1": y1, "x0": x0, "x1": x1}
    out["cells"] = cells
    out["assignment"] = assign
    out["independent"] = independent
    out["n_cells"] = len(cells)
    out["icm_counts"] = dict(Counter(assign.values()))
    out["independent_counts"] = dict(Counter(independent.values()))
    out["n_boundaries_icm"] = _n_boundaries(by_xy, keys)
    out["n_leftover"] = sum(1 for c in cells if c["leftover"])
    out["n_joined_primary"] = sum(1 for c in cells if c["joined_primary"])
    out["n_joined_neighbour"] = sum(1 for c in cells if c["joined_neighbour"])
    return out


def write_identity_manifests(
    plan: dict,
    *,
    granule_manifest_path: Path,
    out_dir: Path,
) -> tuple[Path, Path]:
    """Write all planned cells and the identity-changed subset as manifests."""
    granule = json.loads(granule_manifest_path.read_text())
    wanted = set(plan["assignment"])
    independent = plan["independent"]
    tiles = []
    for source in granule.get("tiles") or []:
        tile_id = source["tile_id"]
        if tile_id not in wanted:
            continue
        tile = dict(source)
        tile["force_base_date"] = plan["assignment"][tile_id]
        tiles.append(tile)
    tiles.sort(key=lambda tile: (int(tile["iy"]), int(tile["ix"])))
    if len(tiles) != len(wanted):
        found = {tile["tile_id"] for tile in tiles}
        missing = sorted(wanted - found)
        raise ValueError(f"granule manifest is missing {len(missing)} planned cells: {missing[:5]}")

    common = {
        key: value
        for key, value in granule.items()
        if key != "tiles"
    }
    common.update(
        {
            "identity_plan": str(plan.get("path") or ""),
            "identity_primary": plan.get("primary"),
            "identity_hard_cloud": plan.get("hard_cloud"),
            "identity_slack_cloud": plan.get("slack_cloud"),
        }
    )
    all_manifest = {**common, "tiles": tiles}
    changed = [
        tile
        for tile in tiles
        if plan["assignment"][tile["tile_id"]] != independent[tile["tile_id"]]
    ]
    changed_manifest = {
        **common,
        "tiles": changed,
        "subset": "identity_changed_only",
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    all_path = out_dir / "manifest_icm_all.json"
    changed_path = out_dir / "manifest_icm_changed.json"
    all_path.write_text(json.dumps(all_manifest, indent=2) + "\n")
    changed_path.write_text(json.dumps(changed_manifest, indent=2) + "\n")
    return all_path, changed_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parent", default="32VNM")
    ap.add_argument(
        "--s2-dir",
        type=Path,
        default=ROOT / "data/s2_revisits/national_2025_v2/32VNM",
    )
    ap.add_argument(
        "--plan-ondisk",
        type=Path,
        default=ROOT / "production/national_2025/plans/32VNM/plan_ondisk.json",
    )
    ap.add_argument("--y0", type=int, default=0)
    ap.add_argument("--y1", type=int, default=13)
    ap.add_argument("--x0", type=int, default=10)
    ap.add_argument("--x1", type=int, default=21)
    ap.add_argument("--hard-cloud", type=float, default=0.02)
    ap.add_argument("--slack-cloud", type=float, default=0.05)
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "production/seams/32VNM_stack/identity_plan.json",
    )
    ap.add_argument(
        "--granule-manifest",
        type=Path,
        default=None,
        help="Also write manifest_icm_all.json and manifest_icm_changed.json beside --out.",
    )
    ap.add_argument(
        "--manifest-cells",
        action="store_true",
        help="Plan directly on every tile window in --granule-manifest.",
    )
    args = ap.parse_args()

    parent = args.s2_dir if args.s2_dir.is_absolute() else ROOT / args.s2_dir
    ondisk = args.plan_ondisk if args.plan_ondisk.is_absolute() else ROOT / args.plan_ondisk
    parent_meta = json.loads((parent / "meta.json").read_text())
    frames_by_day = {frame_day(fr): fr for fr in parent_meta.get("frames") or []}
    manifest_path = None
    if args.granule_manifest is not None:
        manifest_path = (
            args.granule_manifest
            if args.granule_manifest.is_absolute()
            else ROOT / args.granule_manifest
        )
    if args.manifest_cells:
        if manifest_path is None:
            raise SystemExit("--manifest-cells requires --granule-manifest")
        cells = list((json.loads(manifest_path.read_text()).get("tiles") or []))
        if any(not rec.get("dates") for rec in cells):
            raise SystemExit("manifest cells must record non-empty planned dates")
    else:
        plan_all = json.loads(ondisk.read_text())
        by_xy = cell_lookup(plan_all)
        cells = [
            rec
            for (iy, ix), rec in by_xy.items()
            if args.y0 <= iy < args.y1 and args.x0 <= ix < args.x1
        ]
    if not cells:
        raise SystemExit("no plan cells in block")
    print(f"Scoring SCL on {len(cells)} cells …", flush=True)
    date_clouds = score_cell_dates(
        parent=parent, frames_by_day=frames_by_day, cells=cells
    )
    tile_ids = {
        (int(rec["iy"]), int(rec["ix"])): str(rec["tile_id"])
        for rec in cells
        if rec.get("tile_id")
    }
    geometry = {
        (int(rec["iy"]), int(rec["ix"])): {
            key: rec[key]
            for key in ("row_off", "col_off", "side", "stride", "overlap_frac")
            if key in rec
        }
        for rec in cells
    }
    plan = assign_icm(
        date_clouds,
        center=CENTER,
        hard_cloud=args.hard_cloud,
        slack_cloud=args.slack_cloud,
        parent=args.parent,
        tile_ids=tile_ids or None,
        geometry=geometry or None,
    )
    plan["parent"] = args.parent
    if args.manifest_cells:
        ys = [int(rec["iy"]) for rec in cells]
        xs = [int(rec["ix"]) for rec in cells]
        plan["block"] = {
            "y0": min(ys),
            "y1": max(ys) + 1,
            "x0": min(xs),
            "x1": max(xs) + 1,
        }
    else:
        plan["block"] = {
            "y0": args.y0,
            "y1": args.y1,
            "x0": args.x0,
            "x1": args.x1,
        }
    plan["s2_dir"] = str(parent.relative_to(ROOT))
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    plan["path"] = str(out.relative_to(ROOT)) if out.is_relative_to(ROOT) else str(out)
    out.write_text(json.dumps(plan, indent=2) + "\n")
    grid_i = write_ascii_grid(plan, "independent_date")
    grid_c = write_ascii_grid(plan, "icm_date")
    (out.parent / "identity_grid_independent.txt").write_text(grid_i)
    (out.parent / "identity_grid_icm.txt").write_text(grid_c)
    print(grid_i)
    print(grid_c)
    print(
        f"primary={plan['primary']}  independent {plan['independent_counts']}  "
        f"icm {plan['icm_counts']}  boundaries {plan['n_boundaries_independent']}"
        f"→{plan['n_boundaries_icm']}  joined_primary={plan['n_joined_primary']}  "
        f"leftover={plan['n_leftover']}  → {out}"
    )
    if manifest_path is not None:
        all_path, changed_path = write_identity_manifests(
            plan,
            granule_manifest_path=manifest_path,
            out_dir=out.parent,
        )
        print(
            f"manifests: all={all_path} ({len(plan['cells'])} tiles), "
            f"changed={changed_path} "
            f"({sum(c['icm_date'] != c['independent_date'] for c in plan['cells'])} tiles)"
        )


if __name__ == "__main__":
    main()
