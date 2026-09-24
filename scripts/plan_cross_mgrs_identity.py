#!/usr/bin/env python3
"""Audit and reduce identity-date cuts between physically neighbouring MGRS cells.

Inputs may be repeated ``--granule MGRS IDENTITY_PLAN GRANULE_MANIFEST S2_META``
arguments or a JSON ``--inputs-manifest`` containing a ``granules`` list with
those four fields. Revised plans are written to a new output directory; source
plans are never modified. ``--audit-only`` writes only the audit report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from affine import Affine
from pyproj import CRS, Transformer

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.s2_cloud_mask import score_scl_cell  # noqa: E402
from scripts.national_cell_queue import frame_day  # noqa: E402

VERSION = 1


@dataclass
class Granule:
    mgrs: str
    plan_path: Path
    manifest_path: Path
    meta_path: Path
    plan: dict
    manifest: dict
    meta: dict


@dataclass
class Cell:
    key: str
    mgrs: str
    tile_id: str
    iy: int
    ix: int
    row_off: int
    col_off: int
    side: int
    original_date: str | None
    available: set[str]
    polygon: list[tuple[float, float]]
    scores: dict[str, float | None] = field(default_factory=dict)
    availability_source: str = "cell_or_tile"


def _json(path: Path) -> Any:
    with path.open(encoding="utf-8") as src:
        return json.load(src)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as src:
        for chunk in iter(lambda: src.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(path: str | Path, base: Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (base / value).resolve()


def _normal_mgrs(value: Any) -> str:
    text = str(value or "").upper()
    return text[1:] if text.startswith("T") else text


def _infer_mgrs(explicit: str | None, plan: dict, manifest: dict, meta: dict) -> str:
    values = [
        explicit,
        plan.get("parent"),
        plan.get("mgrs_tile"),
        manifest.get("parent"),
        manifest.get("mgrs_tile"),
        meta.get("mgrs_tile"),
    ]
    for value in values:
        mgrs = _normal_mgrs(value)
        if mgrs:
            return mgrs
    raise ValueError("could not infer MGRS; provide it in the input record")


def load_granules(specs: list[dict[str, Any]]) -> list[Granule]:
    out: list[Granule] = []
    seen: set[str] = set()
    for spec in specs:
        base = Path(spec.get("_base", Path.cwd()))
        plan_path = _resolve(spec["identity_plan"], base)
        manifest_path = _resolve(spec["granule_manifest"], base)
        meta_path = _resolve(spec["s2_meta"], base)
        plan, manifest, meta = _json(plan_path), _json(manifest_path), _json(meta_path)
        mgrs = _infer_mgrs(spec.get("mgrs"), plan, manifest, meta)
        if mgrs in seen:
            raise ValueError(f"duplicate MGRS input: {mgrs}")
        seen.add(mgrs)
        out.append(
            Granule(
                mgrs=mgrs,
                plan_path=plan_path,
                manifest_path=manifest_path,
                meta_path=meta_path,
                plan=plan,
                manifest=manifest,
                meta=meta,
            )
        )
    if len(out) < 2:
        raise ValueError("at least two granules are required")
    return sorted(out, key=lambda item: item.mgrs)


def _parse_transform(value: Any) -> Affine | None:
    if value is None:
        return None
    if isinstance(value, dict):
        try:
            return Affine(*(float(value[k]) for k in ("a", "b", "c", "d", "e", "f")))
        except (KeyError, TypeError, ValueError):
            return None
    if isinstance(value, (list, tuple)) and len(value) >= 6:
        return Affine(*(float(v) for v in value[:6]))
    return None


def _source_georef(granule: Granule) -> tuple[Affine, CRS]:
    transform = _parse_transform(granule.meta.get("transform"))
    crs_value = granule.meta.get("crs")
    if transform is not None and crs_value:
        return transform, CRS.from_user_input(crs_value)

    # Tile metadata created by make_granule_tiles inherits the parent transform.
    # It is a useful recovery source when an older supplied parent meta omitted it.
    metadata = [(granule.meta_path, granule.meta)]
    for tile in granule.manifest.get("tiles") or []:
        for meta_path in _tile_meta_paths(tile, granule.manifest_path):
            metadata.append((meta_path, _json(meta_path)))
            candidate_transform = _parse_transform(metadata[-1][1].get("transform"))
            candidate_crs = metadata[-1][1].get("crs")
            if candidate_transform is not None and candidate_crs:
                return candidate_transform, CRS.from_user_input(candidate_crs)

    # Recover from a real source raster rather than guessing an MGRS origin.
    import rasterio

    for meta_path, meta in metadata:
        for frame in meta.get("frames") or []:
            for key in ("scl_path", "path"):
                value = frame.get(key)
                if not value:
                    continue
                raster_path = _resolve(value, meta_path.parent)
                if raster_path.is_file():
                    with rasterio.open(raster_path) as src:
                        if src.crs is None:
                            continue
                        return src.transform, CRS.from_user_input(src.crs)
    raise ValueError(f"{granule.meta_path}: no usable transform/CRS or source raster")


def _cell_polygon(
    transform: Affine,
    transformer: Transformer,
    *,
    row_off: int,
    col_off: int,
    side: int,
) -> list[tuple[float, float]]:
    source = [
        transform * (col_off, row_off),
        transform * (col_off + side, row_off),
        transform * (col_off + side, row_off + side),
        transform * (col_off, row_off + side),
    ]
    return [transformer.transform(x, y) for x, y in source]


def _dates_from_frames(frames: list[dict]) -> set[str]:
    dates: set[str] = set()
    for frame in frames:
        try:
            dates.add(frame_day(frame))
        except ValueError:
            continue
    return dates


def _tile_meta_paths(tile: dict, manifest_path: Path) -> list[Path]:
    s2_dir = tile.get("s2_dir")
    if not s2_dir:
        return []
    directories = [
        _resolve(s2_dir, ROOT),
        _resolve(s2_dir, manifest_path.parent),
    ]
    paths = []
    for directory in directories:
        path = directory / "meta.json"
        if path.is_file() and path not in paths:
            paths.append(path)
    return paths


def _tile_meta_dates(tile: dict, manifest_path: Path) -> set[str]:
    for path in _tile_meta_paths(tile, manifest_path):
        return _dates_from_frames(_json(path).get("frames") or [])
    return set()


def build_cells(granules: list[Granule], common_crs: str) -> dict[str, Cell]:
    target = CRS.from_user_input(common_crs)
    cells: dict[str, Cell] = {}
    for granule in granules:
        transform, source_crs = _source_georef(granule)
        transformer = Transformer.from_crs(source_crs, target, always_xy=True)
        assignment = granule.plan.get("assignment") or {}
        plan_cells = {
            str(rec.get("tile_id")): rec
            for rec in granule.plan.get("cells") or []
            if rec.get("tile_id")
        }
        tiles = sorted(
            granule.manifest.get("tiles") or [],
            key=lambda rec: (int(rec.get("iy", 0)), int(rec.get("ix", 0)), str(rec.get("tile_id"))),
        )
        if not tiles:
            raise ValueError(f"{granule.manifest_path}: manifest has no tiles")
        parent_dates = _dates_from_frames(granule.meta.get("frames") or [])
        for tile in tiles:
            tile_id = str(tile["tile_id"])
            rec = plan_cells.get(tile_id, {})
            original = assignment.get(tile_id, rec.get("icm_date"))
            if not original:
                raise ValueError(
                    f"{granule.plan_path}: no identity assignment for manifest tile {tile_id}"
                )
            embedded = rec.get("date_clouds") or tile.get("date_clouds") or {}
            available = set(embedded)
            available.update(str(d)[:10] for d in (rec.get("dates") or tile.get("dates") or []))
            available.update(_tile_meta_dates(tile, granule.manifest_path))
            availability_source = "cell_or_tile"
            if not available:
                # Parent frames are actual acquisitions, not invented dates. This
                # fallback is recorded in each audit cell for downstream review.
                available = set(parent_dates)
                availability_source = "parent_s2_frames"
            if original:
                # The source identity planner guarantees its own assignment was
                # in-stack; retaining it avoids corrupting an otherwise valid plan.
                available.add(str(original)[:10])
            row_off = int(tile.get("row_off", rec.get("row_off", 0)))
            col_off = int(tile.get("col_off", rec.get("col_off", 0)))
            side = int(tile.get("side", granule.manifest.get("side", 512)))
            local_transform = _parse_transform(tile.get("transform"))
            local_crs = tile.get("crs")
            if local_transform is not None and local_crs:
                local_tx = Transformer.from_crs(local_crs, target, always_xy=True)
                polygon = _cell_polygon(local_transform, local_tx, row_off=0, col_off=0, side=side)
            else:
                polygon = _cell_polygon(
                    transform,
                    transformer,
                    row_off=row_off,
                    col_off=col_off,
                    side=side,
                )
            key = f"{granule.mgrs}/{tile_id}"
            scores = {
                str(day)[:10]: None if score is None else float(score)
                for day, score in embedded.items()
            }
            cell = Cell(
                key=key,
                mgrs=granule.mgrs,
                tile_id=tile_id,
                iy=int(tile.get("iy", rec.get("iy", 0))),
                ix=int(tile.get("ix", rec.get("ix", 0))),
                row_off=row_off,
                col_off=col_off,
                side=side,
                original_date=None if original is None else str(original)[:10],
                available=available,
                polygon=polygon,
                scores=scores,
                availability_source=availability_source,
            )
            if key in cells:
                raise ValueError(f"duplicate cell key: {key}")
            cells[key] = cell
    return cells


def score_cells(cells: dict[str, Cell], granules: list[Granule]) -> None:
    """Fill missing available-date cloud scores from parent SCL rasters."""
    import rasterio
    from rasterio.windows import Window

    by_mgrs = {granule.mgrs: granule for granule in granules}
    for mgrs in sorted(by_mgrs):
        granule = by_mgrs[mgrs]
        group = [cell for cell in cells.values() if cell.mgrs == mgrs]
        frames = {}
        for frame in granule.meta.get("frames") or []:
            try:
                frames[frame_day(frame)] = frame
            except ValueError:
                continue
        wanted = sorted({day for cell in group for day in cell.available if day not in cell.scores})
        for day in wanted:
            frame = frames.get(day)
            scl_value = None if frame is None else frame.get("scl_path")
            path = None if not scl_value else _resolve(scl_value, granule.meta_path.parent)
            targets = [cell for cell in group if day in cell.available and day not in cell.scores]
            if path is None or not path.is_file():
                for cell in targets:
                    cell.scores[day] = None
                continue
            with rasterio.open(path) as src:
                for cell in targets:
                    array = src.read(
                        1,
                        window=Window(cell.col_off, cell.row_off, cell.side, cell.side),
                    )
                    _ok, cloud, _snow, _valid = score_scl_cell(
                        array, max_cloud_frac=1.0, min_valid_frac=0.0
                    )
                    cell.scores[day] = None if cloud != cloud else float(cloud)


def _bbox(poly: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs, ys = zip(*poly)
    return min(xs), min(ys), max(xs), max(ys)


def _cross(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _point_segment_distance(
    point: tuple[float, float], a: tuple[float, float], b: tuple[float, float]
) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    if dx == 0.0 and dy == 0.0:
        return math.hypot(point[0] - a[0], point[1] - a[1])
    t = max(0.0, min(1.0, ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / (dx * dx + dy * dy)))
    nearest = (a[0] + t * dx, a[1] + t * dy)
    return math.hypot(point[0] - nearest[0], point[1] - nearest[1])


def _segments_intersect(a, b, c, d, epsilon: float = 1e-8) -> bool:
    values = (_cross(a, b, c), _cross(a, b, d), _cross(c, d, a), _cross(c, d, b))
    if values[0] * values[1] < -epsilon and values[2] * values[3] < -epsilon:
        return True
    return any(
        abs(value) <= epsilon
        and min(p[0], q[0]) - epsilon <= r[0] <= max(p[0], q[0]) + epsilon
        and min(p[1], q[1]) - epsilon <= r[1] <= max(p[1], q[1]) + epsilon
        for value, p, q, r in (
            (values[0], a, b, c),
            (values[1], a, b, d),
            (values[2], c, d, a),
            (values[3], c, d, b),
        )
    )


def _point_in_polygon(point, poly) -> bool:
    inside = False
    j = len(poly) - 1
    for i, current in enumerate(poly):
        previous = poly[j]
        if (current[1] > point[1]) != (previous[1] > point[1]):
            x_hit = (previous[0] - current[0]) * (point[1] - current[1]) / (
                previous[1] - current[1]
            ) + current[0]
            if point[0] < x_hit:
                inside = not inside
        j = i
    return inside


def polygon_distance(first, second) -> float:
    first_edges = list(zip(first, first[1:] + first[:1]))
    second_edges = list(zip(second, second[1:] + second[:1]))
    if _point_in_polygon(first[0], second) or _point_in_polygon(second[0], first):
        return 0.0
    distance = math.inf
    for a, b in first_edges:
        for c, d in second_edges:
            if _segments_intersect(a, b, c, d):
                return 0.0
            distance = min(
                distance,
                _point_segment_distance(a, c, d),
                _point_segment_distance(b, c, d),
                _point_segment_distance(c, a, b),
                _point_segment_distance(d, a, b),
            )
    return distance


def find_cross_edges(cells: dict[str, Cell], tolerance_m: float) -> list[tuple[str, str, float]]:
    indexed = sorted(((*_bbox(cell.polygon), key) for key, cell in cells.items()))
    edges: list[tuple[str, str, float]] = []
    for index, (minx, miny, maxx, maxy, key) in enumerate(indexed):
        left = cells[key]
        for other_minx, other_miny, other_maxx, other_maxy, other_key in indexed[index + 1 :]:
            if other_minx > maxx + tolerance_m:
                break
            right = cells[other_key]
            if left.mgrs == right.mgrs:
                continue
            if other_miny > maxy + tolerance_m or miny > other_maxy + tolerance_m:
                continue
            distance = polygon_distance(left.polygon, right.polygon)
            if distance <= tolerance_m:
                edges.append((key, other_key, distance))
    return sorted(edges)


def _score_ok(cell: Cell, day: str, slack: float) -> bool:
    value = cell.scores.get(day)
    return value is not None and math.isfinite(value) and value <= slack


def _candidate_dates(left: Cell, right: Cell, slack: float) -> list[str]:
    return sorted(
        day
        for day in left.available & right.available
        if _score_ok(left, day, slack) and _score_ok(right, day, slack)
    )


def _candidate_rank(day: str, left: Cell, right: Cell, center: date) -> tuple:
    scores = (float(left.scores[day]), float(right.scores[day]))
    current_hits = int(day == left.original_date) + int(day == right.original_date)
    return (
        -current_hits,
        max(scores),
        sum(scores),
        abs((date.fromisoformat(day) - center).days),
        day,
    )


def revise_assignments(
    cells: dict[str, Cell],
    edges: list[tuple[str, str, float]],
    *,
    slack: float,
    center: date,
) -> dict[str, str | None]:
    assignment = {key: cell.original_date for key, cell in cells.items()}
    neighbours: dict[str, set[str]] = {key: set() for key in cells}
    for left, right, _distance in edges:
        neighbours[left].add(right)
        neighbours[right].add(left)

    def objective(values: dict[str, str | None]) -> tuple[int, int]:
        cuts = sum(values[left] != values[right] for left, right, _ in edges)
        changes = sum(values[key] != cells[key].original_date for key in values)
        return cuts, changes

    changed = True
    while changed:
        changed = False
        for left_key, right_key, _distance in edges:
            if assignment[left_key] == assignment[right_key]:
                continue
            left, right = cells[left_key], cells[right_key]
            candidates = sorted(
                _candidate_dates(left, right, slack),
                key=lambda day: _candidate_rank(day, left, right, center),
            )
            baseline = objective(assignment)
            best = baseline
            best_values = None
            for day in candidates:
                trial = dict(assignment)
                # Assigning both endpoints is safe because the date was explicitly
                # verified as available and slack-valid in both source cells.
                trial[left_key] = day
                trial[right_key] = day
                value = objective(trial)
                if value < best:
                    best, best_values = value, trial
            if best_values is not None:
                assignment = best_values
                changed = True
    return assignment


def build_audit(
    granules: list[Granule],
    cells: dict[str, Cell],
    edges: list[tuple[str, str, float]],
    revised: dict[str, str | None],
    *,
    common_crs: str,
    tolerance_m: float,
    slack: float,
    center: date,
    audit_only: bool,
) -> dict:
    boundaries = []
    for left_key, right_key, distance in edges:
        left, right = cells[left_key], cells[right_key]
        candidates = sorted(
            _candidate_dates(left, right, slack),
            key=lambda day: _candidate_rank(day, left, right, center),
        )
        common = sorted(left.available & right.available)
        boundaries.append(
            {
                "left": left_key,
                "right": right_key,
                "distance_m": round(distance, 6),
                "overlap_or_touch": distance <= 1e-6,
                "before": [left.original_date, right.original_date],
                "before_cut": left.original_date != right.original_date,
                "after": [revised[left_key], revised[right_key]],
                "after_cut": revised[left_key] != revised[right_key],
                "shared_available_dates": common,
                "slack_valid_candidates": [
                    {
                        "date": day,
                        "left_cloud": left.scores[day],
                        "right_cloud": right.scores[day],
                    }
                    for day in candidates
                ],
                "opportunity": bool(candidates),
                "resolved": (
                    left.original_date != right.original_date
                    and revised[left_key] == revised[right_key]
                ),
                "reason": (
                    "shared_date_slack_valid"
                    if candidates
                    else (
                        "no_shared_available_date"
                        if not common
                        else "shared_dates_fail_slack_or_lack_scl_score"
                    )
                ),
            }
        )
    before_cuts = sum(item["before_cut"] for item in boundaries)
    after_cuts = sum(item["after_cut"] for item in boundaries)
    changed = sorted(key for key in cells if revised[key] != cells[key].original_date)
    return {
        "schema": "scalef.cross_mgrs_identity_audit",
        "version": VERSION,
        "configuration": {
            "common_crs": str(CRS.from_user_input(common_crs)),
            "adjacency_tolerance_m": tolerance_m,
            "slack_cloud": slack,
            "center_date": center.isoformat(),
            "audit_only": audit_only,
        },
        "provenance": {
            "inputs": [
                {
                    "mgrs": granule.mgrs,
                    "identity_plan": str(granule.plan_path),
                    "identity_plan_sha256": _sha256(granule.plan_path),
                    "granule_manifest": str(granule.manifest_path),
                    "granule_manifest_sha256": _sha256(granule.manifest_path),
                    "s2_meta": str(granule.meta_path),
                    "s2_meta_sha256": _sha256(granule.meta_path),
                }
                for granule in granules
            ],
            "date_rule": "candidate must be available and have finite SCL cloud <= slack in both cells",
            "unavailable_dates_forced": False,
        },
        "summary": {
            "n_granules": len(granules),
            "n_cells": len(cells),
            "n_cross_edges": len(edges),
            "n_boundary_cuts_before": before_cuts,
            "n_shared_date_opportunities": sum(item["opportunity"] for item in boundaries),
            "n_boundary_cuts_after": after_cuts,
            "n_cells_revised": len(changed),
        },
        "changed_cells": [
            {
                "cell": key,
                "before": cells[key].original_date,
                "after": revised[key],
            }
            for key in changed
        ],
        "cells": [
            {
                "cell": key,
                "mgrs": cell.mgrs,
                "tile_id": cell.tile_id,
                "availability_source": cell.availability_source,
                "available_dates": sorted(cell.available),
            }
            for key, cell in sorted(cells.items())
        ],
        "boundaries": boundaries,
    }


def revised_plan(
    granule: Granule, cells: dict[str, Cell], revised: dict[str, str | None], audit: dict
) -> dict:
    out = json.loads(json.dumps(granule.plan))
    assignment = dict(out.get("assignment") or {})
    relevant = {cell.tile_id: cell for cell in cells.values() if cell.mgrs == granule.mgrs}
    for tile_id, cell in relevant.items():
        assignment[tile_id] = revised[cell.key]
    out["assignment"] = {key: assignment[key] for key in sorted(assignment)}
    for rec in out.get("cells") or []:
        tile_id = rec.get("tile_id")
        if tile_id not in relevant:
            continue
        cell = relevant[tile_id]
        rec["cross_mgrs_original_date"] = cell.original_date
        rec["icm_date"] = revised[cell.key]
        rec["cross_mgrs_revised"] = revised[cell.key] != cell.original_date
    out["icm_counts"] = dict(Counter(out["assignment"].values()))
    out["cross_mgrs_identity"] = {
        "schema_version": VERSION,
        "source_identity_plan_sha256": _sha256(granule.plan_path),
        "audit_report": "../cross_mgrs_identity_audit.json",
        "common_crs": audit["configuration"]["common_crs"],
        "adjacency_tolerance_m": audit["configuration"]["adjacency_tolerance_m"],
        "slack_cloud": audit["configuration"]["slack_cloud"],
        "unavailable_dates_forced": False,
        "n_revised": sum(revised[cell.key] != cell.original_date for cell in relevant.values()),
    }
    return out


def _write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as dst:
            dst.write(data)
            dst.flush()
            os.fsync(dst.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_outputs(
    out_dir: Path,
    granules: list[Granule],
    cells: dict[str, Cell],
    revised: dict[str, str | None],
    audit: dict,
) -> None:
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} already exists; refusing a non-atomic replacement")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.", dir=out_dir.parent))
    try:
        for granule in granules:
            _write_atomic(
                staging / granule.mgrs / "identity_plan.json",
                revised_plan(granule, cells, revised, audit),
            )
        _write_atomic(staging / "cross_mgrs_identity_audit.json", audit)
        os.replace(staging, out_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _input_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for values in args.granule or []:
        mgrs, identity_plan, granule_manifest, s2_meta = values
        specs.append(
            {
                "mgrs": mgrs,
                "identity_plan": identity_plan,
                "granule_manifest": granule_manifest,
                "s2_meta": s2_meta,
                "_base": Path.cwd(),
            }
        )
    for values in args.input or []:
        identity_plan, granule_manifest, s2_meta = values
        specs.append(
            {
                "identity_plan": identity_plan,
                "granule_manifest": granule_manifest,
                "s2_meta": s2_meta,
                "_base": Path.cwd(),
            }
        )
    if args.inputs_manifest:
        path = args.inputs_manifest.resolve()
        raw = _json(path)
        records = raw.get("granules") if isinstance(raw, dict) else raw
        if not isinstance(records, list):
            raise ValueError(f"{path}: expected a list or an object with 'granules'")
        for record in records:
            specs.append({**record, "_base": path.parent})
    return specs


def run(args: argparse.Namespace) -> dict:
    granules = load_granules(_input_specs(args))
    cells = build_cells(granules, args.common_crs)
    score_cells(cells, granules)
    edges = find_cross_edges(cells, args.adjacency_tolerance_m)
    center = date.fromisoformat(args.center_date)
    revised = revise_assignments(cells, edges, slack=args.slack_cloud, center=center)
    audit = build_audit(
        granules,
        cells,
        edges,
        revised,
        common_crs=args.common_crs,
        tolerance_m=args.adjacency_tolerance_m,
        slack=args.slack_cloud,
        center=center,
        audit_only=args.audit_only,
    )
    if args.audit_only:
        report = args.report or Path("cross_mgrs_identity_audit.json")
        _write_atomic(report.resolve(), audit)
    else:
        if args.out_dir is None:
            raise ValueError("--out-dir is required unless --audit-only is used")
        write_outputs(args.out_dir.resolve(), granules, cells, revised, audit)
    return audit


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--granule",
        nargs=4,
        action="append",
        metavar=("MGRS", "IDENTITY_PLAN", "GRANULE_MANIFEST", "S2_META"),
        help="Repeat for each granule.",
    )
    parser.add_argument(
        "--input",
        nargs=3,
        action="append",
        metavar=("IDENTITY_PLAN", "GRANULE_MANIFEST", "S2_META"),
        help="Repeat for each granule; MGRS is inferred.",
    )
    parser.add_argument(
        "--inputs-manifest",
        type=Path,
        help="JSON list (or {'granules': [...]}) of input records.",
    )
    parser.add_argument("--common-crs", default="EPSG:3857")
    parser.add_argument("--adjacency-tolerance-m", type=float, default=1.0)
    parser.add_argument("--slack-cloud", type=float, default=0.05)
    parser.add_argument("--center-date", default="2025-07-15")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--report", type=Path, help="Audit report path in audit-only mode.")
    args = parser.parse_args(argv)
    if not (args.granule or args.input or args.inputs_manifest):
        parser.error("provide repeated --granule/--input or --inputs-manifest")
    if args.adjacency_tolerance_m < 0:
        parser.error("--adjacency-tolerance-m must be non-negative")
    if not 0 <= args.slack_cloud <= 1:
        parser.error("--slack-cloud must be between 0 and 1")
    return args


def main() -> None:
    try:
        audit = run(parse_args())
    except (FileNotFoundError, ValueError, KeyError) as exc:
        raise SystemExit(str(exc)) from exc
    summary = audit["summary"]
    print(
        "cross edges={n_cross_edges} cuts={n_boundary_cuts_before}"
        "->{n_boundary_cuts_after} opportunities={n_shared_date_opportunities} "
        "revised={n_cells_revised}".format(**summary)
    )


if __name__ == "__main__":
    main()
