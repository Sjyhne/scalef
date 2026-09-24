#!/usr/bin/env python3
"""Select a deterministic contiguous Norway MGRS demo set from planning artifacts.

The selector combines one window from ``compare_norway_season_windows.py`` with
per-MGRS LR512 heatmaps. Heatmap cells are qualified against the mainland mask
before scoring, or precomputed synthetic/operational counts may be supplied via
``--qualified-counts``. ``--mode block`` favors geographic compactness;
``--mode connected`` favors score while requiring graph connectivity.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import xy
from rasterio.warp import transform as warp_xy
from rasterio.warp import transform_bounds

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.land_mask_lr512 import DEFAULT_LAND_MASK, land_hit_frac, load_land_paths


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _tile_rows(payload: dict) -> list[dict]:
    date_range = payload.get("date_range") or (payload.get("meta") or {}).get("date_range")
    rows = payload.get("tiles")
    if isinstance(rows, dict):
        out = [{"mgrs_tile": key, **value} for key, value in rows.items()]
    else:
        out = list(rows or payload.get("results") or [])
    if date_range:
        for row in out:
            row.setdefault("date_range", date_range)
    return out


def qualify_heatmap(
    path: Path,
    *,
    land_paths: list,
    min_clear: int,
    min_land_frac: float,
    sample_n: int,
) -> dict:
    """Count LR cells meeting land and clear rules in one clear-count GeoTIFF."""
    with rasterio.open(path) as src:
        counts = src.read(1)
        valid = np.ones(counts.shape, dtype=bool)
        if src.nodata is not None:
            valid &= counts != src.nodata
        qualified = np.zeros(counts.shape, dtype=bool)
        n = max(2, int(sample_n))
        offsets = (np.arange(n) + 0.5) / n - 0.5
        for row in range(src.height):
            for col in range(src.width):
                if not valid[row, col]:
                    continue
                rr, cc = np.meshgrid(row + 0.5 + offsets, col + 0.5 + offsets, indexing="ij")
                xs, ys = xy(src.transform, rr.ravel(), cc.ravel(), offset="ul")
                lons, lats = warp_xy(src.crs, "EPSG:4326", xs, ys)
                frac = land_hit_frac(
                    land_paths,
                    np.column_stack([np.asarray(lons), np.asarray(lats)]),
                )
                qualified[row, col] = (
                    frac > 0 if min_land_frac <= 0 else frac >= min_land_frac
                )
        vals = counts[qualified]
        bbox = list(transform_bounds(src.crs, "EPSG:4326", *src.bounds))
    return {
        "mgrs_tile": path.name.split("_")[0],
        "mainland_cells_total": int(qualified.sum()),
        "mainland_cells_ge_min_clear": int((vals >= min_clear).sum()),
        "clear_count_mean_mainland": float(vals.mean()) if vals.size else 0.0,
        "bbox": bbox,
        "heatmap": str(path),
    }


def load_heatmap_records(
    heatmap_dir: Path,
    *,
    qualified_counts: Path | None,
    land_mask: Path,
    min_clear: int,
    min_land_frac: float,
    sample_n: int,
) -> list[dict]:
    if qualified_counts is not None:
        return _tile_rows(_load_json(qualified_counts))
    paths = sorted(heatmap_dir.glob("*_lr512_clear_counts.tif"))
    if not paths:
        raise ValueError(f"no *_lr512_clear_counts.tif in {heatmap_dir}")
    land_paths = load_land_paths(land_mask)
    rows = [
        qualify_heatmap(
            path,
            land_paths=land_paths,
            min_clear=min_clear,
            min_land_frac=min_land_frac,
            sample_n=sample_n,
        )
        for path in paths
    ]
    summary_path = heatmap_dir / "summary.json"
    if summary_path.is_file():
        date_range = _load_json(summary_path).get("date_range")
        if date_range:
            for row in rows:
                row["date_range"] = date_range
    return rows


def season_rows(payload: dict, window_name: str | None) -> tuple[dict, dict[str, dict]]:
    windows = ((payload.get("stac") or {}).get("windows") or [])
    if not windows:
        raise ValueError("season artifact has no stac.windows")
    name = window_name or (payload.get("recommendation") or {}).get("top")
    if name is None:
        name = windows[0]["name"]
    matches = [window for window in windows if window.get("name") == name]
    if not matches:
        raise ValueError(f"season window {name!r} not found")
    window = matches[0]
    return window, {
        str(row["mgrs_tile"]).upper().lstrip("T"): row
        for row in window.get("tiles") or []
    }


def score_records(
    heatmap_rows: list[dict],
    season_by_tile: dict[str, dict],
    *,
    min_clear: int,
) -> list[dict]:
    max_clear = max(
        [float(row.get("clear_days") or 0) for row in season_by_tile.values()] or [1.0]
    )
    max_cells = max(
        [int(row.get("mainland_cells_total") or 0) for row in heatmap_rows] or [1]
    )
    out = []
    for heat in heatmap_rows:
        mgrs = str(heat["mgrs_tile"]).upper().lstrip("T")
        if mgrs not in season_by_tile:
            continue
        season = season_by_tile[mgrs]
        total = int(heat.get("mainland_cells_total") or 0)
        good = int(heat.get("mainland_cells_ge_min_clear") or 0)
        coverage = good / total if total else 0.0
        season_days = float(season.get("clear_days") or 0)
        area = total / max(max_cells, 1)
        score = 0.35 * (season_days / max(max_clear, 1.0)) + 0.45 * coverage + 0.20 * area
        reasons = [
            f"{season_days:g} scene-clear days in selected season",
            f"{good}/{total} mainland LR512 cells have >= {min_clear} clear days",
            f"mainland-qualified coverage={coverage:.3f}",
        ]
        out.append(
            {
                **heat,
                "mgrs_tile": mgrs,
                "season_clear_days": season_days,
                "qualified_clear_fraction": coverage,
                "score": round(score, 8),
                "reasons": reasons,
                "bbox": heat.get("bbox") or season.get("bbox"),
            }
        )
    return sorted(out, key=lambda row: (-row["score"], row["mgrs_tile"]))


def _haversine(a: tuple[float, float], b: tuple[float, float]) -> float:
    lon1, lat1, lon2, lat2 = map(math.radians, (*a, *b))
    dlon, dlat = lon2 - lon1, lat2 - lat1
    value = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(value))


def build_adjacency(rows: list[dict], *, factor: float = 1.65) -> dict[str, set[str]]:
    """Build deterministic four-neighbor/grid or geographic-neighbor adjacency."""
    if len(rows) < 2:
        raise ValueError("at least two scored tiles are required for adjacency")
    ids = [row["mgrs_tile"] for row in rows]
    graph = {mgrs: set() for mgrs in ids}
    by_id = {row["mgrs_tile"]: row for row in rows}
    if all(row.get("grid_x") is not None and row.get("grid_y") is not None for row in rows):
        pos = {(int(row["grid_x"]), int(row["grid_y"])): row["mgrs_tile"] for row in rows}
        for (x, y), mgrs in pos.items():
            for neighbor_pos in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if neighbor_pos in pos:
                    graph[mgrs].add(pos[neighbor_pos])
        return graph
    centers = {}
    for row in rows:
        bbox = row.get("bbox")
        if bbox and len(bbox) == 4:
            centers[row["mgrs_tile"]] = (
                0.5 * (float(bbox[0]) + float(bbox[2])),
                0.5 * (float(bbox[1]) + float(bbox[3])),
            )
    if len(centers) != len(rows):
        missing = sorted(set(ids) - set(centers))
        raise ValueError(f"missing bbox or grid_x/grid_y for adjacency: {missing[:5]}")
    nearest = []
    for mgrs, center in centers.items():
        nearest.append(min(_haversine(center, other) for key, other in centers.items() if key != mgrs))
    threshold = float(np.median(nearest)) * factor
    for i, mgrs in enumerate(ids):
        for other in ids[i + 1 :]:
            if _haversine(centers[mgrs], centers[other]) <= threshold:
                graph[mgrs].add(other)
                graph[other].add(mgrs)
    return graph


def _compactness(chosen: frozenset[str], by_id: dict[str, dict]) -> float:
    coords = []
    for mgrs in chosen:
        row = by_id[mgrs]
        if row.get("grid_x") is not None:
            coords.append((float(row["grid_x"]), float(row["grid_y"])))
        else:
            bbox = row["bbox"]
            coords.append((0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3])))
    xs, ys = zip(*coords)
    span_x, span_y = max(xs) - min(xs), max(ys) - min(ys)
    return 1.0 / (1.0 + span_x * span_y + 0.25 * (span_x + span_y))


def select_connected(
    rows: list[dict],
    graph: dict[str, set[str]],
    *,
    count: int,
    mode: str,
    beam_width: int = 5000,
) -> list[str]:
    by_id = {row["mgrs_tile"]: row for row in rows}

    def objective(group: frozenset[str]) -> tuple[float, tuple[str, ...]]:
        mean_score = sum(float(by_id[key]["score"]) for key in group) / len(group)
        compact = _compactness(group, by_id)
        value = mean_score + (0.18 if mode == "block" else 0.03) * compact
        return value, tuple(sorted(group))

    frontier = {frozenset([mgrs]) for mgrs in sorted(by_id)}
    for _ in range(1, count):
        grown = set()
        for group in frontier:
            boundary = set().union(*(graph[key] for key in group)) - set(group)
            grown.update(group | {neighbor} for neighbor in boundary)
        if not grown:
            raise ValueError(f"no connected set of {count} tiles")
        frontier = set(
            sorted(grown, key=lambda group: (-objective(group)[0], objective(group)[1]))[:beam_width]
        )
    best = sorted(frontier, key=lambda group: (-objective(group)[0], objective(group)[1]))[0]
    return sorted(best)


def build_manifest(
    *,
    season_payload: dict,
    window_name: str | None,
    heatmap_rows: list[dict],
    count: int,
    mode: str,
    min_clear: int,
    adjacency_factor: float,
) -> dict:
    window, season_by = season_rows(season_payload, window_name)
    heatmap_ranges = {
        row.get("date_range") for row in heatmap_rows if row.get("date_range")
    }
    if heatmap_ranges and heatmap_ranges != {window["date_range"]}:
        raise ValueError(
            f"heatmap date range(s) {sorted(heatmap_ranges)} do not match "
            f"season window {window['date_range']}"
        )
    scored = score_records(heatmap_rows, season_by, min_clear=min_clear)
    if len(scored) < count:
        raise ValueError(
            f"only {len(scored)} tiles have both season and heatmap artifacts; "
            f"cannot select {count}"
        )
    graph = build_adjacency(scored, factor=adjacency_factor)
    selected_ids = select_connected(scored, graph, count=count, mode=mode)
    selected_set = set(selected_ids)
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection": {
            "mode": mode,
            "count": count,
            "min_clear": min_clear,
            "window_name": window["name"],
            "date_range": window["date_range"],
            "score_formula": "0.35*season_clear_norm + 0.45*qualified_clear_fraction + 0.20*mainland_area_norm",
            "adjacency_factor": adjacency_factor,
        },
        "selected_mgrs": selected_ids,
        "tiles": [row for row in scored if row["mgrs_tile"] in selected_set],
        "adjacency": {key: sorted(value) for key, value in graph.items()},
        "ranked_candidates": scored,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season-artifact", type=Path, required=True)
    parser.add_argument("--heatmap-dir", type=Path, required=True)
    parser.add_argument("--qualified-counts", type=Path, default=None)
    parser.add_argument("--window", default=None, help="Season window name (default: recommendation.top).")
    parser.add_argument("--count", type=int, choices=range(6, 10), default=6)
    parser.add_argument("--mode", choices=["block", "connected"], default="block")
    parser.add_argument("--min-clear", type=int, default=6)
    parser.add_argument("--land-mask", type=Path, default=DEFAULT_LAND_MASK)
    parser.add_argument("--min-land-frac", type=float, default=0.0)
    parser.add_argument("--land-sample-n", type=int, default=9)
    parser.add_argument("--adjacency-factor", type=float, default=1.65)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "production" / "national_demo" / "mgrs_selection.json",
    )
    args = parser.parse_args()
    resolve = lambda path: path if path.is_absolute() else ROOT / path
    season_path = resolve(args.season_artifact)
    heatmap_dir = resolve(args.heatmap_dir)
    qualified = None if args.qualified_counts is None else resolve(args.qualified_counts)
    rows = load_heatmap_records(
        heatmap_dir,
        qualified_counts=qualified,
        land_mask=resolve(args.land_mask),
        min_clear=args.min_clear,
        min_land_frac=args.min_land_frac,
        sample_n=args.land_sample_n,
    )
    manifest = build_manifest(
        season_payload=_load_json(season_path),
        window_name=args.window,
        heatmap_rows=rows,
        count=args.count,
        mode=args.mode,
        min_clear=args.min_clear,
        adjacency_factor=args.adjacency_factor,
    )
    manifest["inputs"] = {
        "season_artifact": str(season_path),
        "heatmap_dir": str(heatmap_dir),
        "qualified_counts": None if qualified is None else str(qualified),
        "land_mask": str(resolve(args.land_mask)),
    }
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Selected {manifest['selected_mgrs']} -> {out}")


if __name__ == "__main__":
    main()
