#!/usr/bin/env python3
"""Harmonize LR512 radiometry and hide residual cuts in a delivery mosaic.

Does not retrain. An optional graph solve estimates bounded additive RGB
corrections from robust statistics at assignment changes and ICM-joined
independent-base changes. Fully consistent same-date edges only smooth the
correction field. A cosine ramp then equalizes residual targeted edge strips.
This is not overlap-feather: there is still only one prediction per pixel.

Example::
    python scripts/mosaic_seam_ramp.py \\
      --manifest data/s2_revisits/national_2025_v2/32VNM/granule_tiles_lr512_manifest.json \\
      --run-prefix prod_k4_base2 \\
      --ramp-px 128 \\
      --out production/mosaics/32VNM_base2_seamramp_sr_2p5m.tif
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CELL_HR = 2048
EDGE = 8
HARMONIZE_STRIP = 256
HARMONIZE_SEGMENTS = 16


def _sr_path(parent: str, tile_id: str, prefix: str) -> Path:
    return ROOT / "single_samples" / parent / "sample" / f"{prefix}_{tile_id}" / "qgis" / "sr_pred.tif"


def _tile_id(parent: str, iy: int, ix: int) -> str:
    return f"{parent}_t512_y{iy:02d}_x{ix:02d}"


def _smooth_ramp(dist: np.ndarray, ramp_px: int) -> np.ndarray:
    u = np.clip(1.0 - dist / float(max(ramp_px, 1)), 0.0, 1.0)
    return 0.5 * (1.0 - np.cos(np.pi * u)).astype(np.float32)


def _read_rgb(path: Path) -> np.ndarray:
    import rasterio

    with rasterio.open(path) as src:
        return np.transpose(src.read(indexes=(1, 2, 3)), (1, 2, 0)).astype(np.float32)


def _read_strip(path: Path, *, side: str) -> np.ndarray:
    import rasterio
    from rasterio.windows import Window

    with rasterio.open(path) as src:
        h, w = src.height, src.width
        if side == "right":
            win = Window(w - EDGE, 0, EDGE, h)
        elif side == "left":
            win = Window(0, 0, EDGE, h)
        elif side == "bottom":
            win = Window(0, h - EDGE, w, EDGE)
        else:
            win = Window(0, 0, w, EDGE)
        arr = np.transpose(src.read(indexes=(1, 2, 3), window=win), (1, 2, 0))
    return arr.astype(np.float32)


def _read_harmonize_strip(path: Path, *, side: str, strip_px: int) -> np.ndarray:
    """Read a broad inward strip with the along-boundary axis first."""
    import rasterio
    from rasterio.windows import Window

    with rasterio.open(path) as src:
        h, w = src.height, src.width
        width = max(1, min(int(strip_px), w))
        height = max(1, min(int(strip_px), h))
        if side == "right":
            win = Window(w - width, 0, width, h)
        elif side == "left":
            win = Window(0, 0, width, h)
        elif side == "bottom":
            win = Window(0, h - height, w, height)
        elif side == "top":
            win = Window(0, 0, w, height)
        else:
            raise ValueError(f"unknown strip side: {side}")
        arr = np.transpose(src.read(indexes=(1, 2, 3), window=win), (1, 2, 0))
    arr = arr.astype(np.float32)
    return arr if side in {"left", "right"} else np.transpose(arr, (1, 0, 2))


def _segment_rgb_medians(strip: np.ndarray, segments: int) -> np.ndarray:
    """Return robust RGB summaries along an edge, excluding nodata."""
    records = []
    for chunk in np.array_split(strip, max(1, min(int(segments), strip.shape[0])), axis=0):
        valid = np.all(np.isfinite(chunk), axis=2) & np.any(chunk > 0, axis=2)
        if valid.any():
            records.append(np.median(chunk[valid], axis=0))
    if not records:
        return np.empty((0, 3), dtype=np.float32)
    return np.asarray(records, dtype=np.float32)


def _measure_edge(
    first: Path,
    second: Path,
    *,
    first_side: str,
    second_side: str,
    strip_px: int,
    segments: int,
) -> tuple[np.ndarray, float, int, np.ndarray, np.ndarray] | None:
    left = _segment_rgb_medians(
        _read_harmonize_strip(first, side=first_side, strip_px=strip_px), segments
    )
    right = _segment_rgb_medians(
        _read_harmonize_strip(second, side=second_side, strip_px=strip_px), segments
    )
    n = min(len(left), len(right))
    if n < max(2, segments // 4):
        return None
    differences = right[:n] - left[:n]
    delta = np.median(differences, axis=0).astype(np.float32)
    dispersion = float(np.median(np.abs(differences - delta), axis=None))
    completeness = min(1.0, n / max(float(segments), 1.0))
    weight = completeness * 0.01 / (0.01 + dispersion)
    return delta, max(weight, 0.05), n, left[:n], right[:n]


def _edge_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def estimate_harmonization(
    grid: dict[tuple[int, int], Path],
    *,
    dates: dict[tuple[int, int], str | None] | None = None,
    independent_dates: dict[tuple[int, int], str | None] | None = None,
    strip_px: int = HARMONIZE_STRIP,
    segments: int = HARMONIZE_SEGMENTS,
    regularization: float = 0.02,
    max_offset: float = 0.08,
    smoothness: float = 0.1,
) -> dict:
    """Solve bounded whole-cell additive RGB transforms over the adjacency graph."""
    nodes = sorted(grid)
    index = {xy: i for i, xy in enumerate(nodes)}
    measured = []
    for xy in nodes:
        iy, ix = xy
        for neighbour, sides, orientation in (
            ((iy, ix + 1), ("right", "left"), "vertical"),
            ((iy + 1, ix), ("bottom", "top"), "horizontal"),
        ):
            if neighbour not in grid:
                continue
            result = _measure_edge(
                grid[xy],
                grid[neighbour],
                first_side=sides[0],
                second_side=sides[1],
                strip_px=strip_px,
                segments=segments,
            )
            if result is None:
                continue
            delta, weight, n_segments, first_profile, second_profile = result
            measured.append(
                {
                    "first": xy,
                    "second": neighbour,
                    "orientation": orientation,
                    "delta": delta,
                    "weight": weight,
                    "n_segments": n_segments,
                    "first_profile": first_profile,
                    "second_profile": second_profile,
                    "date_cut": dates is not None and dates.get(xy) != dates.get(neighbour),
                    "independent_cut": (
                        independent_dates is not None
                        and independent_dates.get(xy) != independent_dates.get(neighbour)
                    ),
                }
            )
    if not measured:
        raise ValueError("no valid adjacent edges for radiometric harmonization")

    from scipy import sparse
    from scipy.sparse.linalg import lsqr

    # Find shared-date components for audit. The solve below uses only observed
    # deltas at date cuts and a smooth correction prior inside each component.
    parent = {xy: xy for xy in nodes}

    def find(xy: tuple[int, int]) -> tuple[int, int]:
        while parent[xy] != xy:
            parent[xy] = parent[parent[xy]]
            xy = parent[xy]
        return xy

    def union(first: tuple[int, int], second: tuple[int, int]) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parent[max(first_root, second_root)] = min(first_root, second_root)

    if dates is not None:
        for iy, ix in nodes:
            for neighbour in ((iy, ix + 1), (iy + 1, ix)):
                if neighbour in grid and dates.get((iy, ix)) == dates.get(neighbour):
                    union((iy, ix), neighbour)
    roots = sorted({find(xy) for xy in nodes})
    constrained_edges = (
        [
            edge
            for edge in measured
            if edge["date_cut"] or edge["independent_cut"]
        ]
        if dates is not None
        else measured
    )

    # Date cuts carry observed RGB deltas. Same-date edges carry only a smooth
    # correction-field prior with a zero target; their scene radiometry is
    # never used as a balancing target.
    solved_edges = measured
    rows = np.repeat(np.arange(len(solved_edges)), 2)
    columns = np.asarray(
        [
            value
            for edge in solved_edges
            for value in (index[edge["first"]], index[edge["second"]])
        ]
    )
    values = np.tile(np.asarray([1.0, -1.0]), len(solved_edges))
    matrix = sparse.csr_matrix(
        (values, (rows, columns)), shape=(len(solved_edges), len(nodes))
    )
    targets = np.asarray(
        [
            (
                edge["delta"]
                if dates is None or edge["date_cut"] or edge["independent_cut"]
                else np.zeros(3)
            )
            for edge in solved_edges
        ],
        dtype=np.float64,
    )
    weights = np.sqrt(
        np.asarray(
            [
                edge["weight"]
                if dates is None or edge["date_cut"] or edge["independent_cut"]
                else float(smoothness)
                for edge in solved_edges
            ],
            dtype=np.float64,
        )
    )
    design = sparse.diags(weights) @ matrix
    response = targets * weights[:, None]
    if regularization > 0:
        design = sparse.vstack(
            [design, np.sqrt(float(regularization)) * sparse.eye(len(nodes))],
            format="csr",
        )
        response = np.vstack(
            [response, np.zeros((len(nodes), 3), dtype=np.float64)]
        )
    corrections = np.column_stack(
        [
            lsqr(design, response[:, channel], atol=1e-9, btol=1e-9)[0]
            for channel in range(3)
        ]
    )
    corrections -= corrections.mean(axis=0, keepdims=True)
    max_unclipped = float(np.max(np.abs(corrections)))
    n_clipped_components = int(
        np.count_nonzero(np.abs(corrections) > float(max_offset))
    )
    corrections = np.clip(corrections, -float(max_offset), float(max_offset))

    before_all, after_all, before_cut, after_cut = [], [], [], []
    before_risk, after_risk = [], []
    edge_records = []
    for edge in measured:
        i, j = index[edge["first"]], index[edge["second"]]
        before = float(np.mean(np.abs(edge["delta"])))
        residual = edge["delta"] - (corrections[i] - corrections[j])
        after = float(np.mean(np.abs(residual)))
        before_all.append(before)
        after_all.append(after)
        if edge["date_cut"]:
            before_cut.append(before)
            after_cut.append(after)
        if edge["date_cut"] or edge["independent_cut"]:
            before_risk.append(before)
            after_risk.append(after)
        edge_records.append(
            {
                "first": list(edge["first"]),
                "second": list(edge["second"]),
                "orientation": edge["orientation"],
                "date_cut": edge["date_cut"],
                "independent_cut": edge["independent_cut"],
                "delta_rgb": [float(v) for v in edge["delta"]],
                "weight": float(edge["weight"]),
                "n_segments": int(edge["n_segments"]),
                "residual_rgb": [float(v) for v in residual],
            }
        )
    correction_map = {
        xy: corrections[index[xy]].astype(np.float32) for xy in nodes
    }
    return {
        "corrections": correction_map,
        "records": edge_records,
        "metrics": {
            "all_edges_before": _edge_summary(before_all),
            "all_edges_after_global": _edge_summary(after_all),
            "date_cuts_before": _edge_summary(before_cut),
            "date_cuts_after_global": _edge_summary(after_cut),
            "identity_risk_edges_before": _edge_summary(before_risk),
            "identity_risk_edges_after_global": _edge_summary(after_risk),
            "max_unclipped_abs_offset": max_unclipped,
            "n_clipped_components": n_clipped_components,
            "max_abs_offset": float(np.max(np.abs(corrections))),
            "mean_abs_offset": float(np.mean(np.abs(corrections))),
            "n_date_components": len(roots),
            "n_constrained_identity_risk_edges": len(constrained_edges),
        },
        "configuration": {
            "method": "robust_identity_risk_graph_additive_rgb_v4",
            "constraint_scope": (
                "assignment_or_independent_base_cuts"
                if dates is not None
                else "all_edges"
            ),
            "strip_px": int(strip_px),
            "segments": int(segments),
            "regularization": float(regularization),
            "same_date_correction_smoothness": float(smoothness),
            "max_offset": float(max_offset),
        },
    }


def _apply_offset(array: np.ndarray, offset: np.ndarray | None) -> np.ndarray:
    if offset is None:
        return array
    out = array.copy()
    valid = np.all(np.isfinite(out), axis=2) & np.any(out > 0, axis=2)
    out[valid] = np.clip(out[valid] + offset[None, :], 0.0, 1.0)
    return out


def _apply_vertical(left: np.ndarray, right: np.ndarray, ramp_px: int) -> tuple[np.ndarray, np.ndarray]:
    dl = left[:, -EDGE:, :].mean(axis=1)
    dr = right[:, :EDGE, :].mean(axis=1)
    delta = 0.5 * (dr - dl).astype(np.float32)
    xs = np.arange(ramp_px, dtype=np.float32)
    w_left = _smooth_ramp(xs[::-1], ramp_px)[None, :, None]
    w_right = _smooth_ramp(xs, ramp_px)[None, :, None]
    out_l = left.copy()
    out_r = right.copy()
    out_l[:, -ramp_px:, :] = np.clip(out_l[:, -ramp_px:, :] + delta[:, None, :] * w_left, 0.0, 1.0)
    out_r[:, :ramp_px, :] = np.clip(out_r[:, :ramp_px, :] - delta[:, None, :] * w_right, 0.0, 1.0)
    return out_l, out_r


def _apply_horizontal(top: np.ndarray, bottom: np.ndarray, ramp_px: int) -> tuple[np.ndarray, np.ndarray]:
    dt = top[-EDGE:, :, :].mean(axis=0)
    db = bottom[:EDGE, :, :].mean(axis=0)
    delta = 0.5 * (db - dt).astype(np.float32)
    ys = np.arange(ramp_px, dtype=np.float32)
    w_top = _smooth_ramp(ys[::-1], ramp_px)[:, None, None]
    w_bot = _smooth_ramp(ys, ramp_px)[:, None, None]
    out_t = top.copy()
    out_b = bottom.copy()
    out_t[-ramp_px:, :, :] = np.clip(out_t[-ramp_px:, :, :] + delta[None, :, :] * w_top, 0.0, 1.0)
    out_b[:ramp_px, :, :] = np.clip(out_b[:ramp_px, :, :] - delta[None, :, :] * w_bot, 0.0, 1.0)
    return out_t, out_b


def _edge_mae(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.abs(left[:, -1, :] - right[:, 0, :]).mean())


def _stretch(rgb: np.ndarray, lo, hi) -> np.ndarray:
    return np.clip((rgb - lo) / np.maximum(hi - lo, 1e-6), 0.0, 1.0)


def _collect_grid(manifest: dict, prefix: str) -> dict[tuple[int, int], Path]:
    parent = manifest["parent"]
    grid = {}
    for tile in manifest["tiles"]:
        tid = tile["tile_id"]
        parts = tid.split("_")
        iy, ix = int(parts[-2][1:]), int(parts[-1][1:])
        path = _sr_path(parent, tid, prefix)
        if path.is_file():
            grid[iy, ix] = path
    return grid


def _collect_delivery_grid(
    manifest: dict,
    plan: dict,
    *,
    changed_prefix: str,
    fallback_prefix: str,
) -> tuple[dict[tuple[int, int], Path], dict[tuple[int, int], str | None]]:
    """Select changed ICM outputs and unchanged independent-2% outputs."""
    parent = manifest["parent"]
    assignment = plan.get("assignment") or {}
    independent = plan.get("independent") or {}
    grid: dict[tuple[int, int], Path] = {}
    dates: dict[tuple[int, int], str | None] = {}
    missing = []
    for tile in manifest.get("tiles") or []:
        tile_id = tile["tile_id"]
        if tile_id not in assignment:
            continue
        iy, ix = int(tile["iy"]), int(tile["ix"])
        changed = assignment[tile_id] != independent.get(tile_id)
        prefix = changed_prefix if changed else fallback_prefix
        path = _sr_path(parent, tile_id, prefix)
        if not path.is_file():
            missing.append(
                {
                    "tile_id": tile_id,
                    "changed": changed,
                    "expected": str(path.relative_to(ROOT)),
                }
            )
            continue
        grid[iy, ix] = path
        dates[iy, ix] = assignment[tile_id]
    if missing:
        sample = ", ".join(item["tile_id"] for item in missing[:5])
        raise FileNotFoundError(
            f"missing {len(missing)} planned SR tiles ({sample}); "
            f"changed prefix={changed_prefix}, fallback={fallback_prefix}"
        )
    return grid, dates


def _is_date_cut(
    dates: dict[tuple[int, int], str | None] | None,
    xy: tuple[int, int],
    neighbour: tuple[int, int],
) -> bool:
    """Without a plan preserve legacy all-edge behavior; with a plan ramp cuts only."""
    return dates is None or dates.get(xy) != dates.get(neighbour)


def _is_balance_cut(
    dates: dict[tuple[int, int], str | None] | None,
    independent_dates: dict[tuple[int, int], str | None] | None,
    xy: tuple[int, int],
    neighbour: tuple[int, int],
) -> bool:
    """Target assignment changes and ICM-joined independent-base changes."""
    return _is_date_cut(dates, xy, neighbour) or (
        independent_dates is not None
        and independent_dates.get(xy) != independent_dates.get(neighbour)
    )


def write_preview(
    grid: dict[tuple[int, int], Path],
    *,
    ramp_px: int,
    out: Path,
) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pair = ((4, 17), (4, 18))
    left = _read_rgb(grid[pair[0]])
    right = _read_rgb(grid[pair[1]])
    mae0 = _edge_mae(left, right)
    left_a, right_a = _apply_vertical(left, right, ramp_px)
    mae1 = _edge_mae(left_a, right_a)

    pad = 220
    cut = left.shape[1]
    sl = slice(int(0.42 * left.shape[0]), int(0.42 * left.shape[0]) + 480)
    before = np.concatenate([left[sl, -pad:], right[sl, :pad]], axis=1)
    after = np.concatenate([left_a[sl, -pad:], right_a[sl, :pad]], axis=1)
    stack = np.concatenate([before.reshape(-1, 3), after.reshape(-1, 3)], axis=0)
    lo, hi = np.percentile(stack, [2, 98], axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 6.2))
    for ax, img, title in (
        (axes[0], before, f"base2 hard cut  ·  edge MAE {mae0:.4f}"),
        (axes[1], after, f"seam ramp {ramp_px} px  ·  edge MAE {mae1:.4f}"),
    ):
        ax.imshow(_stretch(img, lo, hi), interpolation="nearest")
        ax.axvline(pad - 0.5, color="red", lw=0.8, alpha=0.85)
        ax.set_title(title, fontsize=10)
        ax.set_axis_off()
    fig.suptitle("32VNM y04_x17 | x18  ·  delivery ramp on base2 (no retrain)", fontsize=12)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return {"png": str(out), "edge_mae_before": mae0, "edge_mae_after": mae1, "ramp_px": ramp_px}


def _ramp_one_cell(
    arr: np.ndarray,
    *,
    west_strip: np.ndarray | None,
    east_strip: np.ndarray | None,
    north_strip: np.ndarray | None,
    south_strip: np.ndarray | None,
    ramp_px: int,
) -> np.ndarray:
    out = arr.copy()
    if east_strip is not None:
        dl = out[:, -EDGE:, :].mean(axis=1)
        dr = east_strip.mean(axis=1)
        delta = 0.5 * (dr - dl).astype(np.float32)
        xs = np.arange(ramp_px, dtype=np.float32)
        w = _smooth_ramp(xs[::-1], ramp_px)[None, :, None]
        out[:, -ramp_px:, :] = np.clip(out[:, -ramp_px:, :] + delta[:, None, :] * w, 0.0, 1.0)
    if west_strip is not None:
        dr = out[:, :EDGE, :].mean(axis=1)
        dl = west_strip.mean(axis=1)
        delta = 0.5 * (dr - dl).astype(np.float32)
        xs = np.arange(ramp_px, dtype=np.float32)
        w = _smooth_ramp(xs, ramp_px)[None, :, None]
        out[:, :ramp_px, :] = np.clip(out[:, :ramp_px, :] - delta[:, None, :] * w, 0.0, 1.0)
    if south_strip is not None:
        dt = out[-EDGE:, :, :].mean(axis=0)
        db = south_strip.mean(axis=0)
        delta = 0.5 * (db - dt).astype(np.float32)
        ys = np.arange(ramp_px, dtype=np.float32)
        w = _smooth_ramp(ys[::-1], ramp_px)[:, None, None]
        out[-ramp_px:, :, :] = np.clip(out[-ramp_px:, :, :] + delta[None, :, :] * w, 0.0, 1.0)
    if north_strip is not None:
        db = out[:EDGE, :, :].mean(axis=0)
        dt = north_strip.mean(axis=0)
        delta = 0.5 * (db - dt).astype(np.float32)
        ys = np.arange(ramp_px, dtype=np.float32)
        w = _smooth_ramp(ys, ramp_px)[:, None, None]
        out[:ramp_px, :, :] = np.clip(out[:ramp_px, :, :] - delta[None, :, :] * w, 0.0, 1.0)
    return out


def write_mosaic_strips(
    manifest: dict,
    grid: dict[tuple[int, int], Path],
    *,
    ramp_px: int,
    out: Path,
    dates: dict[tuple[int, int], str | None] | None = None,
    independent_dates: dict[tuple[int, int], str | None] | None = None,
    harmonize: bool = False,
    harmonize_strip_px: int = HARMONIZE_STRIP,
    harmonize_segments: int = HARMONIZE_SEGMENTS,
    harmonize_regularization: float = 0.02,
    harmonize_max_offset: float = 0.08,
    harmonize_smoothness: float = 0.1,
    max_harmonized_edge_p95: float | None = None,
) -> dict:
    import rasterio
    from affine import Affine
    from rasterio.windows import Window

    started = time.perf_counter()
    n_y = 1 + max(iy for iy, _ in grid)
    n_x = 1 + max(ix for _, ix in grid)
    origin_xy, origin_path = min(grid.items())
    with rasterio.open(origin_path) as src0:
        origin_transform = src0.transform * Affine.translation(
            -origin_xy[1] * CELL_HR,
            -origin_xy[0] * CELL_HR,
        )
        profile = {
            "driver": "GTiff",
            "height": n_y * CELL_HR,
            "width": n_x * CELL_HR,
            "count": 3,
            "dtype": "float32",
            "crs": src0.crs,
            "transform": origin_transform,
            "nodata": 0.0,
            "compress": "deflate",
            "tiled": True,
            "blockxsize": 512,
            "blockysize": 512,
            "BIGTIFF": "YES",
        }
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing {out}  {n_y}×{n_x} cells  ramp={ramp_px}px", flush=True)
    harmony = (
        estimate_harmonization(
            grid,
            dates=dates,
            independent_dates=independent_dates,
            strip_px=harmonize_strip_px,
            segments=harmonize_segments,
            regularization=harmonize_regularization,
            max_offset=harmonize_max_offset,
            smoothness=harmonize_smoothness,
        )
        if harmonize
        else None
    )
    corrections = {} if harmony is None else harmony["corrections"]
    if harmony is not None:
        metrics = harmony["metrics"]
        before_metric = (
            metrics.get("identity_risk_edges_before")
            or metrics["all_edges_before"]
        )
        after_metric = (
            metrics.get("identity_risk_edges_after_global")
            or metrics["all_edges_after_global"]
        )
        print(
            "  targeted edge p95 "
            f"{before_metric['p95']:.4f}"
            f"->{after_metric['p95']:.4f}; "
            f"max offset={metrics['max_abs_offset']:.4f}",
            flush=True,
        )
    n_written = 0
    ramped_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    with rasterio.open(out, "w", **profile) as dst:
        for (iy, ix), path in sorted(grid.items()):
            if iy >= n_y or ix >= n_x:
                continue
            arr = _apply_offset(_read_rgb(path), corrections.get((iy, ix)))
            west = grid.get((iy, ix - 1))
            east = grid.get((iy, ix + 1))
            north = grid.get((iy - 1, ix))
            south = grid.get((iy + 1, ix))
            xy = (iy, ix)
            west_xy, east_xy = (iy, ix - 1), (iy, ix + 1)
            north_xy, south_xy = (iy - 1, ix), (iy + 1, ix)
            for neighbour in (west_xy, east_xy, north_xy, south_xy):
                if neighbour in grid and _is_balance_cut(
                    dates, independent_dates, xy, neighbour
                ):
                    ramped_edges.add(tuple(sorted((xy, neighbour))))
            arr = _ramp_one_cell(
                arr,
                west_strip=(
                    _apply_offset(
                        _read_strip(west, side="right"),
                        corrections.get(west_xy),
                    )
                    if west and _is_balance_cut(
                        dates, independent_dates, xy, west_xy
                    )
                    else None
                ),
                east_strip=(
                    _apply_offset(
                        _read_strip(east, side="left"),
                        corrections.get(east_xy),
                    )
                    if east and _is_balance_cut(
                        dates, independent_dates, xy, east_xy
                    )
                    else None
                ),
                north_strip=(
                    _apply_offset(
                        _read_strip(north, side="bottom"),
                        corrections.get(north_xy),
                    )
                    if north and _is_balance_cut(
                        dates, independent_dates, xy, north_xy
                    )
                    else None
                ),
                south_strip=(
                    _apply_offset(
                        _read_strip(south, side="top"),
                        corrections.get(south_xy),
                    )
                    if south and _is_balance_cut(
                        dates, independent_dates, xy, south_xy
                    )
                    else None
                ),
                ramp_px=ramp_px,
            )
            dst.write(
                np.transpose(arr, (2, 0, 1)),
                window=Window(ix * CELL_HR, iy * CELL_HR, CELL_HR, CELL_HR),
            )
            n_written += 1
            if n_written % 40 == 0:
                print(f"  {n_written}/{len(grid)}", flush=True)
        for i, name in enumerate(("R", "G", "B"), start=1):
            dst.set_band_description(i, name)
    harmonized_p95 = (
        None
        if harmony is None
        else harmony["metrics"]["identity_risk_edges_after_global"]["p95"]
    )
    qa_pass = (
        (max_harmonized_edge_p95 is None or (
            harmonized_p95 is not None
            and harmonized_p95 <= max_harmonized_edge_p95
        ))
        and (
            harmony is None
            or harmony["metrics"]["n_clipped_components"] == 0
        )
    )
    meta = {
        "path": str(out),
        "n_tiles": n_written,
        "ramp_px": ramp_px,
        "date_cuts_only": dates is not None and independent_dates is None,
        "ramp_scope": (
            "identity_risk_edges_after_global_harmonization"
            if harmonize
            else "date_cuts_only" if dates is not None else "all_edges"
        ),
        "n_ramped_edges": len(ramped_edges),
        "harmonization": (
            None
            if harmony is None
            else {
                "configuration": harmony["configuration"],
                "metrics": harmony["metrics"],
                "corrections": {
                    _tile_id(manifest["parent"], iy, ix): [float(v) for v in value]
                    for (iy, ix), value in sorted(corrections.items())
                },
                "edges": harmony["records"],
            }
        ),
        "qa": {
            "metric": "identity_risk_edges_after_global.p95",
            "measured": harmonized_p95,
            "max_harmonized_edge_p95": max_harmonized_edge_p95,
            "passed": bool(qa_pass),
        },
        "bytes": out.stat().st_size,
        "write_wall_seconds": time.perf_counter() - started,
        "note": (
            "0-overlap global additive radiometric graph balance plus residual "
            "edge ramp" if harmonize else
            "0-overlap delivery ramp; same-date edge cores are not targeted "
            "(corners may receive a perpendicular cut ramp); interiors unchanged"
        ),
    }
    out.with_suffix(out.suffix + ".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Wrote {out} ({out.stat().st_size / 1e9:.2f} GB)", flush=True)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--run-prefix", default="prod_k4_base2")
    ap.add_argument(
        "--fallback-prefix",
        default="prod_k4_base2",
        help="Prefix for cells unchanged by --identity-plan.",
    )
    ap.add_argument(
        "--identity-plan",
        type=Path,
        default=None,
        help="Use changed outputs, fallback unchanged outputs, and identity-aware balancing.",
    )
    ap.add_argument("--ramp-px", type=int, default=128)
    ap.add_argument(
        "--harmonize",
        action="store_true",
        help="Apply bounded whole-cell RGB graph balancing before ramping all edges.",
    )
    ap.add_argument("--harmonize-strip-px", type=int, default=HARMONIZE_STRIP)
    ap.add_argument("--harmonize-segments", type=int, default=HARMONIZE_SEGMENTS)
    ap.add_argument("--harmonize-regularization", type=float, default=0.02)
    ap.add_argument("--harmonize-max-offset", type=float, default=0.08)
    ap.add_argument("--harmonize-smoothness", type=float, default=0.1)
    ap.add_argument("--max-harmonized-edge-p95", type=float, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--preview", type=Path, default=None)
    ap.add_argument("--no-preview", action="store_true")
    ap.add_argument("--mosaic", action="store_true", help="Write the full GeoTIFF (slow).")
    args = ap.parse_args()

    man_path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    man = json.loads(man_path.read_text())
    dates = None
    independent_dates = None
    if args.identity_plan is not None:
        plan_path = (
            args.identity_plan
            if args.identity_plan.is_absolute()
            else ROOT / args.identity_plan
        )
        plan = json.loads(plan_path.read_text())
        grid, dates = _collect_delivery_grid(
            man,
            plan,
            changed_prefix=args.run_prefix,
            fallback_prefix=args.fallback_prefix,
        )
        independent = plan.get("independent") or {}
        independent_dates = {
            (int(tile["iy"]), int(tile["ix"])): independent.get(tile["tile_id"])
            for tile in man.get("tiles") or []
            if (int(tile["iy"]), int(tile["ix"])) in grid
        }
    else:
        grid = _collect_grid(man, args.run_prefix)
    if not grid:
        raise SystemExit("no sr_pred tiles found")

    if not args.no_preview:
        preview = args.preview
        if preview is None:
            preview = ROOT / "production" / "mosaics" / f"{man['parent']}_base2_seams.png"
        elif not preview.is_absolute():
            preview = ROOT / preview
        stats = write_preview(grid, ramp_px=args.ramp_px, out=preview)
        print(json.dumps(stats, indent=2), flush=True)

    if args.mosaic:
        out = args.out
        if out is None:
            out = ROOT / "production" / "mosaics" / f"{man['parent']}_base2_seamramp_sr_2p5m.tif"
        elif not out.is_absolute():
            out = ROOT / out
        write_mosaic_strips(
            man,
            grid,
            ramp_px=args.ramp_px,
            out=out,
            dates=dates,
            independent_dates=independent_dates,
            harmonize=args.harmonize,
            harmonize_strip_px=args.harmonize_strip_px,
            harmonize_segments=args.harmonize_segments,
            harmonize_regularization=args.harmonize_regularization,
            harmonize_max_offset=args.harmonize_max_offset,
            harmonize_smoothness=args.harmonize_smoothness,
            max_harmonized_edge_p95=args.max_harmonized_edge_p95,
        )


if __name__ == "__main__":
    main()
