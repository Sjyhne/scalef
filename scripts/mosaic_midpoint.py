#!/usr/bin/env python3
"""Mosaic overlapping production tiles with a hard midpoint cut.

Each output pixel comes from the prediction whose window center is nearest.
There is no average and no ramp. On a regular grid that cut is the middle of
each overlap, and a corner belongs to one cell.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from rasterio.shutil import copy as rio_copy
from rasterio.transform import array_bounds, from_bounds
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mosaic_granule_sr import _SourceCache, _sr_path  # noqa: E402


def load_assignment(plan_path: Path) -> tuple[str, dict[tuple[int, int], str]]:
    """Return (primary date, (iy, ix) -> assigned date) from an ICM plan."""
    plan = json.loads(Path(plan_path).read_text())
    primary = str(plan["primary"])
    dates: dict[tuple[int, int], str] = {}
    for cell in plan.get("cells") or []:
        dates[(int(cell["iy"]), int(cell["ix"]))] = str(cell["icm_date"])
    if not dates and plan.get("assignment"):
        # Older plans key the assignment by tile id. Callers that only have
        # iy/ix on the mosaic sources should pass a plan with cells.
        raise ValueError(f"{plan_path} has no per-cell dates")
    return primary, dates


def _valid_mean(data: np.ndarray, *, min_valid: int) -> tuple[float, int] | None:
    """Mean reflectance over pixels that are finite and not all-zero."""
    valid = np.any(data != 0.0, axis=0) & np.all(np.isfinite(data), axis=0)
    n = int(valid.sum())
    if n < min_valid:
        return None
    return float(data[:, valid].mean()), n


def solve_overlap_gains(
    sources: list[tuple[Path, int, int]],
    dates: dict[tuple[int, int], str],
    primary: str,
    *,
    min_valid: int = 1000,
    min_mean: float = 0.008,
    gain_clip: tuple[float, float] = (1.0 / 3.0, 3.0),
) -> dict[tuple[int, int], float]:
    """One scalar gain per tile so overlap means agree.

    Tiles on ``primary`` stay at gain 1. Every other tile is scaled so the
    overlap with its neighbours matches. There is no spatial blend: the
    midpoint cut is unchanged, and the gain is constant across a tile.
    """
    by_yx = {(int(iy), int(ix)): Path(path) for path, iy, ix in sources}
    # log(g_i) - log(g_j) = log(mean_j / mean_i), weighted by valid pixels.
    edges: list[tuple[tuple[int, int], tuple[int, int], float, float]] = []

    def consider(a: tuple[int, int], b: tuple[int, int], mean_a: float, mean_b: float, n: int) -> None:
        if mean_a < min_mean or mean_b < min_mean:
            return
        edges.append((a, b, float(np.log(mean_b / mean_a)), float(n)))

    pairs: list[tuple[tuple[int, int], tuple[int, int], str]] = []
    for iy, ix in by_yx:
        if (iy, ix + 1) in by_yx:
            pairs.append(((iy, ix), (iy, ix + 1), "h"))
        if (iy + 1, ix) in by_yx:
            pairs.append(((iy, ix), (iy + 1, ix), "v"))

    for north_or_west, south_or_east, kind in pairs:
        path_a, path_b = by_yx[north_or_west], by_yx[south_or_east]
        with rasterio.open(path_a) as src_a, rasterio.open(path_b) as src_b:
            res = float(abs(src_a.res[0]))
            if kind == "h":
                overlap_m = float(src_a.bounds.right - src_b.bounds.left)
                px = int(round(overlap_m / res))
                if px < 2 or px >= src_a.width or px >= src_b.width:
                    continue
                data_a = src_a.read(window=Window(src_a.width - px, 0, px, src_a.height))
                data_b = src_b.read(window=Window(0, 0, px, src_b.height))
            else:
                overlap_m = float(src_b.bounds.top - src_a.bounds.bottom)
                px = int(round(overlap_m / res))
                if px < 2 or px >= src_a.height or px >= src_b.height:
                    continue
                data_a = src_a.read(window=Window(0, src_a.height - px, src_a.width, px))
                data_b = src_b.read(window=Window(0, 0, src_b.width, px))
        stat_a = _valid_mean(data_a.astype(np.float32, copy=False), min_valid=min_valid)
        stat_b = _valid_mean(data_b.astype(np.float32, copy=False), min_valid=min_valid)
        if stat_a is None or stat_b is None:
            continue
        consider(north_or_west, south_or_east, stat_a[0], stat_b[0], min(stat_a[1], stat_b[1]))

    anchored = {
        yx for yx in by_yx if dates.get(yx) == primary
    }
    free = [yx for yx in sorted(by_yx) if yx not in anchored]
    gains = {yx: 1.0 for yx in by_yx}
    if not free or not edges:
        return gains
    index = {yx: k for k, yx in enumerate(free)}
    rows: list[np.ndarray] = []
    rhs: list[float] = []
    n_free = len(free)
    for a, b, log_ratio, weight in edges:
        row = np.zeros(n_free, dtype=np.float64)
        value = log_ratio
        # Anchored tiles have log-gain 0, so they add nothing to the right-hand side.
        if a in index:
            row[index[a]] += 1.0
        if b in index:
            row[index[b]] -= 1.0
        # Drop edges that do not involve a free tile.
        if not np.any(row):
            continue
        rows.append(row * weight)
        rhs.append(value * weight)
    # Weak pull toward 1 so a tile with no usable overlap stays put.
    for k in range(n_free):
        row = np.zeros(n_free, dtype=np.float64)
        row[k] = 1.0
        rows.append(row)
        rhs.append(0.0)
    solved, _, _, _ = np.linalg.lstsq(np.vstack(rows), np.asarray(rhs), rcond=None)
    lo, hi = gain_clip
    for yx, k in index.items():
        gains[yx] = float(np.clip(np.exp(solved[k]), lo, hi))
    return gains


def write_midpoint(
    sources: list[tuple[Path, int, int]],
    out_path: Path,
    *,
    cog: bool,
    block_size: int = 512,
    gains: dict[tuple[int, int], float] | None = None,
) -> dict:
    """Stream a nearest-center mosaic. ``sources`` is (path, iy, ix)."""
    details = []
    bounds = None
    crs = None
    res = None
    count = None
    dtype = None
    for path, iy, ix in sources:
        with rasterio.open(path) as src:
            if crs is None:
                crs, res, count, dtype = src.crs, src.res, src.count, src.dtypes[0]
            elif src.crs != crs or not np.allclose(src.res, res):
                raise ValueError(f"{path} CRS or resolution does not match the first tile")
            b = src.bounds
            bounds = (
                [b.left, b.bottom, b.right, b.top]
                if bounds is None
                else [
                    min(bounds[0], b.left),
                    min(bounds[1], b.bottom),
                    max(bounds[2], b.right),
                    max(bounds[3], b.top),
                ]
            )
            details.append(
                {
                    "path": path,
                    "bounds": b,
                    "width": src.width,
                    "height": src.height,
                    "iy": int(iy),
                    "ix": int(ix),
                    "gain": 1.0 if gains is None else float(gains.get((int(iy), int(ix)), 1.0)),
                }
            )
    assert bounds is not None and res is not None and count is not None
    res_x, res_y = float(res[0]), float(abs(res[1]))
    width = int(round((bounds[2] - bounds[0]) / res_x))
    height = int(round((bounds[3] - bounds[1]) / res_y))
    transform = from_bounds(*bounds, width, height)
    # Northwest wins an exact tie, so later tiles only replace a strictly nearer center.
    details.sort(key=lambda d: (d["iy"], d["ix"]))
    row_index: dict[int, list[tuple[int, int, int]]] = {}
    for index, detail in enumerate(details):
        b = detail["bounds"]
        c0 = int(round((b.left - bounds[0]) / res_x))
        r0 = int(round((bounds[3] - b.top) / res_y))
        detail.update(
            {
                "r0": r0,
                "c0": c0,
                "cy": r0 + detail["height"] / 2.0,
                "cx": c0 + detail["width"] / 2.0,
            }
        )
        c1, r1 = c0 + detail["width"], r0 + detail["height"]
        for block_row in range(r0 // block_size, (r1 - 1) // block_size + 1):
            row_index.setdefault(block_row, []).append(
                (c0 // block_size, (c1 - 1) // block_size, index)
            )

    token = uuid.uuid4().hex
    temp_gtiff = out_path.with_name(f".{out_path.name}.{token}.partial.tif")
    temp_cog = out_path.with_name(f".{out_path.name}.{token}.partial.cog.tif")
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": count,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "nodata": 0.0,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": block_size,
        "blockysize": block_size,
        "BIGTIFF": "YES",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cache = _SourceCache([d["path"] for d in details])
    valid_pixels = 0
    windows_written = 0
    try:
        with rasterio.open(temp_gtiff, "w", **profile) as dst:
            n_windows = sum(1 for _ in dst.block_windows(1))
            for (block_row, block_col), window in dst.block_windows(1):
                h, w = int(window.height), int(window.width)
                output = np.zeros((count, h, w), dtype=np.float32)
                best = np.full((h, w), np.inf, dtype=np.float64)
                wr0, wc0 = int(window.row_off), int(window.col_off)
                wr1, wc1 = wr0 + h, wc0 + w
                yy = np.arange(wr0, wr1, dtype=np.float64)[:, None] + 0.5
                xx = np.arange(wc0, wc1, dtype=np.float64)[None, :] + 0.5
                candidates = (
                    index
                    for first_col, last_col, index in row_index.get(block_row, [])
                    if first_col <= block_col <= last_col
                )
                for index in candidates:
                    detail = details[index]
                    sr0, sc0 = int(detail["r0"]), int(detail["c0"])
                    sr1 = sr0 + int(detail["height"])
                    sc1 = sc0 + int(detail["width"])
                    r0, r1 = max(wr0, sr0), min(wr1, sr1)
                    c0, c1 = max(wc0, sc0), min(wc1, sc1)
                    if r1 <= r0 or c1 <= c0:
                        continue
                    local = Window(c0 - sc0, r0 - sr0, c1 - c0, r1 - r0)
                    data = cache.get(index).read(window=local).astype(np.float32, copy=False)
                    valid = np.any(data != 0.0, axis=0) & np.all(np.isfinite(data), axis=0)
                    gain = float(detail["gain"])
                    if gain != 1.0:
                        data = data * np.float32(gain)
                    dr = slice(r0 - wr0, r1 - wr0)
                    dc = slice(c0 - wc0, c1 - wc0)
                    dist = (yy[dr] - detail["cy"]) ** 2 + (xx[:, dc] - detail["cx"]) ** 2
                    closer = valid & (dist < best[dr, dc])
                    best[dr, dc] = np.where(closer, dist, best[dr, dc])
                    output[(slice(None), dr, dc)] = np.where(closer[None], data, output[(slice(None), dr, dc)])
                good = np.isfinite(best)
                dst.write(output.astype(dtype, copy=False), window=window)
                dst.write_mask(good.astype("uint8") * 255, window=window)
                valid_pixels += int(np.count_nonzero(good))
                windows_written += 1
                if windows_written % 400 == 0 or windows_written == n_windows:
                    print(f"  midpoint {windows_written}/{n_windows} blocks", flush=True)
            for index in range(1, count + 1):
                dst.set_band_description(
                    index, ("R", "G", "B", "A")[index - 1] if index <= 4 else f"B{index}"
                )
        if cog:
            with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
                rio_copy(
                    temp_gtiff,
                    temp_cog,
                    driver="COG",
                    COMPRESS="DEFLATE",
                    BLOCKSIZE=block_size,
                    BIGTIFF="YES",
                    OVERVIEWS="AUTO",
                )
            os.replace(temp_cog, out_path)
        else:
            os.replace(temp_gtiff, out_path)
    finally:
        cache.close()
        temp_gtiff.unlink(missing_ok=True)
        temp_cog.unlink(missing_ok=True)
    return {
        "path": str(out_path),
        "width": width,
        "height": height,
        "count": count,
        "gsd_m": float(abs(transform.a)),
        "extent_m": [width * res_x, height * res_y],
        "bounds": list(array_bounds(height, width, transform)),
        "crs": str(crs),
        "n_sources": len(sources),
        "bytes": out_path.stat().st_size,
        "cog": bool(cog),
        "method": "midpoint",
        "gain_match": gains is not None,
        "gain_min": None if gains is None else min(gains.values()),
        "gain_max": None if gains is None else max(gains.values()),
        "streaming": True,
        "windows_written": windows_written,
        "valid_pixels": valid_pixels,
        "nodata_fraction": 1.0 - valid_pixels / max(width * height, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--run-prefix", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-cog", action="store_true")
    ap.add_argument(
        "--allow-missing",
        action="store_true",
        help="Mosaic the tiles that finished. Fail when none exist.",
    )
    ap.add_argument(
        "--gain-match",
        action="store_true",
        help=(
            "Scale each off-date tile by one gain so its overlap matches "
            "the primary-date neighbours. The midpoint cut is unchanged."
        ),
    )
    ap.add_argument(
        "--identity-plan",
        type=Path,
        default=None,
        help="ICM plan with primary and per-cell icm_date. Required with --gain-match.",
    )
    args = ap.parse_args()
    man_path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    man = json.loads(man_path.read_text())
    parent = man["parent"]
    out = args.out if args.out.is_absolute() else ROOT / args.out
    sources = []
    missing = []
    for tile in man["tiles"]:
        path = _sr_path(parent, tile["tile_id"], args.run_prefix)
        if path.is_file():
            sources.append((path, int(tile["iy"]), int(tile["ix"])))
        else:
            missing.append(tile["tile_id"])
    if missing and not args.allow_missing:
        raise SystemExit(f"{len(missing)} tiles missing SR GeoTIFFs, e.g. {missing[0]}")
    if not sources:
        raise SystemExit("no source tiles found")
    gains = None
    if args.gain_match:
        if args.identity_plan is None:
            raise SystemExit("--gain-match requires --identity-plan")
        plan_path = args.identity_plan if args.identity_plan.is_absolute() else ROOT / args.identity_plan
        primary, dates = load_assignment(plan_path)
        print(f"Gain match against primary {primary}", flush=True)
        gains = solve_overlap_gains(sources, dates, primary)
        moved = [g for g in gains.values() if abs(g - 1.0) > 1e-3]
        print(
            f"  gains {len(gains)} tiles, {len(moved)} moved, "
            f"range {min(gains.values()):.3f}–{max(gains.values()):.3f}",
            flush=True,
        )
    print(f"Midpoint mosaic {len(sources)} tiles ({len(missing)} missing)", flush=True)
    meta = write_midpoint(sources, out, cog=not args.no_cog, gains=gains)
    side = out.with_suffix(out.suffix + ".json")
    side.write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "manifest": str(man_path),
                "run_prefix": args.run_prefix,
                "method": "midpoint",
                "rule": "nearest window center, hard cut, no blend",
                "gain_match": gains is not None,
                "primary_date": None if gains is None else primary,
                "gains": None
                if gains is None
                else {f"{iy}_{ix}": round(g, 4) for (iy, ix), g in sorted(gains.items()) if abs(g - 1) > 1e-3},
                "overlap_frac": man.get("overlap_frac"),
                "overlap_px": man.get("overlap_px"),
                "n_missing": len(missing),
                "missing_sample": missing[:8],
                "mosaic": meta,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"Wrote {out} ({meta['width']}×{meta['height']} @ {meta['gsd_m']:g} m, "
        f"{meta['bytes'] / 1e9:.2f} GB)",
        flush=True,
    )


if __name__ == "__main__":
    main()
