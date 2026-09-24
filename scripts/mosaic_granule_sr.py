#!/usr/bin/env python3
"""Mosaic per-AOI production ``sr_pred.tif`` files into one full-MGRS GeoTIFF/COG.

Reads a granule tile manifest (from ``make_granule_tiles.py``) and the matching
``run_production.py`` outputs under ``single_samples/<parent>/sample/prod_k4_*/qgis/``.

Non-overlapping tiles use ``method=first``. Overlapping tiles default to
``feather``: distance-to-edge weights so seams ramp smoothly instead of a hard
average / painter's cut.

Example
-------
    python scripts/mosaic_granule_sr.py \\
      --manifest data/s2_revisits/asker/granule_tiles_lr512_manifest.json \\
      --out production/mosaics/asker_32VNM_sr_2p5m.tif

    python scripts/mosaic_granule_sr.py \\
      --manifest data/s2_revisits/asker/granule_tiles_lr512_ovl10_manifest.json \\
      --method feather --out production/mosaics/asker_ovl10_feather_sr_2p5m.tif
"""

from __future__ import annotations

import argparse
import json
import os
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.shutil import copy as rio_copy
from rasterio.transform import array_bounds, from_bounds, rowcol
from rasterio.windows import Window
from rasterio.windows import from_bounds as window_from_bounds

from scripts.mosaic_seam_ramp import estimate_harmonization

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUN_PREFIX = "prod_k4"


def _apply_radiometric_correction(
    data: np.ndarray,
    valid: np.ndarray,
    correction: np.ndarray | dict | None,
) -> None:
    """Apply an additive or affine RGB correction in place."""
    if correction is None or not np.any(valid):
        return
    if isinstance(correction, dict):
        gain = np.asarray(correction["gain"], dtype=np.float32)
        offset = np.asarray(correction["offset"], dtype=np.float32)
        data[:, valid] = np.clip(
            gain[:, None] * data[:, valid] + offset[:, None], 0.0, 1.0
        )
    else:
        offset = np.asarray(correction, dtype=np.float32)
        data[:, valid] = np.clip(data[:, valid] + offset[:, None], 0.0, 1.0)


def _run_name(tile_id: str, prefix: str = DEFAULT_RUN_PREFIX) -> str:
    return f"{prefix}_{tile_id}"


def _sr_path(parent: str, tile_id: str, prefix: str = DEFAULT_RUN_PREFIX) -> Path:
    return (
        ROOT
        / "single_samples"
        / parent
        / "sample"
        / _run_name(tile_id, prefix)
        / "qgis"
        / "sr_pred.tif"
    )


def _bil_path(parent: str, tile_id: str, prefix: str = DEFAULT_RUN_PREFIX) -> Path:
    return _sr_path(parent, tile_id, prefix).with_name("s2_bilinear.tif")


def collect_sources(
    manifest: dict, *, layer: str, run_prefix: str = DEFAULT_RUN_PREFIX
) -> tuple[list[Path], list[dict]]:
    parent = manifest["parent"]
    missing: list[dict] = []
    paths: list[Path] = []
    for tile in manifest["tiles"]:
        tid = tile["tile_id"]
        path = (
            _sr_path(parent, tid, run_prefix)
            if layer == "sr"
            else _bil_path(parent, tid, run_prefix)
        )
        if not path.is_file():
            missing.append({"tile_id": tid, "expected": str(path.relative_to(ROOT))})
            continue
        paths.append(path)
    return paths, missing


def _edge_distance_weights(height: int, width: int, *, feather_px: float) -> np.ndarray:
    """Per-pixel weight in [0, 1]: 0 at tile border, 1 beyond ``feather_px`` inward."""
    ys = np.arange(height, dtype=np.float32)
    xs = np.arange(width, dtype=np.float32)
    dist = np.minimum(
        np.minimum(ys, (height - 1) - ys)[:, None],
        np.minimum(xs, (width - 1) - xs)[None, :],
    )
    if feather_px <= 0:
        return np.ones((height, width), dtype=np.float32)
    return np.clip(dist / float(feather_px), 0.0, 1.0).astype(np.float32)


def _merge_mean(datasets, *, nodata: float):
    """Uniform average of overlapping pixels."""
    sum_arr, transform = merge(datasets, nodata=nodata, method="sum")
    count_arr, _ = merge(datasets, nodata=nodata, method="count")
    count = np.maximum(count_arr.astype(np.float32), 0.0)
    out = np.full_like(sum_arr, nodata, dtype=np.float32)
    valid = count > 0
    np.divide(sum_arr, count, out=out, where=valid)
    return out.astype(sum_arr.dtype, copy=False), transform


def _merge_feather(
    sources: list[Path],
    *,
    nodata: float,
    feather_px: float,
) -> tuple[np.ndarray, object, object]:
    """Distance-to-edge weighted blend; streams one tile at a time."""
    # Output extent from union of sources (same CRS / resolution assumed).
    bounds = None
    res = None
    crs = None
    count = None
    dtype = None
    for p in sources:
        with rasterio.open(p) as src:
            b = src.bounds
            if bounds is None:
                bounds = [b.left, b.bottom, b.right, b.top]
                res = src.res
                crs = src.crs
                count = src.count
                dtype = src.dtypes[0]
            else:
                bounds[0] = min(bounds[0], b.left)
                bounds[1] = min(bounds[1], b.bottom)
                bounds[2] = max(bounds[2], b.right)
                bounds[3] = max(bounds[3], b.top)

    assert bounds is not None and res is not None and count is not None
    res_x, res_y = float(res[0]), float(abs(res[1]))
    width = int(round((bounds[2] - bounds[0]) / res_x))
    height = int(round((bounds[3] - bounds[1]) / res_y))
    transform = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], width, height)

    acc = np.zeros((count, height, width), dtype=np.float64)
    wacc = np.zeros((height, width), dtype=np.float64)

    for i, p in enumerate(sources):
        with rasterio.open(p) as src:
            data = src.read().astype(np.float32, copy=False)
            h, w = int(data.shape[1]), int(data.shape[2])
            weight = _edge_distance_weights(h, w, feather_px=feather_px)
            # Mask nodata (all-zero treated as empty, matching prior mosaics).
            valid = np.any(data != nodata, axis=0)
            weight = np.where(valid, weight, 0.0)

            # Map tile into destination grid.
            r0, c0 = rowcol(transform, src.bounds.left, src.bounds.top)
            r0, c0 = int(r0), int(c0)
            r1, c1 = r0 + h, c0 + w
            # Clip if floating-point alignment drifts by a pixel.
            if r0 < 0 or c0 < 0 or r1 > height or c1 > width:
                rs0, cs0 = max(0, -r0), max(0, -c0)
                rs1 = h - max(0, r1 - height)
                cs1 = w - max(0, c1 - width)
                data = data[:, rs0:rs1, cs0:cs1]
                weight = weight[rs0:rs1, cs0:cs1]
                r0, c0 = max(0, r0), max(0, c0)
                r1, c1 = r0 + data.shape[1], c0 + data.shape[2]

            acc[:, r0:r1, c0:c1] += data.astype(np.float64) * weight[None, :, :]
            wacc[r0:r1, c0:c1] += weight

        if (i + 1) % 50 == 0 or i + 1 == len(sources):
            print(f"  feather {i + 1}/{len(sources)}", flush=True)

    out = np.full((count, height, width), nodata, dtype=np.float32)
    m = wacc > 0
    for b in range(count):
        out[b][m] = (acc[b][m] / wacc[m]).astype(np.float32)
    return out, transform, crs


class _SourceCache:
    def __init__(self, sources: list[Path], max_open: int = 64) -> None:
        self.sources = sources
        self.max_open = max(1, int(max_open))
        self.opened: OrderedDict[int, object] = OrderedDict()

    def get(self, index: int):
        if index in self.opened:
            src = self.opened.pop(index)
            self.opened[index] = src
            return src
        src = rasterio.open(self.sources[index])
        self.opened[index] = src
        while len(self.opened) > self.max_open:
            _, old = self.opened.popitem(last=False)
            old.close()
        return src

    def close(self) -> None:
        for src in self.opened.values():
            src.close()
        self.opened.clear()


def _stream_feather(
    sources: list[Path],
    out_path: Path,
    *,
    cog: bool,
    feather_px: float,
    corrections: dict[Path, np.ndarray | dict] | None = None,
    block_size: int = 512,
) -> dict:
    """Blend aligned overlap tiles by output block, never allocating a granule array."""
    details = []
    bounds = None
    crs = None
    res = None
    count = None
    dtype = None
    for path in sources:
        with rasterio.open(path) as src:
            if crs is None:
                crs, res, count, dtype = src.crs, src.res, src.count, src.dtypes[0]
            elif src.crs != crs or not np.allclose(src.res, res):
                raise ValueError("feather sources must share CRS and resolution")
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
                }
            )
    assert bounds is not None and res is not None and count is not None
    res_x, res_y = float(res[0]), float(abs(res[1]))
    width = int(round((bounds[2] - bounds[0]) / res_x))
    height = int(round((bounds[3] - bounds[1]) / res_y))
    transform = from_bounds(*bounds, width, height)
    row_index: dict[int, list[tuple[int, int, int]]] = {}
    for index, detail in enumerate(details):
        b = detail["bounds"]
        c0 = int(round((b.left - bounds[0]) / res_x))
        r0 = int(round((bounds[3] - b.top) / res_y))
        detail.update({"r0": r0, "c0": c0})
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
    cache = _SourceCache(sources)
    valid_pixels = 0
    overlap_pixels = 0
    windows_written = 0
    try:
        with rasterio.open(temp_gtiff, "w", **profile) as dst:
            for (block_row, block_col), window in dst.block_windows(1):
                h, w = int(window.height), int(window.width)
                acc = np.zeros((count, h, w), dtype=np.float32)
                wacc = np.zeros((h, w), dtype=np.float32)
                coverage = np.zeros((h, w), dtype=np.uint16)
                candidates = (
                    index
                    for first_col, last_col, index in row_index.get(block_row, [])
                    if first_col <= block_col <= last_col
                )
                wr0, wc0 = int(window.row_off), int(window.col_off)
                wr1, wc1 = wr0 + h, wc0 + w
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
                    data = cache.get(index).read(window=local).astype(np.float32)
                    valid = np.any(data != 0.0, axis=0) & np.all(np.isfinite(data), axis=0)
                    correction = (corrections or {}).get(detail["path"])
                    _apply_radiometric_correction(data, valid, correction)
                    ys = np.arange(int(local.row_off), int(local.row_off + local.height))
                    xs = np.arange(int(local.col_off), int(local.col_off + local.width))
                    edge = np.minimum(
                        np.minimum(ys, detail["height"] - 1 - ys)[:, None],
                        np.minimum(xs, detail["width"] - 1 - xs)[None, :],
                    )
                    weight = np.clip(edge / float(feather_px), 0.0, 1.0).astype(
                        np.float32
                    )
                    weight = np.where(valid, np.maximum(weight, 1e-6), 0.0)
                    dr = slice(r0 - wr0, r1 - wr0)
                    dc = slice(c0 - wc0, c1 - wc0)
                    acc[:, dr, dc] += data * weight[None]
                    wacc[dr, dc] += weight
                    coverage[dr, dc] += valid
                output = np.zeros((count, h, w), dtype=np.float32)
                good = wacc > 0
                np.divide(acc, wacc[None], out=output, where=good[None])
                dst.write(output.astype(dtype, copy=False), window=window)
                dst.write_mask(good.astype("uint8") * 255, window=window)
                valid_pixels += int(np.count_nonzero(good))
                overlap_pixels += int(np.count_nonzero(coverage >= 2))
                windows_written += 1
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
    output_pixels = width * height
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
        "method": "feather",
        "feather_px": feather_px,
        "streaming": True,
        "windows_written": windows_written,
        "valid_pixels": valid_pixels,
        "overlap_pixels": overlap_pixels,
        "nodata_fraction": 1.0 - valid_pixels / max(output_pixels, 1),
    }


def _values_summary(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": len(values),
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def estimate_date_affine_harmonization(
    grid: dict[tuple[int, int], Path],
    *,
    dates: dict[tuple[int, int], str | None],
    initial_corrections: dict[tuple[int, int], np.ndarray] | None = None,
    margin_px: int = 8,
    samples_per_pair: int = 512,
    gain_regularization: float = 0.2,
    offset_regularization: float = 1.0,
    max_gain_delta: float = 0.15,
    max_offset: float = 0.03,
) -> dict:
    """Robustly solve one near-identity RGB affine transform per base-frame date.

    Corresponding pixels in cross-date tile overlaps constrain a global date
    graph. Pair normalization prevents long overlaps from dominating, while
    IRLS suppresses changed surfaces and prediction outliers.
    """
    from scipy import sparse
    from scipy.sparse.linalg import lsqr

    active_dates = sorted({date for date in dates.values() if date is not None})
    if len(active_dates) < 2:
        raise ValueError("affine date harmonization requires at least two dates")
    date_index = {date: index for index, date in enumerate(active_dates)}
    tile_counts = {
        date: sum(value == date for value in dates.values()) for date in active_dates
    }
    reference_date = max(active_dates, key=lambda date: (tile_counts[date], date))

    pair_samples = []
    pair_records = []
    for xy, first_path in sorted(grid.items()):
        for neighbour in ((xy[0], xy[1] + 1), (xy[0] + 1, xy[1])):
            second_path = grid.get(neighbour)
            first_date, second_date = dates.get(xy), dates.get(neighbour)
            if (
                second_path is None
                or first_date is None
                or second_date is None
                or first_date == second_date
            ):
                continue
            with rasterio.open(first_path) as first, rasterio.open(second_path) as second:
                left = max(first.bounds.left, second.bounds.left)
                bottom = max(first.bounds.bottom, second.bounds.bottom)
                right = min(first.bounds.right, second.bounds.right)
                top = min(first.bounds.top, second.bounds.top)
                if right <= left or top <= bottom:
                    continue
                first_window = window_from_bounds(
                    left, bottom, right, top, transform=first.transform
                ).round_offsets().round_lengths()
                second_window = window_from_bounds(
                    left, bottom, right, top, transform=second.transform
                ).round_offsets().round_lengths()
                a = first.read(indexes=(1, 2, 3), window=first_window).astype(np.float32)
                b = second.read(indexes=(1, 2, 3), window=second_window).astype(np.float32)
            h, w = min(a.shape[1], b.shape[1]), min(a.shape[2], b.shape[2])
            a, b = a[:, :h, :w], b[:, :h, :w]
            valid = (
                np.all(np.isfinite(a), axis=0)
                & np.all(np.isfinite(b), axis=0)
                & np.any(a != 0.0, axis=0)
                & np.any(b != 0.0, axis=0)
            )
            if initial_corrections:
                _apply_radiometric_correction(a, valid, initial_corrections.get(xy))
                _apply_radiometric_correction(
                    b, valid, initial_corrections.get(neighbour)
                )
            if margin_px > 0 and h > 2 * margin_px and w > 2 * margin_px:
                interior = np.zeros((h, w), dtype=bool)
                interior[margin_px:-margin_px, margin_px:-margin_px] = True
                valid &= interior
            indices = np.flatnonzero(valid)
            if indices.size < 64:
                continue
            # Drop pair-specific extremes (cloud, shadow, saturated prediction)
            # for fitting only; this does not alter output reflectance.
            lum = 0.5 * (
                a.reshape(3, -1)[:, indices].mean(axis=0)
                + b.reshape(3, -1)[:, indices].mean(axis=0)
            )
            lo, hi = np.percentile(lum, [2, 98])
            indices = indices[(lum >= lo) & (lum <= hi)]
            if indices.size < 64:
                continue
            if indices.size > samples_per_pair:
                # Deterministic spatially uniform sample.
                take = np.linspace(
                    0, indices.size - 1, samples_per_pair, dtype=np.int64
                )
                indices = indices[take]
            av = a.reshape(3, -1)[:, indices].T
            bv = b.reshape(3, -1)[:, indices].T
            pair_samples.append((first_date, second_date, av, bv))
            pair_records.append(
                {
                    "first": list(xy),
                    "second": list(neighbour),
                    "first_date": first_date,
                    "second_date": second_date,
                    "n_samples": int(indices.size),
                    "mae_before": float(np.mean(np.abs(av - bv))),
                }
            )
    if not pair_samples:
        raise ValueError("no valid cross-date overlap pixels for affine harmonization")

    n_dates = len(active_dates)
    transforms = np.zeros((n_dates, 3, 2), dtype=np.float64)
    solver_iterations = []
    for channel in range(3):
        row_parts, col_parts, value_parts, responses, base_weights = [], [], [], [], []
        row0 = 0
        for first_date, second_date, a, b in pair_samples:
            n = len(a)
            rows = np.repeat(np.arange(row0, row0 + n), 4)
            i, j = date_index[first_date], date_index[second_date]
            columns = np.tile([2 * i, 2 * i + 1, 2 * j, 2 * j + 1], n)
            values = np.column_stack(
                [a[:, channel], np.ones(n), -b[:, channel], -np.ones(n)]
            ).ravel()
            row_parts.append(rows)
            col_parts.append(columns)
            value_parts.append(values)
            responses.append(b[:, channel] - a[:, channel])
            base_weights.append(np.full(n, 1.0 / np.sqrt(n), dtype=np.float64))
            row0 += n
        matrix = sparse.csr_matrix(
            (
                np.concatenate(value_parts),
                (np.concatenate(row_parts), np.concatenate(col_parts)),
            ),
            shape=(row0, 2 * n_dates),
        )
        response = np.concatenate(responses)
        base_weight = np.concatenate(base_weights)
        robust_weight = np.ones(row0, dtype=np.float64)
        solution = np.zeros(2 * n_dates, dtype=np.float64)
        for iteration in range(5):
            weights = base_weight * np.sqrt(robust_weight)
            design = sparse.diags(weights) @ matrix
            target = response * weights
            regularizer = sparse.diags(
                np.tile(
                    [
                        np.sqrt(float(gain_regularization)),
                        np.sqrt(float(offset_regularization)),
                    ],
                    n_dates,
                )
            )
            ref = date_index[reference_date]
            pin = sparse.csr_matrix(
                ([10.0, 10.0], ([0, 1], [2 * ref, 2 * ref + 1])),
                shape=(2, 2 * n_dates),
            )
            design = sparse.vstack([design, regularizer, pin], format="csr")
            target = np.concatenate([target, np.zeros(2 * n_dates + 2)])
            solution = lsqr(design, target, atol=1e-10, btol=1e-10)[0]
            residual = matrix @ solution - response
            scale = max(1.4826 * float(np.median(np.abs(residual))), 1e-5)
            normalized = np.abs(residual) / (1.5 * scale)
            robust_weight = np.ones_like(normalized)
            tail = normalized > 1.0
            robust_weight[tail] = 1.0 / normalized[tail]
        transforms[:, channel, 0] = np.clip(
            solution[0::2], -float(max_gain_delta), float(max_gain_delta)
        )
        transforms[:, channel, 1] = np.clip(
            solution[1::2], -float(max_offset), float(max_offset)
        )
        solver_iterations.append(iteration + 1)

    by_date = {
        date: {
            "gain": (1.0 + transforms[index, :, 0]).astype(np.float32),
            "offset": transforms[index, :, 1].astype(np.float32),
        }
        for date, index in date_index.items()
    }
    corrections = {}
    for xy, date in dates.items():
        if date not in by_date:
            continue
        correction = by_date[date]
        initial = (initial_corrections or {}).get(xy)
        corrections[xy] = {
            "gain": correction["gain"],
            "offset": (
                correction["offset"]
                if initial is None
                else correction["gain"] * np.asarray(initial) + correction["offset"]
            ).astype(np.float32),
        }
    after_values = []
    for record, (first_date, second_date, a, b) in zip(pair_records, pair_samples):
        first_correction, second_correction = by_date[first_date], by_date[second_date]
        ac = a * first_correction["gain"] + first_correction["offset"]
        bc = b * second_correction["gain"] + second_correction["offset"]
        after = float(np.mean(np.abs(ac - bc)))
        record["mae_after"] = after
        after_values.append(after)
    return {
        "corrections": corrections,
        "records": pair_records,
        "metrics": {
            "cross_date_overlap_before": _values_summary(
                [record["mae_before"] for record in pair_records]
            ),
            "cross_date_overlap_after": _values_summary(after_values),
            "n_dates": n_dates,
            "n_cross_date_pairs": len(pair_records),
            "n_fit_samples": int(sum(record["n_samples"] for record in pair_records)),
            "max_abs_gain_delta": float(np.max(np.abs(transforms[:, :, 0]))),
            "max_abs_offset": float(np.max(np.abs(transforms[:, :, 1]))),
        },
        "configuration": {
            "method": "robust_cross_date_overlap_affine_rgb_v1",
            "scope": "one_global_transform_per_assigned_base_frame_date",
            "initial_correction": (
                "none" if initial_corrections is None else "additive_edge_graph"
            ),
            "reference_date": reference_date,
            "margin_px": int(margin_px),
            "samples_per_pair": int(samples_per_pair),
            "gain_regularization": float(gain_regularization),
            "offset_regularization": float(offset_regularization),
            "max_gain_delta": float(max_gain_delta),
            "max_offset": float(max_offset),
            "irls_iterations": max(solver_iterations),
            "date_transforms": {
                date: {
                    "gain": [float(value) for value in correction["gain"]],
                    "offset": [float(value) for value in correction["offset"]],
                }
                for date, correction in by_date.items()
            },
        },
    }


def measure_overlap_qa(
    grid: dict[tuple[int, int], Path],
    *,
    corrections: dict[tuple[int, int], np.ndarray | dict] | None = None,
    dates: dict[tuple[int, int], str | None] | None = None,
    independent_dates: dict[tuple[int, int], str | None] | None = None,
    margin_px: int = 8,
) -> dict:
    """Measure corresponding-pixel MAE in every east/south tile overlap."""
    records = []
    groups = {"same_identity": [], "identity_risk": [], "all": []}
    for xy, first_path in sorted(grid.items()):
        for neighbour in ((xy[0], xy[1] + 1), (xy[0] + 1, xy[1])):
            second_path = grid.get(neighbour)
            if second_path is None:
                continue
            with rasterio.open(first_path) as first, rasterio.open(second_path) as second:
                left = max(first.bounds.left, second.bounds.left)
                bottom = max(first.bounds.bottom, second.bounds.bottom)
                right = min(first.bounds.right, second.bounds.right)
                top = min(first.bounds.top, second.bounds.top)
                if right <= left or top <= bottom:
                    continue
                first_window = window_from_bounds(
                    left, bottom, right, top, transform=first.transform
                ).round_offsets().round_lengths()
                second_window = window_from_bounds(
                    left, bottom, right, top, transform=second.transform
                ).round_offsets().round_lengths()
                a = first.read(indexes=(1, 2, 3), window=first_window).astype(np.float32)
                b = second.read(indexes=(1, 2, 3), window=second_window).astype(np.float32)
            h, w = min(a.shape[1], b.shape[1]), min(a.shape[2], b.shape[2])
            a, b = a[:, :h, :w], b[:, :h, :w]
            valid = np.any(a != 0.0, axis=0) & np.any(b != 0.0, axis=0)
            if margin_px > 0 and h > 2 * margin_px and w > 2 * margin_px:
                interior = np.zeros((h, w), dtype=bool)
                interior[margin_px:-margin_px, margin_px:-margin_px] = True
                valid &= interior
            if corrections:
                _apply_radiometric_correction(a, valid, corrections.get(xy))
                _apply_radiometric_correction(b, valid, corrections.get(neighbour))
            n_valid = int(np.count_nonzero(valid))
            if n_valid < 64:
                continue
            mae = float(np.mean(np.abs(a[:, valid] - b[:, valid])))
            risk = (
                dates is not None
                and dates.get(xy) != dates.get(neighbour)
            ) or (
                independent_dates is not None
                and independent_dates.get(xy) != independent_dates.get(neighbour)
            )
            group = "identity_risk" if risk else "same_identity"
            groups[group].append(mae)
            groups["all"].append(mae)
            records.append(
                {
                    "first": list(xy),
                    "second": list(neighbour),
                    "group": group,
                    "mae": mae,
                    "n_valid_pixels": n_valid,
                }
            )
    return {
        "metric": "corresponding_overlap_rgb_mae",
        "margin_px": margin_px,
        "same_identity": _values_summary(groups["same_identity"]),
        "identity_risk": _values_summary(groups["identity_risk"]),
        "all": _values_summary(groups["all"]),
        "pairs": records,
    }


def write_mosaic(
    sources: list[Path],
    out_path: Path,
    *,
    cog: bool,
    method: str,
    feather_px: float,
    corrections: dict[Path, np.ndarray | dict] | None = None,
) -> dict:
    if method == "feather":
        return _stream_feather(
            sources,
            out_path,
            cog=cog,
            feather_px=feather_px,
            corrections=corrections,
        )
    else:
        datasets = [rasterio.open(p) for p in sources]
        try:
            if method == "mean":
                mosaic, transform = _merge_mean(datasets, nodata=0.0)
            else:
                mosaic, transform = merge(datasets, nodata=0.0, method=method)
            crs = datasets[0].crs
        finally:
            for ds in datasets:
                ds.close()

    count, height, width = mosaic.shape
    profile = {
        "driver": "COG" if cog else "GTiff",
        "height": height,
        "width": width,
        "count": count,
        "dtype": mosaic.dtype,
        "crs": crs,
        "transform": transform,
        "nodata": 0.0,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "BIGTIFF": "YES",
    }
    if cog:
        profile.update(
            {
                "blocksize": 512,
                "compress": "deflate",
                "resampling": Resampling.bilinear,
                "BIGTIFF": "YES",
            }
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mosaic)
        for i in range(1, count + 1):
            dst.set_band_description(i, ("R", "G", "B", "A")[i - 1] if i <= 4 else f"B{i}")
    bounds = array_bounds(height, width, transform)
    gsd = float(abs(transform.a))
    return {
        "path": str(out_path),
        "width": width,
        "height": height,
        "count": count,
        "gsd_m": gsd,
        "extent_m": [width * gsd, height * gsd],
        "bounds": list(bounds),
        "crs": str(crs),
        "n_sources": len(sources),
        "bytes": out_path.stat().st_size,
        "cog": bool(cog),
        "method": method,
        "feather_px": feather_px if method == "feather" else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output GeoTIFF/COG (default: production/mosaics/<parent>_sr_2p5m.tif)",
    )
    ap.add_argument(
        "--run-prefix",
        default=DEFAULT_RUN_PREFIX,
        help="Must match run_production.py --run-prefix (default prod_k4).",
    )
    ap.add_argument(
        "--layer",
        choices=["sr", "bilinear"],
        default="sr",
        help="Which per-tile GeoTIFF to mosaic (default: sr_pred).",
    )
    ap.add_argument(
        "--method",
        choices=["first", "last", "min", "max", "sum", "count", "mean", "feather"],
        default=None,
        help="Blend method. Default: feather if overlap_frac>0 else first.",
    )
    ap.add_argument(
        "--feather-px",
        type=float,
        default=None,
        help="HR-pixel ramp width for --method feather. "
        "Default: overlap_px * (sr_gsd inferred as 4× LR) from manifest.",
    )
    ap.add_argument(
        "--no-cog",
        action="store_true",
        help="Write classic GeoTIFF instead of COG.",
    )
    ap.add_argument(
        "--allow-missing",
        action="store_true",
        help="Mosaic whatever tiles exist; fail only if none exist.",
    )
    ap.add_argument("--identity-plan", type=Path, default=None)
    ap.add_argument("--harmonize", action="store_true")
    ap.add_argument(
        "--harmonize-mode",
        choices=["additive", "date-affine"],
        default="additive",
        help="Additive edge-strip correction, or global per-date affine overlap normalization.",
    )
    ap.add_argument("--harmonize-strip-px", type=int, default=256)
    ap.add_argument("--harmonize-segments", type=int, default=16)
    ap.add_argument("--harmonize-regularization", type=float, default=0.02)
    ap.add_argument("--harmonize-max-offset", type=float, default=0.04)
    ap.add_argument("--harmonize-smoothness", type=float, default=0.1)
    ap.add_argument("--harmonize-affine-samples-per-pair", type=int, default=512)
    ap.add_argument("--harmonize-gain-regularization", type=float, default=0.2)
    ap.add_argument("--harmonize-offset-regularization", type=float, default=1.0)
    ap.add_argument("--harmonize-max-gain-delta", type=float, default=0.15)
    ap.add_argument("--max-same-identity-overlap-p95", type=float, default=None)
    ap.add_argument("--max-identity-risk-overlap-p95", type=float, default=None)
    ap.add_argument(
        "--allow-qa-fail",
        action="store_true",
        help="Write a diagnostic mosaic while recording failed overlap gates.",
    )
    args = ap.parse_args()

    man_path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    man = json.loads(man_path.read_text())
    parent = man["parent"]
    sources, missing = collect_sources(man, layer=args.layer, run_prefix=args.run_prefix)
    if missing and not args.allow_missing:
        raise SystemExit(
            f"{len(missing)}/{len(man['tiles'])} tiles missing SR GeoTIFFs "
            f"(e.g. {missing[0]['expected']}). Pass --allow-missing to mosaic partial."
        )
    if not sources:
        raise SystemExit("no source tiles found")

    tile_rows = []
    for tile in man["tiles"]:
        path = (
            _sr_path(parent, tile["tile_id"], args.run_prefix)
            if args.layer == "sr"
            else _bil_path(parent, tile["tile_id"], args.run_prefix)
        )
        if path.is_file():
            tile_rows.append((tile, path))
    grid = {
        (int(tile["iy"]), int(tile["ix"])): path for tile, path in tile_rows
    }
    identity_path = None
    identity = None
    dates = None
    independent_dates = None
    if args.identity_plan is not None:
        identity_path = (
            args.identity_plan
            if args.identity_plan.is_absolute()
            else ROOT / args.identity_plan
        )
        identity = json.loads(identity_path.read_text())
        assignment = identity.get("assignment") or {}
        independent = identity.get("independent") or {}
        dates = {
            (int(tile["iy"]), int(tile["ix"])): assignment.get(tile["tile_id"])
            for tile, _path in tile_rows
        }
        independent_dates = {
            (int(tile["iy"]), int(tile["ix"])): independent.get(tile["tile_id"])
            for tile, _path in tile_rows
        }
        missing_identity = [
            tile["tile_id"]
            for tile, _path in tile_rows
            if assignment.get(tile["tile_id"]) is None
        ]
        if missing_identity:
            raise SystemExit(
                f"identity plan missing {len(missing_identity)} source tiles"
            )
    harmony = None
    corrections_xy: dict[tuple[int, int], np.ndarray | dict] = {}
    if args.harmonize:
        if dates is None:
            raise SystemExit("--harmonize requires --identity-plan")
        if args.harmonize_mode == "date-affine":
            harmony = estimate_date_affine_harmonization(
                grid,
                dates=dates,
                samples_per_pair=args.harmonize_affine_samples_per_pair,
                gain_regularization=args.harmonize_gain_regularization,
                offset_regularization=args.harmonize_offset_regularization,
                max_gain_delta=args.harmonize_max_gain_delta,
                max_offset=args.harmonize_max_offset,
            )
        else:
            harmony = estimate_harmonization(
                grid,
                dates=dates,
                independent_dates=independent_dates,
                strip_px=args.harmonize_strip_px,
                segments=args.harmonize_segments,
                regularization=args.harmonize_regularization,
                max_offset=args.harmonize_max_offset,
                smoothness=args.harmonize_smoothness,
            )
        corrections_xy = harmony["corrections"]
    corrections_by_path = {
        grid[xy]: correction for xy, correction in corrections_xy.items()
    }
    overlap_qa = measure_overlap_qa(
        grid,
        corrections=corrections_xy,
        dates=dates,
        independent_dates=independent_dates,
    )
    same_p95 = overlap_qa["same_identity"]["p95"]
    risk_p95 = overlap_qa["identity_risk"]["p95"]
    qa_failures = []
    if (
        args.max_same_identity_overlap_p95 is not None
        and same_p95 is not None
        and same_p95 > args.max_same_identity_overlap_p95
    ):
        qa_failures.append(
            f"same-identity overlap p95 {same_p95:.6f} exceeds "
            f"{args.max_same_identity_overlap_p95:.6f}"
        )
    if (
        args.max_identity_risk_overlap_p95 is not None
        and risk_p95 is not None
        and risk_p95 > args.max_identity_risk_overlap_p95
    ):
        qa_failures.append(
            f"identity-risk overlap p95 {risk_p95:.6f} exceeds "
            f"{args.max_identity_risk_overlap_p95:.6f}"
        )
    if qa_failures and not args.allow_qa_fail:
        raise SystemExit("; ".join(qa_failures))

    ovl = float(man.get("overlap_frac") or 0)
    method = args.method
    if method is None:
        method = "feather" if ovl > 0 else "first"

    # LR overlap → HR feather width (production df=4).
    overlap_lr = int(man.get("overlap_px") or round(float(man.get("side", 512)) * ovl))
    feather_px = float(args.feather_px) if args.feather_px is not None else float(max(1, overlap_lr * 4))

    tag = "sr" if args.layer == "sr" else "bilinear"
    ovl_suffix = f"_ovl{int(round(ovl * 100))}" if ovl > 0 else ""
    method_suffix = f"_{method}" if method == "feather" and ovl > 0 else ""
    out = args.out
    if out is None:
        out = (
            ROOT
            / "production"
            / "mosaics"
            / f"{parent}{ovl_suffix}{method_suffix}_{tag}_2p5m.tif"
        )
    elif not out.is_absolute():
        out = ROOT / out

    print(
        f"Mosaic {len(sources)} tiles method={method} feather_px={feather_px:g}",
        flush=True,
    )
    meta = write_mosaic(
        sources,
        out,
        cog=not args.no_cog,
        method=method,
        feather_px=feather_px,
        corrections=corrections_by_path,
    )
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(man_path.relative_to(ROOT)) if man_path.is_relative_to(ROOT) else str(man_path),
        "parent": parent,
        "run_prefix": args.run_prefix,
        "layer": args.layer,
        "merge_method": method,
        "feather_px": feather_px if method == "feather" else None,
        "overlap_frac": man.get("overlap_frac"),
        "n_manifest_tiles": len(man["tiles"]),
        "n_missing": len(missing),
        "missing_sample": missing[:5],
        "identity_plan": (
            None
            if identity_path is None
            else str(
                identity_path.relative_to(ROOT)
                if identity_path.is_relative_to(ROOT)
                else identity_path
            )
        ),
        "harmonization": (
            None
            if harmony is None
            else {
                "configuration": harmony["configuration"],
                "metrics": harmony["metrics"],
                "corrections": {
                    f"{xy[0]},{xy[1]}": (
                        {
                            "gain": [float(value) for value in correction["gain"]],
                            "offset": [float(value) for value in correction["offset"]],
                        }
                        if isinstance(correction, dict)
                        else [float(value) for value in correction]
                    )
                    for xy, correction in corrections_xy.items()
                },
                "edges": harmony["records"],
            }
        ),
        "overlap_qa": overlap_qa,
        "qa": {
            "max_same_identity_overlap_p95": args.max_same_identity_overlap_p95,
            "max_identity_risk_overlap_p95": args.max_identity_risk_overlap_p95,
            "passed": not qa_failures,
            "failures": qa_failures,
            "diagnostic_override": bool(args.allow_qa_fail and qa_failures),
        },
        "mosaic": meta,
    }
    side = out.with_suffix(out.suffix + ".json")
    side.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"Wrote {out} ({meta['width']}×{meta['height']} @ {meta['gsd_m']:g} m, "
        f"{meta['bytes']/1e9:.2f} GB, {meta['n_sources']} tiles, method={method})",
        flush=True,
    )
    print(f"Meta {side}", flush=True)


if __name__ == "__main__":
    main()
