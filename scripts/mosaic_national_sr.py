#!/usr/bin/env python3
"""Reproject and merge per-granule SR mosaics into one cross-granule COG.

The destination is produced a tile at a time.  Memory use is therefore bounded
by the output block size rather than the national mosaic dimensions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
import uuid
from collections import OrderedDict
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.shutil import copy as rio_copy
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from rasterio.windows import Window
from rasterio.windows import bounds as window_bounds

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_NODATA = 0.0
DEFAULT_BLOCK_SIZE = 512


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as src:
        for chunk in iter(lambda: src.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _scan_sources(
    sources: list[Path], dst_crs: str, resolution: float
) -> tuple[list[dict[str, Any]], Affine, int, int, int, str]:
    if not sources:
        raise ValueError("at least one source is required")
    if not math.isfinite(resolution) or resolution <= 0:
        raise ValueError("resolution must be a positive finite number")

    details: list[dict[str, Any]] = []
    union: tuple[float, float, float, float] | None = None
    band_count: int | None = None
    dtype: str | None = None

    for path in sources:
        stat = path.stat()
        with rasterio.open(path) as src:
            if src.crs is None:
                raise ValueError(f"source has no CRS: {path}")
            if src.count < 1:
                raise ValueError(f"source has no raster bands: {path}")
            if band_count is None:
                band_count = src.count
                dtype = src.dtypes[0]
            elif src.count != band_count:
                raise ValueError(
                    f"source band count differs ({src.count} != {band_count}): {path}"
                )
            dst_bounds = transform_bounds(
                src.crs, dst_crs, *src.bounds, densify_pts=21
            )
            union = (
                dst_bounds
                if union is None
                else (
                    min(union[0], dst_bounds[0]),
                    min(union[1], dst_bounds[1]),
                    max(union[2], dst_bounds[2]),
                    max(union[3], dst_bounds[3]),
                )
            )
            details.append(
                {
                    "path": str(path),
                    "sha256": _sha256(path),
                    "size_bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "crs": str(src.crs),
                    "bounds_dst_crs": list(dst_bounds),
                    "nodata": src.nodata,
                }
            )

    assert union is not None and band_count is not None and dtype is not None
    left = math.floor(union[0] / resolution) * resolution
    bottom = math.floor(union[1] / resolution) * resolution
    right = math.ceil(union[2] / resolution) * resolution
    top = math.ceil(union[3] / resolution) * resolution
    width = max(1, int(round((right - left) / resolution)))
    height = max(1, int(round((top - bottom) / resolution)))
    return details, from_origin(left, top, resolution, resolution), width, height, band_count, dtype


def _source_crs_summary(details: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for detail in details:
        crs = detail["crs"]
        counts[crs] = counts.get(crs, 0) + 1
    return {
        "all_same": len(counts) == 1,
        "unique_crs": sorted(counts),
        "counts": counts,
    }


class _VRTCache:
    """A bounded LRU of globally aligned WarpedVRTs."""

    def __init__(
        self,
        details: list[dict[str, Any]],
        *,
        crs: str,
        transform: Affine,
        width: int,
        height: int,
        max_open: int,
    ) -> None:
        self.details = details
        self.crs = crs
        self.transform = transform
        self.width = width
        self.height = height
        self.max_open = max(1, max_open)
        self._entries: OrderedDict[int, tuple[Any, WarpedVRT]] = OrderedDict()

    def get(self, index: int) -> WarpedVRT:
        if index in self._entries:
            dataset, vrt = self._entries.pop(index)
            self._entries[index] = (dataset, vrt)
            return vrt
        dataset = rasterio.open(self.details[index]["path"])
        try:
            vrt = WarpedVRT(
                dataset,
                crs=self.crs,
                transform=self.transform,
                width=self.width,
                height=self.height,
                resampling=Resampling.bilinear,
                add_alpha=True,
            )
        except Exception:
            dataset.close()
            raise
        self._entries[index] = (dataset, vrt)
        if len(self._entries) > self.max_open:
            _, (old_dataset, old_vrt) = self._entries.popitem(last=False)
            old_vrt.close()
            old_dataset.close()
        return vrt

    def close(self) -> None:
        while self._entries:
            _, (dataset, vrt) = self._entries.popitem(last=False)
            vrt.close()
            dataset.close()

    def __enter__(self) -> _VRTCache:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def _intersects(
    source_bounds: list[float], destination_bounds: tuple[float, float, float, float]
) -> bool:
    sl, sb, sr, st = source_bounds
    dl, db, dr, dt = destination_bounds
    return sl < dr and sr > dl and sb < dt and st > db


def _build_row_source_index(
    details: list[dict[str, Any]],
    transform: Affine,
    width: int,
    height: int,
    block_size: int,
) -> dict[int, list[tuple[int, int, int]]]:
    """Index source column spans by destination block row."""
    block_rows = math.ceil(height / block_size)
    block_columns = math.ceil(width / block_size)
    left, top, resolution = transform.c, transform.f, transform.a
    rows: dict[int, list[tuple[int, int, int]]] = {}
    for source_index, detail in enumerate(details):
        source_left, source_bottom, source_right, source_top = detail["bounds_dst_crs"]
        column_start = max(
            0, math.floor((source_left - left) / resolution / block_size)
        )
        column_stop = min(
            block_columns - 1,
            math.ceil((source_right - left) / resolution / block_size) - 1,
        )
        row_start = max(0, math.floor((top - source_top) / resolution / block_size))
        row_stop = min(
            block_rows - 1,
            math.ceil((top - source_bottom) / resolution / block_size) - 1,
        )
        for block_row in range(row_start, row_stop + 1):
            rows.setdefault(block_row, []).append(
                (column_start, column_stop, source_index)
            )
    return rows


def _write_window(
    destination: Any, data: np.ndarray, valid: np.ndarray, window: Window
) -> None:
    """Small seam kept separate so tests can assert every write is windowed."""
    destination.write(data, window=window)
    destination.write_mask(np.any(valid, axis=0).astype("uint8") * 255, window=window)


def _is_bigtiff(path: Path) -> bool:
    with path.open("rb") as src:
        header = src.read(4)
    return header in {b"II+\x00", b"MM\x00+"}


def _threshold_checks(
    verification: dict[str, Any],
    *,
    max_overlap_mae: float | None,
    max_nodata_fraction: float | None,
) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    if max_overlap_mae is not None:
        if not math.isfinite(max_overlap_mae) or max_overlap_mae < 0:
            raise ValueError("max_overlap_mae must be a nonnegative finite number")
        mae = verification["cross_source_overlap_mae"]
        checks["overlap_mae_threshold"] = mae is None or mae <= max_overlap_mae
    if max_nodata_fraction is not None:
        if not math.isfinite(max_nodata_fraction) or not 0 <= max_nodata_fraction <= 1:
            raise ValueError("max_nodata_fraction must be between 0 and 1")
        checks["nodata_fraction_threshold"] = (
            verification["nodata_gap_fraction"] <= max_nodata_fraction
        )
    return checks


def _validate_dataset(
    path: Path,
    *,
    expected: dict[str, Any] | None = None,
    compute_coverage: bool = True,
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    valid_pixels = 0
    with rasterio.open(path) as src:
        checks["readable"] = True
        block_height, block_width = src.block_shapes[0]
        checks["tiled"] = bool(src.profile.get("tiled")) or (
            src.width <= block_width and src.height <= block_height
        )
        checks["compressed"] = str(src.compression).lower() not in {"none", ""}
        checks["bigtiff"] = _is_bigtiff(path)
        if expected is not None:
            checks["width"] = src.width == expected["width"]
            checks["height"] = src.height == expected["height"]
            checks["count"] = src.count == expected["count"]
            checks["crs"] = src.crs == rasterio.crs.CRS.from_user_input(expected["crs"])
            checks["transform"] = src.transform.almost_equals(
                Affine(*expected["transform"])
            )
        if compute_coverage:
            for _, window in src.block_windows(1):
                valid_pixels += int(np.count_nonzero(src.dataset_mask(window=window)))
        image_structure = src.tags(ns="IMAGE_STRUCTURE")
        layout = image_structure.get("LAYOUT", "")
        checks["cog_layout"] = layout.upper() == "COG"
        dimensions = {
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "dtype": src.dtypes[0],
            "block_shapes": [list(shape) for shape in src.block_shapes],
        }
        total_pixels = src.width * src.height

    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"output validation failed for {path}: {', '.join(failed)}")
    result: dict[str, Any] = {"checks": checks, "dimensions": dimensions}
    if compute_coverage:
        result["coverage"] = {
            "valid_pixels": valid_pixels,
            "nodata_pixels": total_pixels - valid_pixels,
            "total_pixels": total_pixels,
            "valid_fraction": valid_pixels / total_pixels if total_pixels else 0.0,
        }
    return result


def validate_output(
    out: Path,
    *,
    max_overlap_mae: float | None = None,
    max_nodata_fraction: float | None = None,
) -> dict[str, Any]:
    """Validate COG structure, checksum, coverage, and source inventory."""
    sidecar = out.with_suffix(out.suffix + ".json")
    if not sidecar.is_file():
        raise RuntimeError(f"metadata sidecar is missing: {sidecar}")
    metadata = json.loads(sidecar.read_text())
    expected = {
        "width": metadata["width"],
        "height": metadata["height"],
        "count": metadata["count"],
        "crs": metadata["dst_crs"],
        "transform": metadata["transform"],
    }
    result = _validate_dataset(out, expected=expected)
    checks = result["checks"]
    checks["sidecar_present"] = True
    checks["coverage_matches_sidecar"] = result["coverage"] == metadata.get("coverage")
    checks["output_checksum"] = _sha256(out) == metadata.get("output_sha256")
    checks["output_size"] = out.stat().st_size == metadata.get("output_size_bytes")

    details = metadata.get("source_details", [])
    checks["source_inventory_count"] = bool(details) and len(details) == len(
        metadata.get("sources", [])
    )
    checks["source_inventory_order"] = [item.get("path") for item in details] == metadata.get(
        "sources"
    )
    source_files_present = True
    source_checksums = True
    source_crs = True
    current_details: list[dict[str, Any]] = []
    for detail in details:
        source_path = Path(detail["path"])
        if not source_path.is_file():
            source_files_present = False
            source_checksums = False
            source_crs = False
            continue
        source_checksums &= _sha256(source_path) == detail.get("sha256")
        try:
            with rasterio.open(source_path) as src:
                current_crs = str(src.crs)
        except (OSError, rasterio.errors.RasterioError):
            source_crs = False
            continue
        source_crs &= current_crs == detail.get("crs")
        current_details.append({"crs": current_crs})
    checks["source_files_present"] = source_files_present
    checks["source_checksums"] = source_checksums
    checks["source_crs_inventory"] = (
        len(current_details) == len(details)
        and _source_crs_summary(current_details) == metadata.get("source_crs")
        and source_crs
    )
    checks.update(
        _threshold_checks(
            metadata["verification"],
            max_overlap_mae=max_overlap_mae,
            max_nodata_fraction=max_nodata_fraction,
        )
    )
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"output validation failed for {out}: {', '.join(failed)}")
    result["sidecar_present"] = True
    result["source_inventory"] = {
        "count": len(details),
        "crs": metadata["source_crs"],
    }
    return result


def mosaic_sources(
    sources: list[Path],
    out: Path,
    *,
    dst_crs: str,
    resolution: float,
    block_size: int = DEFAULT_BLOCK_SIZE,
    max_open_sources: int = 32,
    validate: bool = True,
    resume: bool = True,
    max_overlap_mae: float | None = None,
    max_nodata_fraction: float | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    if block_size < 16 or block_size % 16:
        raise ValueError("block_size must be a multiple of 16")
    if max_open_sources < 1:
        raise ValueError("max_open_sources must be at least 1")
    scan_started = time.monotonic()
    details, transform, width, height, count, dtype = _scan_sources(
        sources, dst_crs, resolution
    )
    scan_seconds = time.monotonic() - scan_started
    fingerprint_payload = {
        "sources": [{"path": item["path"], "sha256": item["sha256"]} for item in details],
        "dst_crs": str(rasterio.crs.CRS.from_user_input(dst_crs)),
        "resolution": resolution,
        "block_size": block_size,
        "nodata": DEFAULT_NODATA,
        "resampling": "bilinear",
        "merge": "first-valid-source",
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode()
    ).hexdigest()
    sidecar = out.with_suffix(out.suffix + ".json")
    if resume and out.is_file() and sidecar.is_file():
        try:
            prior = json.loads(sidecar.read_text())
        except (OSError, ValueError):
            prior = {}
        if prior.get("build_fingerprint") == fingerprint:
            try:
                validation = (
                    validate_output(
                        out,
                        max_overlap_mae=max_overlap_mae,
                        max_nodata_fraction=max_nodata_fraction,
                    )
                    if validate
                    else prior.get("validation")
                )
            except (OSError, ValueError, KeyError, RuntimeError):
                pass
            else:
                return {
                    **prior,
                    "resumed": True,
                    "resume_validation": validation,
                    "resume_check_seconds": time.monotonic() - started,
                }

    out.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    temporary_gtiff = out.with_name(f".{out.name}.{token}.partial.tif")
    temporary_cog = out.with_name(f".{out.name}.{token}.partial.cog.tif")
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": count,
        "dtype": dtype,
        "crs": dst_crs,
        "transform": transform,
        "nodata": DEFAULT_NODATA,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": block_size,
        "blockysize": block_size,
        "BIGTIFF": "YES",
    }
    windows_written = 0
    source_window_reads = 0
    valid_pixels = 0
    overlap_pixels = 0
    overlap_source_observations = 0
    duplicate_valid_observations = 0
    overlap_absolute_error_sum = 0.0
    overlap_band_observations = 0
    per_source = [
        {
            "path": detail["path"],
            "source_valid_pixels": 0,
            "contributed_valid_pixels": 0,
            "overlap_pixels": 0,
            "overlap_band_observations": 0,
            "overlap_absolute_error_sum": 0.0,
        }
        for detail in details
    ]
    row_source_index = _build_row_source_index(
        details, transform, width, height, block_size
    )
    write_started = time.monotonic()
    try:
        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True), ExitStack() as stack:
            destination = stack.enter_context(rasterio.open(temporary_gtiff, "w", **profile))
            cache = stack.enter_context(
                _VRTCache(
                    details,
                    crs=dst_crs,
                    transform=transform,
                    width=width,
                    height=height,
                    max_open=max_open_sources,
                )
            )
            band_indexes = list(range(1, count + 1))
            for (block_row, block_column), window in destination.block_windows(1):
                window_shape = (count, int(window.height), int(window.width))
                output = np.full(window_shape, DEFAULT_NODATA, dtype=dtype)
                valid = np.zeros(window_shape, dtype=bool)
                source_coverage_count = np.zeros(window_shape[1:], dtype=np.uint32)
                bounds = window_bounds(window, transform)
                candidates = (
                    source_index
                    for column_start, column_stop, source_index in row_source_index.get(
                        block_row, []
                    )
                    if column_start <= block_column <= column_stop
                )
                for source_index in candidates:
                    detail = details[source_index]
                    if not _intersects(detail["bounds_dst_crs"], bounds):
                        continue
                    vrt = cache.get(source_index)
                    source = vrt.read(
                        indexes=band_indexes,
                        window=window,
                        out_dtype=dtype,
                        masked=True,
                    )
                    source_window_reads += 1
                    source_data = np.ma.getdata(source)
                    source_valid = ~np.ma.getmaskarray(source)
                    if np.issubdtype(source_data.dtype, np.floating):
                        source_valid &= np.isfinite(source_data)
                    source_pixel_valid = np.any(source_valid, axis=0)
                    source_metric = per_source[source_index]
                    source_metric["source_valid_pixels"] += int(
                        np.count_nonzero(source_pixel_valid)
                    )
                    source_metric["overlap_pixels"] += int(
                        np.count_nonzero(source_pixel_valid & (source_coverage_count > 0))
                    )
                    source_coverage_count += source_pixel_valid

                    comparable = source_valid & valid
                    comparable_count = int(np.count_nonzero(comparable))
                    if comparable_count:
                        absolute_error = np.abs(
                            source_data.astype(np.float64, copy=False)
                            - output.astype(np.float64, copy=False)
                        )
                        error_sum = float(np.sum(absolute_error, where=comparable))
                        overlap_absolute_error_sum += error_sum
                        overlap_band_observations += comparable_count
                        source_metric["overlap_band_observations"] += comparable_count
                        source_metric["overlap_absolute_error_sum"] += error_sum

                    take = source_valid & ~valid
                    source_metric["contributed_valid_pixels"] += int(
                        np.count_nonzero(np.any(take, axis=0))
                    )
                    np.copyto(output, source_data, where=take)
                    valid |= take
                _write_window(destination, output, valid, window)
                window_valid_pixels = int(np.count_nonzero(source_coverage_count))
                valid_pixels += window_valid_pixels
                overlap_mask = source_coverage_count >= 2
                overlap_pixels += int(np.count_nonzero(overlap_mask))
                overlap_source_observations += int(
                    np.sum(source_coverage_count, where=overlap_mask)
                )
                duplicate_valid_observations += int(
                    np.sum(
                        source_coverage_count - 1,
                        where=source_coverage_count > 0,
                    )
                )
                windows_written += 1
        write_seconds = time.monotonic() - write_started

        cog_started = time.monotonic()
        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
            rio_copy(
                temporary_gtiff,
                temporary_cog,
                driver="COG",
                COMPRESS="DEFLATE",
                BLOCKSIZE=max(128, block_size),
                BIGTIFF="YES",
                OVERVIEWS="AUTO",
            )
        cog_seconds = time.monotonic() - cog_started

        expected = {
            "width": width,
            "height": height,
            "count": count,
            "crs": dst_crs,
            "transform": [
                transform.a,
                transform.b,
                transform.c,
                transform.d,
                transform.e,
                transform.f,
            ],
        }
        validation_started = time.monotonic()
        validation = (
            _validate_dataset(temporary_cog, expected=expected)
            if validate
            else {"checks": {"skipped": True}}
        )
        validation_seconds = time.monotonic() - validation_started
        total_pixels = width * height
        coverage = {
            "valid_pixels": valid_pixels,
            "nodata_pixels": total_pixels - valid_pixels,
            "total_pixels": total_pixels,
            "valid_fraction": valid_pixels / total_pixels if total_pixels else 0.0,
        }
        for source_metric in per_source:
            comparisons = source_metric["overlap_band_observations"]
            source_metric["overlap_mae"] = (
                source_metric["overlap_absolute_error_sum"] / comparisons
                if comparisons
                else None
            )
        verification = {
            "metric_definitions": {
                "source_valid_pixels": "Destination pixels valid in each reprojected source.",
                "contributed_valid_pixels": (
                    "Destination pixels where a source supplied at least one first-valid band."
                ),
                "duplicate_valid_observations": (
                    "Source-valid pixel observations beyond the first at each destination pixel."
                ),
                "overlap_pixels": "Unique destination pixels valid in two or more sources.",
                "overlap_source_observations": (
                    "All source-valid observations at pixels valid in two or more sources."
                ),
                "cross_source_overlap_mae": (
                    "Band MAE for each later source against the first-valid selected value."
                ),
            },
            "per_source": per_source,
            "total_source_valid_pixels": sum(
                item["source_valid_pixels"] for item in per_source
            ),
            "unique_output_valid_pixels": valid_pixels,
            "duplicate_valid_observations": duplicate_valid_observations,
            "overlap_pixels": overlap_pixels,
            "overlap_source_observations": overlap_source_observations,
            "cross_source_overlap_band_observations": overlap_band_observations,
            "cross_source_overlap_absolute_error_sum": overlap_absolute_error_sum,
            "cross_source_overlap_mae": (
                overlap_absolute_error_sum / overlap_band_observations
                if overlap_band_observations
                else None
            ),
            "nodata_gap_pixels": total_pixels - valid_pixels,
            "nodata_gap_fraction": (
                (total_pixels - valid_pixels) / total_pixels if total_pixels else 0.0
            ),
        }
        threshold_checks = _threshold_checks(
            verification,
            max_overlap_mae=max_overlap_mae,
            max_nodata_fraction=max_nodata_fraction,
        )
        failed_thresholds = [
            name for name, passed in threshold_checks.items() if not passed
        ]
        if failed_thresholds:
            raise RuntimeError(
                f"mosaic verification failed: {', '.join(failed_thresholds)}"
            )
        if validate and validation.get("coverage") != coverage:
            raise RuntimeError("streaming coverage statistics differ from output mask")
        os.replace(temporary_cog, out)
    finally:
        temporary_gtiff.unlink(missing_ok=True)
        temporary_cog.unlink(missing_ok=True)

    checksum_started = time.monotonic()
    output_sha256 = _sha256(out)
    checksum_seconds = time.monotonic() - checksum_started
    validation["checks"].update(
        {
            "output_checksum": True,
            "output_size": True,
            "source_inventory_count": True,
            "source_inventory_order": True,
            "source_files_present": True,
            "source_checksums": True,
            "source_crs_inventory": True,
            **threshold_checks,
        }
    )
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sources": [str(path) for path in sources],
        "source_details": details,
        "source_hashes": {item["path"]: item["sha256"] for item in details},
        "source_crs": _source_crs_summary(details),
        "out": str(out),
        "output_sha256": output_sha256,
        "output_size_bytes": out.stat().st_size,
        "dst_crs": str(rasterio.crs.CRS.from_user_input(dst_crs)),
        "resolution": resolution,
        "width": width,
        "height": height,
        "count": count,
        "dtype": dtype,
        "transform": [
            transform.a,
            transform.b,
            transform.c,
            transform.d,
            transform.e,
            transform.f,
        ],
        "bounds_dst_crs": [
            transform.c,
            transform.f + height * transform.e,
            transform.c + width * transform.a,
            transform.f,
        ],
        "coverage": coverage,
        "verification": verification,
        "streaming": {
            "block_size": block_size,
            "windows_written": windows_written,
            "source_window_reads": source_window_reads,
            "max_open_sources": max_open_sources,
        },
        "timing_seconds": {
            "source_scan_and_hash": scan_seconds,
            "windowed_write": write_seconds,
            "cog_conversion": cog_seconds,
            "validation": validation_seconds,
            "output_checksum": checksum_seconds,
            "total": time.monotonic() - started,
        },
        "validation": validation,
        "build_fingerprint": fingerprint,
        "resumed": False,
    }
    _atomic_json(sidecar, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--dst-crs",
        default="EPSG:3035",
        help="Common metric CRS; EPSG:3035 avoids mixing Norway's UTM zones.",
    )
    parser.add_argument("--resolution", type=float, default=2.5)
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--max-open-sources", type=int, default=32)
    parser.add_argument(
        "--max-overlap-mae",
        type=float,
        help="Fail when cross-source overlap MAE exceeds this value.",
    )
    parser.add_argument(
        "--max-nodata-fraction",
        type=float,
        help="Fail when the fraction of uncovered output pixels exceeds this value.",
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-validate", action="store_true")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate --out and its metadata sidecar without rebuilding.",
    )
    args = parser.parse_args()
    out = args.out if args.out.is_absolute() else ROOT / args.out
    if args.validate_only:
        print(
            json.dumps(
                validate_output(
                    out,
                    max_overlap_mae=args.max_overlap_mae,
                    max_nodata_fraction=args.max_nodata_fraction,
                ),
                indent=2,
            )
        )
        return
    if not args.sources:
        parser.error("--sources is required unless --validate-only is used")
    sources = [path if path.is_absolute() else ROOT / path for path in args.sources]
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise SystemExit(f"missing source mosaics: {missing}")
    summary = mosaic_sources(
        sources,
        out,
        dst_crs=args.dst_crs,
        resolution=args.resolution,
        block_size=args.block_size,
        max_open_sources=args.max_open_sources,
        validate=not args.no_validate,
        resume=not args.no_resume,
        max_overlap_mae=args.max_overlap_mae,
        max_nodata_fraction=args.max_nodata_fraction,
    )
    action = "Reused" if summary.get("resumed") else "Wrote"
    print(f"{action} cross-granule mosaic {out}")


if __name__ == "__main__":
    main()
