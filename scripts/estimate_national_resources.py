#!/usr/bin/env python3
"""CPU-only preflight resource estimate for national production configurations.

The estimates are deliberately ranges.  They are planning numbers, not
benchmarks: every calibration is emitted in the JSON and can be overridden.
No STAC requests, raster reads, model imports, or GPU probes are performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_national_production as national  # noqa: E402, I001


DEFAULTS = {
    # Broad defaults intended to be replaced by site-specific measurements.
    "gpu_hours_per_cell": [0.018, 0.023],
    "temporary_gib_per_active_cell": [0.25, 0.75],
    "cog_bytes_per_pixel": [1.0, 3.0],
    "mosaic_bytes_per_pixel": [1.0, 3.0],
    "sr_scale": 4,
    "unplanned_cells_per_granule": [1, 441],
}


def _load_object(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _hash_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _fingerprint(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return {
        "path": str(path),
        "sha256": national.sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _resolve_config_path(config: dict, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    candidate = config["_config_path"].parent / path
    return candidate if candidate.is_file() else ROOT / path


def _inventory_bbox_path(config: dict) -> Path | None:
    inventory = config.get("inventory")
    if not isinstance(inventory, dict):
        return None
    return _resolve_config_path(config, inventory.get("mgrs_bboxes_json"))


def _availability_rows(config: dict) -> tuple[dict[str, dict], Path | None]:
    """Read a configured/local scene-availability artifact when one is discoverable."""
    candidates: list[Path] = []
    inventory = config.get("inventory")
    if isinstance(inventory, dict):
        for key in ("availability_json", "mgrs_availability_json"):
            path = _resolve_config_path(config, inventory.get(key))
            if path:
                candidates.append(path)
    candidates.append(ROOT / "production/cloud_availability/july2025_pm45/mgrs_availability.json")
    for path in candidates:
        payload = _load_object(path)
        rows = payload.get("tiles") if payload else None
        if isinstance(rows, list):
            result = {}
            for row in rows:
                if not isinstance(row, dict):
                    continue
                mgrs = str(row.get("mgrs") or row.get("mgrs_tile") or "").upper().lstrip("T")
                if mgrs:
                    result[mgrs] = row
            return result, path
    return {}, None


def _tile_id(mgrs: str, cell: dict, side: int) -> str:
    return str(
        cell.get("tile_id")
        or f"{mgrs}_t{side}_y{int(cell.get('iy', 0)):02d}_x{int(cell.get('ix', 0)):02d}"
    )


def _positive_cells(plan: dict | None) -> list[dict]:
    if not plan:
        return []
    return [
        cell
        for cell in (plan.get("cells") or [])
        if isinstance(cell, dict)
        and int(cell.get("n_frames") or len(cell.get("dates") or [])) > 0
    ]


def _no_pass_count(plan: dict | None) -> int:
    if not plan:
        return 0
    skipped = plan.get("skipped_no_pass")
    if isinstance(skipped, list):
        return len(skipped)
    return int(plan.get("n_cells_no_pass") or 0)


def _manifest_tiles(payload: dict | None) -> list[dict] | None:
    tiles = payload.get("tiles") if payload else None
    return [row for row in tiles if isinstance(row, dict)] if isinstance(tiles, list) else None


def _run_settings(config: dict, row: dict) -> tuple[str, str]:
    production = {**config.get("production", {}), **row.get("production", {})}
    return str(production.get("scope", "identity_changed")), str(
        production.get("run_prefix", national._recipe(config, row, "run_prefix", "prod_k4_icm"))
    )


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _gpu_calibration(
    config: dict, selected: list[dict], args: argparse.Namespace
) -> tuple[float, float, dict[str, Any], list[dict[str, Any]]]:
    default_low, default_high = DEFAULTS["gpu_hours_per_cell"]
    cli_override = args.gpu_hours_low is not None or args.gpu_hours_high is not None
    if cli_override:
        low = default_low if args.gpu_hours_low is None else args.gpu_hours_low
        high = default_high if args.gpu_hours_high is None else args.gpu_hours_high
        return low, high, {
            "method": "cli_override",
            "sample_count": 0,
            "source_files": [],
        }, []
    if not args.auto_calibrate:
        return default_low, default_high, {
            "method": "measured_default_range",
            "sample_count": 0,
            "source_files": [],
        }, []

    samples: list[float] = []
    sources: list[dict[str, Any]] = []
    for row in selected:
        path = national.paths_for(config, row)["summary"]
        summary = _load_object(path)
        if not summary:
            continue
        expected_side = int(national._recipe(config, row, "side", 512))
        if summary.get("side") is not None and int(summary["side"]) != expected_side:
            continue
        values = [
            float(tile["training_time_s"]) / 3600
            for tile in (summary.get("tiles") or [])
            if isinstance(tile, dict)
            and isinstance(tile.get("training_time_s"), (int, float))
            and float(tile["training_time_s"]) > 0
        ]
        if not values:
            continue
        samples.extend(values)
        fingerprint = _fingerprint(path)
        if fingerprint:
            sources.append({**fingerprint, "sample_count": len(values)})
    if not samples:
        return default_low, default_high, {
            "method": "measured_default_range_no_compatible_summaries",
            "sample_count": 0,
            "source_files": [],
        }, []

    # Trim isolated runtime noise with percentiles, add a broad 10% margin,
    # and retain the measured fallback bounds as conservative guardrails.
    observed_p10 = _percentile(samples, 0.10)
    observed_p90 = _percentile(samples, 0.90)
    low = min(default_low, observed_p10 * 0.90)
    high = max(default_high, observed_p90 * 1.10)
    return low, high, {
        "method": "summary_tile_times_p10_p90_with_10_percent_margin_and_default_guardrails",
        "sample_count": len(samples),
        "observed_mean_gpu_hours": round(sum(samples) / len(samples), 6),
        "observed_p10_gpu_hours": round(observed_p10, 6),
        "observed_p90_gpu_hours": round(observed_p90, 6),
        "source_files": sources,
    }, sources


def _output_status(parent: str, tile_id: str, run_prefix: str) -> tuple[bool, bool]:
    base = ROOT / "single_samples" / parent / "sample" / f"{run_prefix}_{tile_id}"
    return (base.joinpath("metrics.json").is_file(), base.joinpath("qgis/sr_pred.tif").is_file())


def _range(low: float | int, high: float | int, digits: int = 3) -> dict[str, float | int]:
    if isinstance(low, int) and isinstance(high, int):
        return {"low": low, "high": high}
    return {"low": round(float(low), digits), "high": round(float(high), digits)}


def _bbox_union(bboxes: list[list[float]]) -> list[float] | None:
    valid = [
        bbox
        for bbox in bboxes
        if isinstance(bbox, list)
        and len(bbox) == 4
        and all(isinstance(value, (int, float)) for value in bbox)
    ]
    if not valid:
        return None
    return [
        min(value[0] for value in valid),
        min(value[1] for value in valid),
        max(value[2] for value in valid),
        max(value[3] for value in valid),
    ]


def _projected_pixel_count(
    bbox: list[float] | None, dst_crs: str, resolution: float
) -> tuple[int | None, dict[str, Any]]:
    if bbox is None:
        return None, {"status": "unknown", "reason": "no selected-granule bbox available"}
    try:
        from pyproj import Transformer
    except ImportError:
        return None, {"status": "unknown", "reason": "pyproj is not installed"}
    try:
        transformer = Transformer.from_crs("EPSG:4326", dst_crs, always_xy=True)
        west, south, east, north = bbox
        points = []
        for index in range(33):
            fraction = index / 32
            lon = west + (east - west) * fraction
            lat = south + (north - south) * fraction
            points.extend([(lon, south), (lon, north), (west, lat), (east, lat)])
        projected = [transformer.transform(lon, lat) for lon, lat in points]
        xs, ys = zip(*projected)
        width = math.ceil((max(xs) - min(xs)) / resolution)
        height = math.ceil((max(ys) - min(ys)) / resolution)
        return width * height, {
            "status": "estimated",
            "method": "projected envelope of 33 samples per WGS84 bbox edge",
            "width_pixels": width,
            "height_pixels": height,
        }
    except Exception as exc:  # noqa: BLE001
        return None, {"status": "unknown", "reason": f"bbox projection failed: {exc}"}


def estimate(args: argparse.Namespace) -> dict[str, Any]:
    config = national.load_country_config(args.config, args.inventory)
    selected = national.select_granules(
        config["_granules"],
        requested=args.granules,
        wave_size=args.wave_size,
        wave_index=args.wave_index,
    )
    state_path = national._resolve(
        args.state, config["_paths"]["production"] / "run_state.json"
    )
    state = _load_object(state_path)
    bbox_path = _inventory_bbox_path(config)
    bbox_payload = _load_object(bbox_path) if bbox_path else None
    availability, availability_path = _availability_rows(config)
    gpu_hours_low, gpu_hours_high, gpu_calibration, calibration_sources = (
        _gpu_calibration(config, selected, args)
    )

    artifacts: list[dict[str, Any]] = []
    for candidate in (state_path, bbox_path, availability_path):
        if candidate:
            fingerprint = _fingerprint(candidate)
            if fingerprint and fingerprint not in artifacts:
                artifacts.append(fingerprint)
    artifacts.extend(calibration_sources)

    granule_results = []
    selected_bboxes: list[list[float]] = []
    target_low = target_high = complete_training = complete_cogs = no_pass_total = 0
    cog_storage_low = cog_storage_high = 0.0
    for row in selected:
        mgrs = row["mgrs"]
        paths = national.paths_for(config, row)
        side = int(national._recipe(config, row, "side", 512))
        plan_path = paths["ondisk"] if paths["ondisk"].is_file() else paths["plan"]
        plan = _load_object(plan_path)
        plan_fp = _fingerprint(plan_path)
        if plan_fp:
            artifacts.append(plan_fp)
        cells = _positive_cells(plan)
        no_pass = _no_pass_count(plan)
        no_pass_total += no_pass
        scope, run_prefix = _run_settings(config, row)

        effective = national.effective_identity_paths(config, row, selected)
        production_manifest_path = (
            effective["identity_manifest"]
            if scope == "all"
            else effective["identity_changed_manifest"]
        )
        production_manifest = _load_object(production_manifest_path)
        manifest_source = production_manifest_path if production_manifest else None
        tiles = _manifest_tiles(production_manifest)
        if tiles is None and scope == "all":
            base_manifest = _load_object(paths["manifest"])
            tiles = _manifest_tiles(base_manifest)
            if tiles is not None:
                manifest_source = paths["manifest"]
        if manifest_source:
            fingerprint = _fingerprint(manifest_source)
            if fingerprint:
                artifacts.append(fingerprint)

        availability_row = availability.get(mgrs)
        availability_scenes = (
            int(availability_row.get("n_scenes", 0)) if availability_row else None
        )
        if tiles is not None:
            candidate_tiles = tiles
            low = high = len(tiles)
            certainty = "exact_from_manifest"
        elif cells and scope == "all":
            candidate_tiles = [
                {**cell, "tile_id": _tile_id(mgrs, cell, side), "parent": mgrs}
                for cell in cells
            ]
            low = high = len(candidate_tiles)
            certainty = "exact_from_plan"
        elif cells:
            candidate_tiles = [
                {**cell, "tile_id": _tile_id(mgrs, cell, side), "parent": mgrs}
                for cell in cells
            ]
            low, high = 0, len(cells)
            certainty = "range_identity_changed_manifest_absent"
        else:
            candidate_tiles = []
            if availability_scenes == 0:
                low = high = 0
                certainty = "no_scenes_in_availability_artifact"
            else:
                low, high = args.unplanned_cells_low, args.unplanned_cells_high
                certainty = "range_plan_absent"

        training_done = cog_done = 0
        for tile in candidate_tiles:
            tile_id = _tile_id(mgrs, tile, side)
            parent = str(tile.get("parent") or mgrs)
            metrics, cog = _output_status(parent, tile_id, run_prefix)
            training_done += int(metrics)
            cog_done += int(cog)
        # An existing training output proves that cell belongs to the effective
        # production set even when the identity-changed manifest is absent.
        low = max(low, training_done)
        remaining_low = max(0, low - training_done)
        remaining_high = max(0, high - training_done)
        target_low += low
        target_high += high
        complete_training += training_done
        complete_cogs += cog_done
        sr_pixels_per_cell = (args.sr_scale * side) ** 2
        granule_cog_storage = _range(
            low * sr_pixels_per_cell * args.cog_bytes_low / 2**30,
            high * sr_pixels_per_cell * args.cog_bytes_high / 2**30,
        )
        cog_storage_low += float(granule_cog_storage["low"])
        cog_storage_high += float(granule_cog_storage["high"])

        bbox = row.get("bbox") or (plan or {}).get("bbox_wgs84")
        if bbox is None and isinstance(bbox_payload, dict):
            bbox = bbox_payload.get(mgrs)
        if bbox is None and availability_row:
            bbox = availability_row.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            selected_bboxes.append(bbox)

        state_stage = ((state or {}).get("stages") or {}).get(f"{mgrs}:production", {})
        granule_results.append(
            {
                "mgrs": mgrs,
                "plan": {
                    "status": "present" if plan else "absent",
                    "path": str(plan_path),
                    "planned_cells": len(cells) + no_pass if plan else None,
                    "runnable_cells": len(cells) if plan else None,
                    "no_pass_cells": no_pass if plan else None,
                },
                "availability_scenes": availability_scenes,
                "production_scope": scope,
                "production_manifest": str(manifest_source) if manifest_source else None,
                "training_target_cells": {**_range(low, high), "basis": certainty},
                "already_complete_training_cells": training_done,
                "already_complete_cog_cells": cog_done,
                "remaining_training_cells": _range(remaining_low, remaining_high),
                "estimated_cog_storage_gib": granule_cog_storage,
                "bbox_wgs84": bbox,
                "state_production_status": state_stage.get("status"),
            }
        )

    remaining_low = max(0, target_low - complete_training)
    remaining_high = max(0, target_high - complete_training)
    gpu_hours = _range(
        remaining_low * gpu_hours_low,
        remaining_high * gpu_hours_high,
    )
    wall_hours = _range(
        gpu_hours["low"] / args.gpus,
        gpu_hours["high"] / args.gpus,
    )
    cog_storage = _range(cog_storage_low, cog_storage_high)
    union_bbox = _bbox_union(selected_bboxes)
    cross = config.get("cross") or {}
    dst_crs = str(cross.get("dst_crs", "EPSG:3035"))
    resolution = float(cross.get("resolution", 2.5))
    mosaic_pixels, mosaic_method = _projected_pixel_count(union_bbox, dst_crs, resolution)
    mosaic_storage = (
        _range(
            mosaic_pixels * args.mosaic_bytes_low / 2**30,
            mosaic_pixels * args.mosaic_bytes_high / 2**30,
        )
        if mosaic_pixels is not None
        else None
    )

    inventory_normalized = selected
    assumptions = {
        "gpu_hours_per_cell": _range(gpu_hours_low, gpu_hours_high, digits=6),
        "gpu_hours_per_cell_defaults": _range(*DEFAULTS["gpu_hours_per_cell"]),
        "temporary_gib_per_active_cell": _range(args.temp_gib_low, args.temp_gib_high),
        "cog_bytes_per_pixel": _range(args.cog_bytes_low, args.cog_bytes_high),
        "mosaic_bytes_per_pixel": _range(args.mosaic_bytes_low, args.mosaic_bytes_high),
        "sr_scale": args.sr_scale,
        "unplanned_cells_per_granule": _range(
            args.unplanned_cells_low, args.unplanned_cells_high
        ),
        "calibration": gpu_calibration,
        "wall_time_model": "serial GPU-hours divided by GPU count; excludes CPU/I/O stages and contention",
        "completion_rule": "metrics.json means training skips; sr_pred.tif means COG exists",
    }
    return {
        "schema_version": 1,
        "kind": "national_production_cpu_preflight",
        "config": {
            "path": str(config["_config_path"]),
            "file_sha256": national.sha256_file(config["_config_path"]),
            "effective_sha256": national.config_digest(config, selected),
        },
        "inventory": {
            "selected_granules": [row["mgrs"] for row in selected],
            "selected_sha256": _hash_json(inventory_normalized),
            "exceptions": config.get("inventory_exceptions", []),
            "exception_granules": len(config.get("inventory_exceptions", [])),
        },
        "selection": {
            "granules": args.granules,
            "wave_size": args.wave_size,
            "wave_index": args.wave_index,
            "gpus": args.gpus,
        },
        "assumptions": assumptions,
        "artifacts": sorted(
            {item["path"]: item for item in artifacts}.values(),
            key=lambda item: item["path"],
        ),
        "granules": granule_results,
        "totals": {
            "planned_no_pass_cells": no_pass_total,
            "training_target_cells": _range(target_low, target_high),
            "already_complete_training_cells": complete_training,
            "already_complete_cog_cells": complete_cogs,
            "remaining_training_cells": _range(remaining_low, remaining_high),
            "serial_gpu_hours": gpu_hours,
            "wall_hours_for_gpus": {**wall_hours, "gpus": args.gpus},
            "temporary_active_storage_gib": _range(
                min(args.gpus, remaining_low) * args.temp_gib_low,
                min(args.gpus, remaining_high) * args.temp_gib_high,
            ),
            "per_granule_cog_storage_gib": {
                **cog_storage,
                "meaning": "sum across selected granules; individual ranges are in granules[]",
            },
            "final_mosaic": {
                "bbox_wgs84": union_bbox,
                "dst_crs": dst_crs,
                "resolution": resolution,
                "pixels": mosaic_pixels,
                "storage_gib": mosaic_storage,
                **mosaic_method,
            },
        },
        "warnings": [
            "Ranges are intentionally retained; do not interpret bounds as confidence intervals.",
            "Final mosaic storage estimates the rectangular projected envelope, not compressed content.",
        ],
    }


def _nonnegative(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    gpu_low = (
        DEFAULTS["gpu_hours_per_cell"][0]
        if args.gpu_hours_low is None
        else args.gpu_hours_low
    )
    gpu_high = (
        DEFAULTS["gpu_hours_per_cell"][1]
        if args.gpu_hours_high is None
        else args.gpu_hours_high
    )
    pairs = [
        ("GPU-hours", gpu_low, gpu_high),
        ("temporary GiB", args.temp_gib_low, args.temp_gib_high),
        ("COG bytes/pixel", args.cog_bytes_low, args.cog_bytes_high),
        ("mosaic bytes/pixel", args.mosaic_bytes_low, args.mosaic_bytes_high),
        ("unplanned cells", args.unplanned_cells_low, args.unplanned_cells_high),
    ]
    for label, low, high in pairs:
        if low < 0 or high < low:
            parser.error(f"{label} bounds require 0 <= low <= high")
    if args.gpus < 1 or args.sr_scale < 1:
        parser.error("--gpus and --sr-scale must be positive")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--granules")
    parser.add_argument("--wave-size", type=int, default=0)
    parser.add_argument("--wave-index", type=int, default=0)
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument(
        "--gpu-hours-low",
        type=float,
        default=None,
        help="Override lower GPU-hours/cell bound (measured default: 0.018).",
    )
    parser.add_argument(
        "--gpu-hours-high",
        type=float,
        default=None,
        help="Override upper GPU-hours/cell bound (measured default: 0.023).",
    )
    parser.add_argument(
        "--auto-calibrate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Calibrate from compatible production summary tile times when available.",
    )
    parser.add_argument("--temp-gib-low", type=float, default=DEFAULTS["temporary_gib_per_active_cell"][0])
    parser.add_argument("--temp-gib-high", type=float, default=DEFAULTS["temporary_gib_per_active_cell"][1])
    parser.add_argument("--cog-bytes-low", type=float, default=DEFAULTS["cog_bytes_per_pixel"][0])
    parser.add_argument("--cog-bytes-high", type=float, default=DEFAULTS["cog_bytes_per_pixel"][1])
    parser.add_argument("--mosaic-bytes-low", type=float, default=DEFAULTS["mosaic_bytes_per_pixel"][0])
    parser.add_argument("--mosaic-bytes-high", type=float, default=DEFAULTS["mosaic_bytes_per_pixel"][1])
    parser.add_argument("--sr-scale", type=int, default=DEFAULTS["sr_scale"])
    parser.add_argument("--unplanned-cells-low", type=int, default=DEFAULTS["unplanned_cells_per_granule"][0])
    parser.add_argument("--unplanned-cells-high", type=int, default=DEFAULTS["unplanned_cells_per_granule"][1])
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _nonnegative(parser, args)
    try:
        payload = estimate(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    national.atomic_write_json(args.out, payload)
    print(f"Resource estimate -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
