#!/usr/bin/env python3
"""Build deterministic, geographically coherent national production waves."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_national_production import load_country_config  # noqa: E402


def _resolve(path: str | Path, *, relative_to: Path = ROOT) -> Path:
    value = Path(path)
    return value if value.is_absolute() else relative_to / value


def _read_ids(values: list[str], files: list[Path]) -> list[str]:
    result: list[str] = []
    for value in values:
        result.extend(part for part in value.split(",") if part.strip())
    for path in files:
        payload = json.loads(path.read_text())
        if isinstance(payload, dict):
            payload = payload.get("mgrs", payload.get("granules"))
        if not isinstance(payload, list):
            raise ValueError(f"{path}: expected a JSON list or object with mgrs/granules")
        result.extend(
            item if isinstance(item, str) else str(item.get("mgrs", "")) for item in payload
        )
    normalized = [str(item).strip().upper().lstrip("T") for item in result]
    if any(not item for item in normalized):
        raise ValueError("MGRS selections must not contain empty values")
    if len(normalized) != len(set(normalized)):
        raise ValueError("MGRS selections must not contain duplicates")
    return normalized


def load_bboxes(config: dict) -> dict[str, tuple[float, float, float, float]]:
    inventory = config.get("inventory")
    if not isinstance(inventory, dict) or not inventory.get("mgrs_bboxes_json"):
        raise ValueError("country config inventory.mgrs_bboxes_json is required")
    path = _resolve(inventory["mgrs_bboxes_json"])
    payload = json.loads(path.read_text())
    if isinstance(payload, dict) and "bboxes" in payload:
        payload = payload["bboxes"]
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected an MGRS-to-bbox object")
    result: dict[str, tuple[float, float, float, float]] = {}
    for raw_id, raw_bbox in payload.items():
        mgrs = str(raw_id).upper().lstrip("T")
        if (
            not isinstance(raw_bbox, list)
            or len(raw_bbox) != 4
            or not all(isinstance(value, (int, float)) for value in raw_bbox)
        ):
            raise ValueError(f"{path}: invalid bbox for {mgrs}")
        west, south, east, north = map(float, raw_bbox)
        if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
            raise ValueError(f"{path}: invalid WGS84 bbox for {mgrs}")
        result[mgrs] = (west, south, east, north)
    return result


def _center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def _bbox_distance_km(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    """Approximate edge-to-edge distance between WGS84 bboxes."""
    lon_gap = max(0.0, left[0] - right[2], right[0] - left[2])
    lat_gap = max(0.0, left[1] - right[3], right[1] - left[3])
    latitude = sum((_center(left)[1], _center(right)[1])) / 2
    return math.hypot(lon_gap * 111.32 * math.cos(math.radians(latitude)), lat_gap * 110.57)


def balanced_wave_sizes(total: int, minimum: int, maximum: int) -> list[int]:
    if minimum < 1 or maximum < minimum:
        raise ValueError("wave sizes must satisfy 1 <= minimum <= maximum")
    if total < 1:
        return []
    if total < minimum:
        return [total]
    least_waves = math.ceil(total / maximum)
    most_waves = total // minimum
    if least_waves > most_waves:
        raise ValueError(f"cannot partition {total} granules into waves of {minimum}-{maximum}")
    target = (minimum + maximum) / 2
    wave_count = min(
        range(least_waves, most_waves + 1),
        key=lambda count: (abs(total / count - target), count),
    )
    base, extra = divmod(total, wave_count)
    return [base + (index < extra) for index in range(wave_count)]


def geographic_waves(
    mgrs_ids: list[str],
    bboxes: dict[str, tuple[float, float, float, float]],
    *,
    minimum: int = 6,
    maximum: int = 9,
) -> list[list[str]]:
    missing = sorted(set(mgrs_ids) - set(bboxes))
    if missing:
        raise ValueError(f"missing country MGRS bboxes for runnable granules: {missing}")
    sizes = balanced_wave_sizes(len(mgrs_ids), minimum, maximum)
    remaining = set(mgrs_ids)
    waves: list[list[str]] = []
    for size in sizes:
        # South-to-north seeds make the national progression reproducible while
        # bbox distances, rather than MGRS spelling, determine each wave.
        seed = min(
            remaining,
            key=lambda item: (_center(bboxes[item])[1], _center(bboxes[item])[0], item),
        )
        wave = [seed]
        remaining.remove(seed)
        while len(wave) < size:
            candidate = min(
                remaining,
                key=lambda item: (
                    min(_bbox_distance_km(bboxes[item], bboxes[member]) for member in wave),
                    min(
                        math.hypot(
                            _center(bboxes[item])[0] - _center(bboxes[member])[0],
                            _center(bboxes[item])[1] - _center(bboxes[member])[1],
                        )
                        for member in wave
                    ),
                    _center(bboxes[item])[1],
                    _center(bboxes[item])[0],
                    item,
                ),
            )
            wave.append(candidate)
            remaining.remove(candidate)
        waves.append(wave)
    if remaining:
        raise AssertionError(f"internal accounting error: unassigned granules {remaining}")
    return waves


def build_wave_plan(
    config_path: Path,
    *,
    completed: list[str],
    excluded: list[str],
    minimum: int = 6,
    maximum: int = 9,
) -> dict:
    config = load_country_config(config_path)
    inventory = [row["mgrs"] for row in config["_granules"]]
    inventory_set = set(inventory)
    completed_set, excluded_set = set(completed), set(excluded)
    overlap = sorted(completed_set & excluded_set)
    if overlap:
        raise ValueError(f"completed and excluded selections overlap: {overlap}")
    unknown_completed = sorted(completed_set - inventory_set)
    unknown_excluded = sorted(excluded_set - inventory_set)
    if unknown_completed:
        raise ValueError(
            f"completed granules are not runnable inventory entries: {unknown_completed}"
        )
    if unknown_excluded:
        raise ValueError(
            f"excluded granules are not runnable inventory entries: {unknown_excluded}"
        )

    scheduled = sorted(inventory_set - completed_set - excluded_set)
    bboxes = load_bboxes(config)
    waves = geographic_waves(scheduled, bboxes, minimum=minimum, maximum=maximum)
    scheduled_wave_order = [mgrs for wave in waves for mgrs in wave]
    final_granules = [mgrs for mgrs in inventory if mgrs not in excluded_set]
    exceptions = list(config.get("inventory_exceptions") or [])
    accounting = {
        "configured_runnable": len(inventory),
        "completed": len(completed),
        "explicitly_excluded_runnable": len(excluded),
        "scheduled": len(scheduled),
        "inventory_exceptions": len(exceptions),
        "final_mosaic_granules": len(final_granules),
        "source_total": len(inventory) + len(exceptions),
    }
    if (
        accounting["completed"]
        + accounting["explicitly_excluded_runnable"]
        + accounting["scheduled"]
        != accounting["configured_runnable"]
    ):
        raise AssertionError("runnable inventory accounting is incomplete")
    expected_source = (config.get("inventory") or {}).get("expected_source_count")
    if expected_source is not None and accounting["source_total"] != int(expected_source):
        raise ValueError(
            f"source accounting {accounting['source_total']} != expected_source_count "
            f"{expected_source}"
        )

    plan = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "country": config["country"],
        "country_config": str(config["_config_path"]),
        "wave_size_policy": {"minimum": minimum, "maximum": maximum},
        "inventory": inventory,
        "completed_granules": sorted(completed_set),
        "excluded_granules": sorted(excluded_set),
        "inventory_exceptions": exceptions,
        "scheduled_granules": scheduled_wave_order,
        "final_mosaic_granules": final_granules,
        "waves": [
            {"index": index, "granules": granules, "n_granules": len(granules)}
            for index, granules in enumerate(waves)
        ],
        "accounting": accounting,
        "spatial_method": "south_seed_nearest_bbox_edge_v1",
    }
    canonical = json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
    plan["plan_sha256"] = hashlib.sha256(canonical).hexdigest()
    return plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--completed", action="append", default=[], help="Comma-separated MGRS IDs")
    parser.add_argument("--completed-file", action="append", type=Path, default=[])
    parser.add_argument("--exclude", action="append", default=[], help="Comma-separated MGRS IDs")
    parser.add_argument("--exclude-file", action="append", type=Path, default=[])
    parser.add_argument("--min-wave-size", type=int, default=6)
    parser.add_argument("--max-wave-size", type=int, default=9)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        completed = _read_ids(args.completed, args.completed_file)
        excluded = _read_ids(args.exclude, args.exclude_file)
        plan = build_wave_plan(
            args.config,
            completed=completed,
            excluded=excluded,
            minimum=args.min_wave_size,
            maximum=args.max_wave_size,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc
    out = _resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(
        f"Wrote {len(plan['waves'])} waves for {plan['accounting']['scheduled']} "
        f"scheduled granules -> {out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
