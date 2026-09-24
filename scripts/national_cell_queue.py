#!/usr/bin/env python3
"""Per-LR512 national fetch queue: closest-to-center dates, cap 16, keep thin stacks.

Observation depth (n_frames, date span, snow/cloud on *used* days) is metadata,
not a certainty / trust score.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds

ROOT = Path(__file__).resolve().parent.parent

NATIONAL_CENTER = date(2025, 7, 15)
DEFAULT_MAX_FRAMES = 16
JUL_PM45 = ("2025-05-31", "2025-08-29")
UINT16_NODATA = 65535


def _yyyy_mm_dd(value: str) -> str:
    return datetime.fromisoformat(str(value)[:10]).strftime("%Y-%m-%d")


def parse_center(value: str | date | None) -> date:
    if value is None:
        return NATIONAL_CENTER
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    return date.fromisoformat(_yyyy_mm_dd(str(value)))


def cell_key(iy: int, ix: int) -> str:
    return f"{int(iy):02d}_{int(ix):02d}"


def rank_day_indices(
    days: list[str],
    passed: np.ndarray,
    *,
    center: date | str | None = None,
    max_frames: int = DEFAULT_MAX_FRAMES,
) -> list[int]:
    """Closest-to-center first among SCL-pass days; keep thin stacks (no min)."""
    center_d = parse_center(center)
    idxs = [i for i, ok in enumerate(passed.tolist()) if bool(ok)]
    idxs.sort(
        key=lambda i: (
            abs((date.fromisoformat(days[i]) - center_d).days),
            days[i],
        )
    )
    cap = int(max_frames)
    if cap <= 0:
        return idxs
    return idxs[:cap]


def date_span_days(days: list[str]) -> int:
    if not days:
        return 0
    parsed = [date.fromisoformat(d) for d in days]
    return int((max(parsed) - min(parsed)).days)


def used_mean(values: np.ndarray, idxs: list[int]) -> float | None:
    if not idxs:
        return None
    picked = np.asarray(values, dtype=np.float32)[idxs]
    finite = picked[np.isfinite(picked)]
    if finite.size == 0:
        return None
    return float(finite.mean())


def land_cell_mask(
    transform,
    crs,
    *,
    side: int,
    n_y: int,
    n_x: int,
    land_paths,
    min_land_frac: float = 0.0,
    sample_n: int = 9,
) -> np.ndarray:
    from scripts.land_mask_lr512 import land_hit_frac, window_lonlat_samples

    out = np.zeros((n_y, n_x), dtype=bool)
    if land_paths is None:
        out[:] = True
        return out
    for iy in range(int(n_y)):
        for ix in range(int(n_x)):
            samples = window_lonlat_samples(
                transform,
                crs,
                row0=iy * side,
                col0=ix * side,
                side=side,
                n=sample_n,
            )
            frac = land_hit_frac(land_paths, samples)
            out[iy, ix] = (
                frac > 0.0 if float(min_land_frac) <= 0.0 else frac >= float(min_land_frac)
            )
    return out


def build_cell_plan(
    days: list[str],
    passed: np.ndarray,
    cloud: np.ndarray,
    snow: np.ndarray,
    land: np.ndarray,
    *,
    center: date | str | None = None,
    max_frames: int = DEFAULT_MAX_FRAMES,
    side: int = 512,
) -> dict:
    """Select per-cell dates and QA stats. ``passed/cloud/snow`` are (n_days, n_y, n_x)."""
    center_d = parse_center(center)
    n_days, n_y, n_x = passed.shape
    if len(days) != n_days:
        raise ValueError(f"days ({len(days)}) != passed n_days ({n_days})")
    if land.shape != (n_y, n_x):
        raise ValueError("land mask shape must match the LR grid")

    n_frames = np.full((n_y, n_x), UINT16_NODATA, dtype=np.uint16)
    span = np.full((n_y, n_x), UINT16_NODATA, dtype=np.uint16)
    mean_snow = np.full((n_y, n_x), np.nan, dtype=np.float32)
    mean_cloud = np.full((n_y, n_x), np.nan, dtype=np.float32)

    cells: list[dict] = []
    skipped: list[dict] = []
    union: set[str] = set()
    n_land = int(land.sum())

    for iy in range(n_y):
        for ix in range(n_x):
            if not bool(land[iy, ix]):
                continue
            idxs = rank_day_indices(
                days,
                passed[:, iy, ix],
                center=center_d,
                max_frames=max_frames,
            )
            selected = sorted(days[i] for i in idxs)
            rec = {
                "iy": int(iy),
                "ix": int(ix),
                "key": cell_key(iy, ix),
                "row_off": int(iy * side),
                "col_off": int(ix * side),
                "dates": selected,
                "n_frames": len(selected),
                "date_span_days": date_span_days(selected),
                "mean_snow_used": used_mean(snow[:, iy, ix], idxs),
                "mean_cloud_used": used_mean(cloud[:, iy, ix], idxs),
            }
            n_frames[iy, ix] = np.uint16(rec["n_frames"])
            # QA GeoTIFFs are uint16; span is max-min days and should be >= 0.
            span[iy, ix] = np.uint16(max(0, int(rec["date_span_days"])))
            if rec["mean_snow_used"] is not None:
                mean_snow[iy, ix] = rec["mean_snow_used"]
            if rec["mean_cloud_used"] is not None:
                mean_cloud[iy, ix] = rec["mean_cloud_used"]
            if selected:
                cells.append(rec)
                union.update(selected)
            else:
                skipped.append({"iy": int(iy), "ix": int(ix), "reason": "no_scl_pass_day"})

    union_dates = sorted(union)
    n_vals = n_frames[n_frames != UINT16_NODATA]
    return {
        "center_date": center_d.isoformat(),
        "max_frames": int(max_frames),
        "side": int(side),
        "grid": {"n_y": int(n_y), "n_x": int(n_x)},
        "n_land_cells": n_land,
        "n_cells_with_frames": len(cells),
        "n_cells_no_pass": len(skipped),
        "union_dates": union_dates,
        "n_union_dates": len(union_dates),
        "n_frames_min": None if n_vals.size == 0 else int(n_vals.min()),
        "n_frames_max": None if n_vals.size == 0 else int(n_vals.max()),
        "n_frames_mean_land": None if n_vals.size == 0 else float(n_vals.mean()),
        "cells": cells,
        "skipped_no_pass": skipped,
        "qa": {
            "n_frames": n_frames,
            "date_span_days": span,
            "mean_snow_used": mean_snow,
            "mean_cloud_used": mean_cloud,
        },
    }


def cell_lookup(plan: dict) -> dict[tuple[int, int], dict]:
    return {(int(c["iy"]), int(c["ix"])): c for c in plan.get("cells") or []}


def frame_day(frame: dict) -> str:
    dt = frame.get("datetime")
    if dt:
        return _yyyy_mm_dd(str(dt))
    path = str(frame.get("path") or "")
    digits = "".join(ch for ch in Path(path).stem if ch.isdigit())
    if len(digits) >= 8:
        # stems like 003_20250715
        chunk = digits[-8:]
        return f"{chunk[:4]}-{chunk[4:6]}-{chunk[6:8]}"
    raise ValueError(f"frame has no parseable datetime: {frame!r}")


def filter_frames_for_cell(frames: list[dict], dates: list[str] | set[str]) -> list[dict]:
    wanted = {_yyyy_mm_dd(d) for d in dates}
    kept = []
    for fr in frames:
        try:
            day = frame_day(fr)
        except ValueError:
            continue
        if day in wanted:
            kept.append(fr)
    return kept


def load_dates_file(path: Path) -> list[str]:
    raw = json.loads(Path(path).read_text())
    if isinstance(raw, list):
        dates = raw
    elif isinstance(raw, dict):
        dates = raw.get("union_dates") or raw.get("dates") or []
    else:
        raise ValueError(f"dates file must be a list or object: {path}")
    return sorted({_yyyy_mm_dd(d) for d in dates})


def load_plan_stac_ids(path: Path) -> dict[str, str]:
    """``{YYYY-MM-DD: stac_id}`` from ``days_scored_items``. Empty if absent."""
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for rec in raw.get("days_scored_items") or []:
        sid = rec.get("stac_id")
        day = rec.get("date")
        if sid and day:
            out[_yyyy_mm_dd(str(day))] = str(sid)
    return out


def _item_day(item) -> str | None:
    dt = getattr(item, "datetime", None)
    if dt is None:
        return None
    if hasattr(dt, "strftime"):
        try:
            return dt.astimezone(datetime.timezone.utc).strftime("%Y-%m-%d")
        except Exception:
            return dt.strftime("%Y-%m-%d")
    return _yyyy_mm_dd(str(dt))


def filter_stac_items_for_plan(
    items: list,
    wanted_dates: set[str],
    wanted_ids: dict[str, str] | None = None,
) -> tuple[list, list[tuple[str, str]]]:
    """Pin planned STAC ids; otherwise keep the lowest ``eo:cloud_cover`` per day.

    Returns ``(selected_items, missing_id_pairs)``. Missing ids are planned
    ``(date, stac_id)`` pairs not present in ``items`` (caller may catalog-get).
    """
    wanted = {_yyyy_mm_dd(d) for d in wanted_dates}
    wanted_ids = { _yyyy_mm_dd(d): str(s) for d, s in (wanted_ids or {}).items() }
    by_id = {it.id: it for it in items}
    best: dict[str, object] = {}
    for it in items:
        day = _item_day(it)
        if day is None or day not in wanted:
            continue
        props = getattr(it, "properties", None) or {}
        cc = float(props.get("eo:cloud_cover", 100.0))
        prev = best.get(day)
        prev_cc = 100.0
        if prev is not None:
            prev_cc = float((getattr(prev, "properties", None) or {}).get("eo:cloud_cover", 100.0))
        if prev is None or cc < prev_cc:
            best[day] = it
    selected = []
    missing: list[tuple[str, str]] = []
    for day in sorted(wanted):
        sid = wanted_ids.get(day)
        if sid:
            if sid in by_id:
                selected.append(by_id[sid])
            else:
                missing.append((day, sid))
        elif day in best:
            selected.append(best[day])
    return selected, missing


def bbox_wgs84_from_profile(profile: dict) -> list[float]:
    transform = profile["transform"]
    crs = profile["crs"]
    height = int(profile["height"])
    width = int(profile["width"])
    west, south, east, north = rasterio.transform.array_bounds(height, width, transform)
    left, bottom, right, top = transform_bounds(
        crs, "EPSG:4326", west, south, east, north, densify_pts=21
    )
    return [float(left), float(bottom), float(right), float(top)]


def write_uint16_geotiff(
    counts: np.ndarray,
    profile: dict,
    side: int,
    dest: Path,
    *,
    description: str,
) -> None:
    n_y, n_x = counts.shape
    res = float(profile["resolution_m"]) * side
    transform = profile["transform"]
    out_transform = from_origin(transform.c, transform.f, res, res)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        dest,
        "w",
        driver="GTiff",
        height=n_y,
        width=n_x,
        count=1,
        dtype="uint16",
        crs=profile["crs"],
        transform=out_transform,
        compress="deflate",
        nodata=UINT16_NODATA,
    ) as dst:
        dst.write(counts.astype(np.uint16), 1)
        dst.set_band_description(1, description)


def write_float_geotiff(
    values: np.ndarray,
    profile: dict,
    side: int,
    dest: Path,
    *,
    description: str,
) -> None:
    n_y, n_x = values.shape
    res = float(profile["resolution_m"]) * side
    transform = profile["transform"]
    out_transform = from_origin(transform.c, transform.f, res, res)
    dest.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(values, dtype=np.float32)
    with rasterio.open(
        dest,
        "w",
        driver="GTiff",
        height=n_y,
        width=n_x,
        count=1,
        dtype="float32",
        crs=profile["crs"],
        transform=out_transform,
        compress="deflate",
        nodata=np.nan,
    ) as dst:
        dst.write(arr, 1)
        dst.set_band_description(1, description)


def write_plan_artifacts(plan: dict, profile: dict, out_dir: Path, mgrs: str) -> dict:
    """Write plan.json (no numpy) plus QA GeoTIFFs. Returns JSON-ready plan."""
    out_dir.mkdir(parents=True, exist_ok=True)
    qa = plan["qa"]
    side = int(plan["side"])
    prefix = out_dir / f"{mgrs}_lr{side}"
    n_frames_tif = Path(f"{prefix}_n_frames.tif")
    span_tif = Path(f"{prefix}_date_span_days.tif")
    snow_tif = Path(f"{prefix}_mean_snow_used.tif")
    cloud_tif = Path(f"{prefix}_mean_cloud_used.tif")
    write_uint16_geotiff(
        qa["n_frames"], profile, side, n_frames_tif, description="n_frames (observation depth)"
    )
    write_uint16_geotiff(
        qa["date_span_days"],
        profile,
        side,
        span_tif,
        description="date_span_days of used frames",
    )
    write_float_geotiff(
        qa["mean_snow_used"],
        profile,
        side,
        snow_tif,
        description="mean SCL snow frac on used days",
    )
    write_float_geotiff(
        qa["mean_cloud_used"],
        profile,
        side,
        cloud_tif,
        description="mean SCL cloud frac on used days",
    )
    json_plan = {k: v for k, v in plan.items() if k != "qa"}
    json_plan["mgrs_tile"] = mgrs
    json_plan["qa_rasters"] = {
        "n_frames": str(n_frames_tif),
        "date_span_days": str(span_tif),
        "mean_snow_used": str(snow_tif),
        "mean_cloud_used": str(cloud_tif),
        "note": (
            "Observation-depth metadata, not a certainty or 2.5 m trust score. "
            "n_frames is a weak proxy."
        ),
    }
    plan_path = out_dir / "plan.json"
    plan_path.write_text(json.dumps(json_plan, indent=2) + "\n")
    json_plan["plan_path"] = str(plan_path)
    return json_plan
