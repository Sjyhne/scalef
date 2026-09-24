#!/usr/bin/env python3
"""Durable, country-configurable national production orchestration.

The driver only composes existing production CLIs.  A dry run records every
command without launching STAC access, tiling, training, or mosaicking.

Minimal config::

  {
    "country": "norway",
    "inventory": ["32VNM", "32VNN"]
  }

Paths default to the existing ``national_2025`` layout.  ``inventory`` may
instead be a JSON filename or be supplied with ``--inventory``.  Relative
paths are resolved from the repository root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.country_config import (  # noqa: E402
    validate_country_config as validate_country_schema_config,
)

STAGES = (
    "plan",
    "fetch",
    "tile",
    "identity",
    "cross_identity",
    "production",
    "delivery",
    "cross",
)
PER_GRANULE_STAGES = ("plan", "fetch", "tile", "identity", "production", "delivery")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_fingerprint(path: Path) -> dict:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def atomic_write_json(path: Path, payload: dict) -> None:
    """Replace a JSON state file atomically and durably."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    try:
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError:
        # Some network filesystems do not support directory fsync.  The file
        # itself was still fsynced before the atomic replacement.
        pass


def _json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _resolve(value: str | Path, default: Path) -> Path:
    path = Path(value) if value is not None else default
    return path if path.is_absolute() else ROOT / path


def _load_inventory(value, *, config_path: Path) -> list:
    if isinstance(value, str):
        path = Path(value)
        if not path.is_absolute():
            config_relative = config_path.parent / path
            path = config_relative if config_relative.is_file() else ROOT / path
        value = json.loads(path.read_text())
    if isinstance(value, dict):
        value = value.get("granules", value.get("inventory", value.get("mgrs")))
    if not isinstance(value, list) or not value:
        raise ValueError("inventory must be a non-empty JSON list or {granules: [...]}")
    return value


def load_country_config(config_path: Path, inventory_path: Path | None = None) -> dict:
    config_path = config_path.resolve()
    config = _json(config_path)
    if config.get("schema_version") is not None or isinstance(config.get("country"), dict):
        validate_country_schema_config(config)
    inventory_config = config.get("inventory")
    if inventory_path is not None:
        inventory_value = str(inventory_path.resolve())
    elif isinstance(inventory_config, dict) and inventory_config.get("mgrs_list_json"):
        inventory_value = inventory_config["mgrs_list_json"]
    else:
        inventory_value = inventory_config
    rows = _load_inventory(inventory_value, config_path=config_path)
    granules = []
    seen = set()
    for raw in rows:
        row = {"mgrs": raw} if isinstance(raw, str) else dict(raw)
        mgrs = str(row.get("mgrs") or row.get("mgrs_tile") or "").upper().lstrip("T")
        if len(mgrs) != 5 or not mgrs.isalnum():
            raise ValueError(f"invalid MGRS inventory entry: {raw!r}")
        if mgrs in seen:
            raise ValueError(f"duplicate MGRS inventory entry: {mgrs}")
        seen.add(mgrs)
        row["mgrs"] = mgrs
        granules.append(row)
    if isinstance(inventory_config, dict):
        expected_count = inventory_config.get("expected_runnable_count")
        if expected_count is not None and len(granules) != int(expected_count):
            raise ValueError(
                f"inventory has {len(granules)} runnable MGRS tiles; "
                f"configured expected_runnable_count is {expected_count}"
            )
    exception_ids = {
        str(row.get("mgrs", "")).upper().lstrip("T")
        for row in config.get("inventory_exceptions", [])
        if isinstance(row, dict)
    }
    overlap = sorted(seen & exception_ids)
    if overlap:
        raise ValueError(
            f"inventory exceptions must not also be runnable inventory entries: {overlap}"
        )
    if isinstance(inventory_config, dict):
        expected_source_count = inventory_config.get("expected_source_count")
        if expected_source_count is not None and (
            len(granules) + len(exception_ids) != int(expected_source_count)
        ):
            raise ValueError(
                "runnable inventory plus inventory exceptions does not match "
                f"expected_source_count {expected_source_count}"
            )

    roots = config.get("roots") or {}
    configured_paths = config.get("paths") or {}
    production_hint = _resolve(
        roots.get("production"), ROOT / "production" / "national_2025"
    )
    plan_root = _resolve(
        configured_paths.get("plan_root", roots.get("plans")),
        production_hint / "plans",
    )
    s2_root = _resolve(
        configured_paths.get("s2_root", roots.get("data")),
        ROOT / "data" / "s2_revisits" / "national_2025",
    )
    production_root = _resolve(roots.get("production"), plan_root.parent)
    config["_paths"] = {
        "production": production_root,
        "plans": plan_root,
        "data": s2_root,
        "tiles": _resolve(roots.get("tiles"), s2_root.parent),
        "mosaics": _resolve(roots.get("mosaics"), production_root / "mosaics"),
        "results": _resolve(
            roots.get("results"), ROOT / "single_samples" / "sweep_results"
        ),
    }
    config["_granules"] = granules
    config["_config_path"] = config_path
    config.setdefault("country", "norway")
    country = config["country"]
    config["_country_slug"] = (
        str(country.get("name") or country.get("iso_a2")).strip().lower().replace(" ", "_")
        if isinstance(country, dict)
        else str(country).strip().lower().replace(" ", "_")
    )
    config.setdefault("recipe", {})
    inputs = config.get("inputs") or {}
    if inputs.get("land_outline_geojson") and "land_mask" not in config["recipe"]:
        config["recipe"]["land_mask"] = inputs["land_outline_geojson"]
    config.setdefault("identity", {})
    config.setdefault("production", {})
    config.setdefault("delivery", {})
    config.setdefault("cross_identity", {})
    return config


def config_digest(config: dict, granules: list[dict] | None = None) -> str:
    public = {key: value for key, value in config.items() if not key.startswith("_")}
    public["inventory_normalized"] = granules if granules is not None else config["_granules"]
    public["resolved_paths"] = {k: str(v) for k, v in config["_paths"].items()}
    encoded = json.dumps(public, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class Stage:
    key: str
    name: str
    command: list[str]
    expected: tuple[Path, ...]
    validate: Callable[[], None]
    mgrs: str | None = None
    prepare: Callable[[], None] | None = None
    finalize: Callable[[], None] | None = None


def _script(name: str) -> str:
    return str(ROOT / "scripts" / name)


def _recipe(config: dict, row: dict, key: str, default):
    return row.get(key, config["recipe"].get(key, default))


def paths_for(config: dict, row: dict) -> dict[str, Path]:
    mgrs = row["mgrs"]
    roots = config["_paths"]
    side = int(_recipe(config, row, "side", 512))
    overlap_frac = float(_recipe(config, row, "overlap_frac", 0.0))
    overlap_suffix = (
        f"_ovl{int(round(overlap_frac * 100))}" if overlap_frac > 0 else ""
    )
    plan_dir = _resolve(row.get("plan_dir"), roots["plans"] / mgrs)
    s2_dir = _resolve(row.get("s2_dir"), roots["data"] / mgrs)
    manifest = _resolve(
        row.get("manifest"),
        s2_dir / f"granule_tiles_lr{side}{overlap_suffix}_manifest.json",
    )
    identity_dir = _resolve(
        row.get("identity_dir"),
        roots["production"] / "identity" / f"{mgrs}{overlap_suffix}",
    )
    return {
        "plan_dir": plan_dir,
        "plan": plan_dir / "plan.json",
        "ondisk": plan_dir / "plan_ondisk.json",
        "s2": s2_dir,
        "meta": s2_dir / "meta.json",
        "manifest": manifest,
        "identity_dir": identity_dir,
        "identity": identity_dir / "identity_plan.json",
        "identity_manifest": identity_dir / "manifest_icm_all.json",
        "identity_changed_manifest": identity_dir / "manifest_icm_changed.json",
        "summary": _resolve(
            row.get("production_summary"),
            roots["results"] / f"production_{mgrs}_lr{side}{overlap_suffix}.json",
        ),
        "mosaic": _resolve(
            row.get("delivery_mosaic"),
            roots["mosaics"] / f"{mgrs}{overlap_suffix}_sr_2p5m.tif",
        ),
    }


def _cross_identity_layout(config: dict, granules: list[dict]) -> dict[str, Path]:
    settings = config.get("cross_identity") or {}
    cache_key = json.dumps(
        {
            "mgrs": sorted(row["mgrs"] for row in granules),
            "settings": settings,
        },
        sort_keys=True,
    )
    cache = config.setdefault("_cross_identity_layout_cache", {})
    if cache_key in cache:
        return cache[cache_key]
    signature = [
        {
            "mgrs": row["mgrs"],
            "identity": str(paths_for(config, row)["identity"]),
            "identity_sha256": (
                sha256_file(paths_for(config, row)["identity"])
                if paths_for(config, row)["identity"].is_file()
                else None
            ),
            "manifest": str(paths_for(config, row)["manifest"]),
            "manifest_sha256": (
                sha256_file(paths_for(config, row)["manifest"])
                if paths_for(config, row)["manifest"].is_file()
                else None
            ),
        }
        for row in sorted(granules, key=lambda item: item["mgrs"])
    ]
    subset_hash = hashlib.sha256(
        json.dumps(signature, sort_keys=True).encode()
    ).hexdigest()[:12]
    base = _resolve(
        settings.get("root"),
        config["_paths"]["production"] / "cross_identity",
    )
    out_dir = _resolve(settings.get("out_dir"), base / f"{subset_hash}_revised")
    layout = {
        "input": _resolve(
            settings.get("inputs_manifest"), base / f"{subset_hash}_inputs.json"
        ),
        "out": out_dir,
        "audit": (
            _resolve(settings.get("report"), base / f"{subset_hash}_audit.json")
            if bool(settings.get("audit_only", False))
            else out_dir / "cross_mgrs_identity_audit.json"
        ),
    }
    cache[cache_key] = layout
    return layout


def effective_identity_paths(
    config: dict, row: dict, granules: list[dict]
) -> dict[str, Path]:
    original = paths_for(config, row)
    settings = config.get("cross_identity") or {}
    if not bool(settings.get("enabled", False)) or bool(
        settings.get("audit_only", False)
    ):
        return {
            "identity": original["identity"],
            "identity_manifest": original["identity_manifest"],
            "identity_changed_manifest": original["identity_changed_manifest"],
        }
    revised_dir = _cross_identity_layout(config, granules)["out"] / row["mgrs"]
    return {
        "identity": revised_dir / "identity_plan.json",
        "identity_manifest": revised_dir / "manifest_icm_all.json",
        "identity_changed_manifest": revised_dir / "manifest_icm_changed.json",
    }


def _append_bool(command: list[str], enabled: bool, flag: str) -> None:
    if enabled:
        command.append(flag)


def _validate_plan(path: Path, mgrs: str) -> None:
    plan = _json(path)
    actual = str(plan.get("mgrs_tile") or mgrs).upper().lstrip("T")
    if actual != mgrs:
        raise ValueError(f"{path}: MGRS {actual} does not match inventory {mgrs}")
    cells = plan.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError(f"{path}: plan has no cells")


def _validate_fetch(meta_path: Path, ondisk_path: Path, mgrs: str) -> None:
    meta = _json(meta_path)
    frames = meta.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"{meta_path}: fetched stack has no frames")
    actual = str(meta.get("mgrs_tile") or mgrs).upper().lstrip("T")
    if actual != mgrs:
        raise ValueError(f"{meta_path}: MGRS {actual} does not match inventory {mgrs}")
    _validate_plan(ondisk_path, mgrs)


def _manifest_tiles(path: Path) -> tuple[dict, list[dict]]:
    manifest = _json(path)
    tiles = manifest.get("tiles")
    if not isinstance(tiles, list) or not tiles:
        raise ValueError(f"{path}: manifest has no tiles")
    ids = [tile.get("tile_id") for tile in tiles]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"{path}: missing or duplicate tile IDs")
    return manifest, tiles


def _validate_tile(manifest_path: Path, ondisk_path: Path) -> None:
    manifest, tiles = _manifest_tiles(manifest_path)
    plan = _json(ondisk_path)
    planned = {
        (int(cell["iy"]), int(cell["ix"]))
        for cell in plan.get("cells") or []
        if int(cell.get("n_frames") or len(cell.get("dates") or [])) > 0
    }
    if float(manifest.get("overlap_frac") or 0) > 0:
        covered = set()
        for tile in tiles:
            if not tile.get("dates") or not tile.get("source_plan_cells"):
                raise ValueError(
                    f"{manifest_path}: overlap tile lacks plan-date provenance"
                )
            covered.update(
                (int(cell["iy"]), int(cell["ix"]))
                for cell in tile["source_plan_cells"]
            )
        if not covered.issubset(planned):
            raise ValueError(f"{manifest_path}: overlap tiles reference unknown plan cells")
        if planned - covered:
            raise ValueError(
                f"{manifest_path}: overlap grid misses planned cells "
                f"{sorted(planned - covered)[:5]}"
            )
        return
    tiled = {(int(tile["iy"]), int(tile["ix"])) for tile in tiles}
    missing, extra = sorted(planned - tiled), sorted(tiled - planned)
    if missing or extra:
        raise ValueError(
            f"{manifest_path}: tile coverage differs from plan "
            f"(missing={missing[:5]}, extra={extra[:5]})"
        )


def _validate_identity(identity_path: Path, manifest_path: Path) -> None:
    plan = _json(identity_path)
    _manifest, tiles = _manifest_tiles(manifest_path)
    expected = {tile["tile_id"] for tile in tiles}
    assignment = plan.get("assignment") or {}
    independent = plan.get("independent") or {}
    assigned = set(assignment)
    if assigned != expected:
        raise ValueError(
            f"{identity_path}: identity coverage differs from manifest "
            f"(missing={sorted(expected - assigned)[:5]}, "
            f"extra={sorted(assigned - expected)[:5]})"
        )
    if set(independent) != expected:
        raise ValueError(f"{identity_path}: independent identity coverage is incomplete")
    if any(not date for date in assignment.values()):
        raise ValueError(f"{identity_path}: identity assignment contains missing dates")
    all_path = identity_path.parent / "manifest_icm_all.json"
    changed_path = identity_path.parent / "manifest_icm_changed.json"
    _all_manifest, all_tiles = _manifest_tiles(all_path)
    all_ids = {tile["tile_id"] for tile in all_tiles}
    if all_ids != expected:
        raise ValueError(f"{all_path}: full ICM manifest coverage is incomplete")
    changed_payload = _json(changed_path)
    changed_tiles = changed_payload.get("tiles")
    if not isinstance(changed_tiles, list):
        raise ValueError(f"{changed_path}: tiles must be a list")
    changed_ids = {tile.get("tile_id") for tile in changed_tiles}
    if None in changed_ids or len(changed_ids) != len(changed_tiles):
        raise ValueError(f"{changed_path}: missing or duplicate tile IDs")
    expected_changed = {
        tile_id
        for tile_id, date in assignment.items()
        if date != independent[tile_id]
    }
    if changed_ids != expected_changed:
        raise ValueError(f"{changed_path}: changed-only ICM coverage is incorrect")


def _same_path(value: str | Path | None, expected: Path) -> bool:
    if value is None:
        return False
    path = Path(value)
    path = path if path.is_absolute() else ROOT / path
    return path.resolve() == expected.resolve()


def _validate_production(
    summary_path: Path,
    manifest_path: Path,
    *,
    identity_path: Path | None = None,
    run_prefix: str | None = None,
) -> None:
    summary = _json(summary_path)
    _manifest, tiles = _manifest_tiles(manifest_path)
    aggregate = summary.get("aggregate") or {}
    expected = len(tiles)
    if not _same_path(summary.get("manifest"), manifest_path):
        raise ValueError(f"{summary_path}: summary manifest provenance is stale")
    if identity_path is not None and not _same_path(
        summary.get("identity_plan"), identity_path
    ):
        raise ValueError(f"{summary_path}: summary identity-plan provenance is stale")
    if run_prefix is not None and summary.get("run_prefix") != run_prefix:
        raise ValueError(f"{summary_path}: summary run prefix is stale")
    if (
        int(aggregate.get("n_fail", -1)) != 0
        or int(aggregate.get("n_ok", -1)) != expected
        or int(aggregate.get("n_with_sr_tif", -1)) != expected
    ):
        raise ValueError(
            f"{summary_path}: incomplete production; expected {expected}, "
            f"aggregate={aggregate}"
        )


def _validate_delivery(
    mosaic: Path,
    manifest_path: Path,
    *,
    ramp_px: int | None = None,
    require_date_cuts: bool = True,
    require_harmonization: bool = False,
    max_harmonized_edge_p95: float | None = None,
    max_same_identity_overlap_p95: float | None = None,
    max_identity_risk_overlap_p95: float | None = None,
    identity_path: Path | None = None,
) -> None:
    sidecar = mosaic.with_suffix(mosaic.suffix + ".json")
    summary = _json(sidecar)
    _manifest, tiles = _manifest_tiles(manifest_path)
    expected = len(tiles)
    if "overlap_qa" in summary:
        if int(summary.get("n_missing", -1)) != 0:
            raise ValueError(f"{sidecar}: overlap delivery reports missing tiles")
        if int((summary.get("mosaic") or {}).get("n_sources", -1)) != expected:
            raise ValueError(f"{sidecar}: overlap delivery source count is stale")
        if summary.get("merge_method") != "feather":
            raise ValueError(f"{sidecar}: overlap delivery is not feathered")
        if require_harmonization and not isinstance(
            summary.get("harmonization"), dict
        ):
            raise ValueError(f"{sidecar}: overlap delivery lacks harmonization")
        if (summary.get("qa") or {}).get("passed") is not True:
            raise ValueError(f"{sidecar}: overlap delivery QA failed")
        overlap_qa = summary["overlap_qa"]
        same = (overlap_qa.get("same_identity") or {}).get("p95")
        risk = (overlap_qa.get("identity_risk") or {}).get("p95")
        if (
            max_same_identity_overlap_p95 is not None
            and same is not None
            and float(same) > max_same_identity_overlap_p95
        ):
            raise ValueError(f"{sidecar}: same-identity overlap p95 failed")
        if (
            max_identity_risk_overlap_p95 is not None
            and risk is not None
            and float(risk) > max_identity_risk_overlap_p95
        ):
            raise ValueError(f"{sidecar}: identity-risk overlap p95 failed")
    elif "n_tiles" in summary:
        if int(summary["n_tiles"]) != expected:
            raise ValueError(f"{sidecar}: delivery tile count does not equal {expected}")
        if ramp_px is not None and int(summary.get("ramp_px", -1)) != ramp_px:
            raise ValueError(f"{sidecar}: ramp_px does not equal configured {ramp_px}")
        if (
            require_date_cuts
            and not require_harmonization
            and summary.get("date_cuts_only") is not True
        ):
            raise ValueError(f"{sidecar}: delivery did not restrict ramps to date cuts")
        if require_harmonization:
            harmony = summary.get("harmonization")
            if not isinstance(harmony, dict):
                raise ValueError(f"{sidecar}: delivery lacks global harmonization")
            if summary.get("ramp_scope") != "identity_risk_edges_after_global_harmonization":
                raise ValueError(
                    f"{sidecar}: harmonized delivery did not preserve identity-risk scope"
                )
            if (summary.get("qa") or {}).get("passed") is not True:
                raise ValueError(f"{sidecar}: harmonized delivery QA failed")
            measured = (
                (
                    (harmony.get("metrics") or {}).get(
                        "identity_risk_edges_after_global"
                    )
                    or {}
                )
                .get("p95")
            )
            if measured is None:
                raise ValueError(f"{sidecar}: missing harmonized edge p95")
            if (
                max_harmonized_edge_p95 is not None
                and float(measured) > max_harmonized_edge_p95
            ):
                raise ValueError(
                    f"{sidecar}: harmonized edge p95 {float(measured):.6f} "
                    f"exceeds {max_harmonized_edge_p95:.6f}"
                )
    else:
        # Compatibility with mosaics produced by mosaic_granule_sr.py before
        # national delivery switched to identity-aware seam ramps.
        if int(summary.get("n_missing", -1)) != 0:
            raise ValueError(f"{sidecar}: delivery reports missing tiles")
        if int((summary.get("mosaic") or {}).get("n_sources", -1)) != expected:
            raise ValueError(f"{sidecar}: delivery source count does not equal {expected}")
    if not mosaic.is_file() or mosaic.stat().st_size == 0:
        raise ValueError(f"{mosaic}: missing or empty delivery mosaic")
    if identity_path is not None and mosaic.stat().st_mtime_ns < identity_path.stat().st_mtime_ns:
        raise ValueError(f"{mosaic}: delivery predates its effective identity plan")


def _validate_cross(
    out: Path,
    sources: list[Path],
    *,
    max_overlap_mae: float | None = None,
    max_nodata_fraction: float | None = None,
) -> None:
    summary = _json(out.with_suffix(out.suffix + ".json"))
    recorded = [str(Path(value).resolve()) for value in summary.get("sources") or []]
    expected = [str(path.resolve()) for path in sources]
    if recorded != expected:
        raise ValueError("cross-granule mosaic source inventory is incomplete or reordered")
    details = summary.get("source_details") or []
    if len(details) != len(sources):
        raise ValueError("cross-granule mosaic source provenance is incomplete")
    for source, detail in zip(sources, details, strict=True):
        stat = source.stat()
        if str(Path(detail.get("path", "")).resolve()) != str(source.resolve()):
            raise ValueError("cross-granule mosaic source provenance is reordered")
        if int(detail.get("size_bytes", -1)) != stat.st_size:
            raise ValueError(f"{out}: source size changed after cross mosaicking")
        if int(detail.get("mtime_ns", -1)) != stat.st_mtime_ns:
            raise ValueError(f"{out}: source mtime changed after cross mosaicking")
    if not out.is_file() or out.stat().st_size == 0:
        raise ValueError(f"{out}: missing or empty cross-granule mosaic")
    if sources and out.stat().st_mtime_ns < max(path.stat().st_mtime_ns for path in sources):
        raise ValueError(f"{out}: cross-granule mosaic predates a delivery source")
    verification = summary.get("verification") or {}
    if max_overlap_mae is not None:
        value = verification.get("cross_source_overlap_mae")
        if value is not None and float(value) > max_overlap_mae:
            raise ValueError(
                f"{out}: cross-source overlap MAE {float(value):.6f} "
                f"exceeds {max_overlap_mae:.6f}"
            )
    if max_nodata_fraction is not None:
        value = verification.get("nodata_gap_fraction")
        if value is None or float(value) > max_nodata_fraction:
            raise ValueError(
                f"{out}: nodata fraction {value} exceeds {max_nodata_fraction:.6f}"
            )


def stages_for_granule(
    config: dict,
    row: dict,
    *,
    gpus: int,
    gpu_offset: int,
    granules: list[dict] | None = None,
) -> list[Stage]:
    mgrs = row["mgrs"]
    paths = paths_for(config, row)
    effective = effective_identity_paths(
        config,
        row,
        config.get("_granules", [row]) if granules is None else granules,
    )
    side = int(_recipe(config, row, "side", 512))
    start = str(_recipe(config, row, "start_date", "2025-05-31"))
    end = str(_recipe(config, row, "end_date", "2025-08-29"))
    center = str(_recipe(config, row, "date", "2025-07-15"))
    max_frames = int(_recipe(config, row, "max_frames", 16))
    overlap_frac = float(_recipe(config, row, "overlap_frac", 0.0))
    max_snow = float(_recipe(config, row, "max_snow_frac", 0.05))
    max_cloud = float(_recipe(config, row, "max_cloud_frac", 0.15))
    min_valid = float(_recipe(config, row, "min_valid_frac", 0.85))
    max_items = int(_recipe(config, row, "max_stac_items", 400))
    max_stac_cloud = float(_recipe(config, row, "max_stac_cloud", 100.0))
    min_land = float(_recipe(config, row, "min_land_frac", 0.0))
    land_mask = row.get("land_mask", config["recipe"].get("land_mask"))
    mainland_only = bool(_recipe(config, row, "mainland_only", True))
    iters = int(_recipe(config, row, "iters", 5000))
    production_config = {
        **config.get("production", {}),
        **row.get("production", {}),
    }
    scope = str(production_config.get("scope", "identity_changed"))
    if scope not in {"all", "identity_changed"}:
        raise ValueError(
            f"{mgrs}: production.scope must be 'all' or 'identity_changed'"
        )
    run_prefix = str(
        production_config.get(
            "run_prefix",
            _recipe(config, row, "run_prefix", "prod_k4_icm"),
        )
    )
    fallback_prefix = str(
        production_config.get(
            "fallback_prefix",
            run_prefix if scope == "all" else "prod_k4",
        )
    )

    plan = [
        sys.executable,
        _script("plan_national_mgrs.py"),
        "--mgrs", mgrs,
        "--date", center,
        "--start-date", start,
        "--end-date", end,
        "--side", str(side),
        "--max-cloud-frac", str(max_cloud),
        "--min-valid-frac", str(min_valid),
        "--max-snow-frac", str(max_snow),
        "--max-stac-items", str(max_items),
        "--max-stac-cloud", str(max_stac_cloud),
        "--max-frames", str(max_frames),
        "--min-land-frac", str(min_land),
        "--out-dir", str(paths["plan_dir"]),
        "--mainland-only" if mainland_only else "--no-mainland-only",
    ]
    if land_mask:
        plan += ["--land-mask", str(_resolve(land_mask, Path(land_mask)))]

    fetch = [
        sys.executable,
        _script("fetch_national_mgrs.py"),
        "--mgrs", mgrs,
        "--date", center,
        "--start-date", start,
        "--end-date", end,
        "--max-frames", str(max_frames),
        "--side", str(side),
        "--max-snow-frac", str(max_snow),
        "--max-cloud-frac", str(max_cloud),
        "--min-valid-frac", str(min_valid),
        "--max-stac-items", str(max_items),
        "--fetch-only",
        "--plan-dir", str(paths["plan_dir"]),
        "--s2-out", str(paths["s2"]),
    ]
    tile = [
        sys.executable,
        _script("make_granule_tiles.py"),
        "--src", str(paths["s2"]),
        "--out-root", str(config["_paths"]["tiles"]),
        "--side", str(side),
        "--manifest", str(paths["manifest"]),
        "--cell-plan", str(paths["ondisk"]),
        "--min-land-frac", str(min_land),
    ]
    if overlap_frac > 0:
        tile += ["--overlap-frac", str(overlap_frac)]
    _append_bool(tile, mainland_only, "--mainland-only")
    if land_mask:
        tile += ["--land-mask", str(_resolve(land_mask, Path(land_mask)))]

    identity_config = {**config.get("identity", {}), **row.get("identity", {})}
    identity_enabled = bool(identity_config.get("enabled", True))
    if not identity_enabled:
        raise ValueError(f"{mgrs}: identity must be enabled for national delivery")
    grid = identity_config.get("grid", [21, 21])
    block = identity_config.get("block", [0, int(grid[0]), 0, int(grid[1])])
    if not isinstance(block, list) or len(block) != 4:
        raise ValueError(f"{mgrs}: identity.block must be [y0, y1, x0, x1]")
    identity = [
        sys.executable,
        _script("plan_identity_icm.py"),
        "--parent", mgrs,
        "--s2-dir", str(paths["s2"]),
        "--plan-ondisk", str(paths["ondisk"]),
        "--y0", str(block[0]),
        "--y1", str(block[1]),
        "--x0", str(block[2]),
        "--x1", str(block[3]),
        "--hard-cloud", str(identity_config.get("hard_cloud", 0.02)),
        "--slack-cloud", str(identity_config.get("slack_cloud", 0.05)),
        "--out", str(paths["identity"]),
        "--granule-manifest", str(paths["manifest"]),
    ]
    if overlap_frac > 0:
        identity.append("--manifest-cells")

    production_manifest = (
        effective["identity_manifest"]
        if scope == "all"
        else effective["identity_changed_manifest"]
    )
    production = [
        sys.executable,
        _script("run_production.py"),
        "--manifest", str(production_manifest),
        "--iters", str(iters),
        "--gpus", str(gpus),
        "--gpu-offset", str(gpu_offset),
        "--skip-existing",
        "--run-prefix", run_prefix,
        "--max-base-cloud-frac", str(identity_config.get("hard_cloud", 0.02)),
        "--out", str(paths["summary"]),
    ]
    production += ["--identity-plan", str(effective["identity"])]

    delivery_config = {**config.get("delivery", {}), **row.get("delivery", {})}
    ramp_px = int(delivery_config.get("ramp_px", 128))
    harmonize = bool(delivery_config.get("harmonize", False))
    harmonized_edge_p95 = delivery_config.get("max_harmonized_edge_p95")
    if overlap_frac > 0:
        if scope != "all" or fallback_prefix != run_prefix:
            raise ValueError(
                f"{mgrs}: overlap production requires scope=all and one run prefix"
            )
        overlap_lr = int(round(side * overlap_frac))
        delivery = [
            sys.executable,
            _script("mosaic_granule_sr.py"),
            "--manifest", str(effective["identity_manifest"]),
            "--out", str(paths["mosaic"]),
            "--run-prefix", run_prefix,
            "--identity-plan", str(effective["identity"]),
            "--method", "feather",
            "--feather-px", str(overlap_lr * 4),
        ]
        if harmonize:
            delivery.append("--harmonize")
        same_limit = delivery_config.get("max_same_identity_overlap_p95")
        risk_limit = delivery_config.get("max_identity_risk_overlap_p95")
        if same_limit is not None:
            delivery += [
                "--max-same-identity-overlap-p95", str(float(same_limit))
            ]
        if risk_limit is not None:
            delivery += [
                "--max-identity-risk-overlap-p95", str(float(risk_limit))
            ]
    else:
        delivery = [
            sys.executable,
            _script("mosaic_seam_ramp.py"),
            "--manifest", str(effective["identity_manifest"]),
            "--out", str(paths["mosaic"]),
            "--run-prefix", run_prefix,
            "--fallback-prefix", fallback_prefix,
            "--identity-plan", str(effective["identity"]),
            "--ramp-px", str(ramp_px),
            "--mosaic",
            "--no-preview",
        ]
        if harmonize:
            delivery.append("--harmonize")
    if harmonize:
        delivery += [
            "--harmonize-strip-px",
            str(int(delivery_config.get("harmonize_strip_px", 256))),
            "--harmonize-segments",
            str(int(delivery_config.get("harmonize_segments", 16))),
            "--harmonize-regularization",
            str(float(delivery_config.get("harmonize_regularization", 0.02))),
            "--harmonize-max-offset",
            str(float(delivery_config.get("harmonize_max_offset", 0.08))),
            "--harmonize-smoothness",
            str(float(delivery_config.get("harmonize_smoothness", 0.1))),
        ]
        if overlap_frac == 0 and harmonized_edge_p95 is not None:
            delivery += [
                "--max-harmonized-edge-p95",
                str(float(harmonized_edge_p95)),
            ]

    stages = [
        Stage(
            f"{mgrs}:plan", "plan", plan, (paths["plan"],),
            lambda: _validate_plan(paths["plan"], mgrs), mgrs,
        ),
        Stage(
            f"{mgrs}:fetch", "fetch", fetch, (paths["meta"], paths["ondisk"]),
            lambda: _validate_fetch(paths["meta"], paths["ondisk"], mgrs), mgrs,
        ),
        Stage(
            f"{mgrs}:tile", "tile", tile, (paths["manifest"],),
            lambda: _validate_tile(paths["manifest"], paths["ondisk"]), mgrs,
        ),
    ]
    stages.append(
        Stage(
            f"{mgrs}:identity", "identity", identity,
            (
                paths["identity"],
                paths["identity_manifest"],
                paths["identity_changed_manifest"],
            ),
            lambda: _validate_identity(paths["identity"], paths["manifest"]), mgrs,
        )
    )
    stages.extend(
        [
            Stage(
                f"{mgrs}:production", "production", production, (paths["summary"],),
                lambda: _validate_production(
                    paths["summary"],
                    production_manifest,
                    identity_path=effective["identity"],
                    run_prefix=run_prefix,
                ),
                mgrs,
            ),
            Stage(
                f"{mgrs}:delivery", "delivery", delivery,
                (paths["mosaic"], paths["mosaic"].with_suffix(paths["mosaic"].suffix + ".json")),
                lambda: _validate_delivery(
                    paths["mosaic"],
                    effective["identity_manifest"],
                    ramp_px=ramp_px,
                    require_date_cuts=True,
                    require_harmonization=harmonize,
                    max_harmonized_edge_p95=(
                        None
                        if harmonized_edge_p95 is None
                        else float(harmonized_edge_p95)
                    ),
                    max_same_identity_overlap_p95=(
                        None
                        if delivery_config.get("max_same_identity_overlap_p95")
                        is None
                        else float(
                            delivery_config["max_same_identity_overlap_p95"]
                        )
                    ),
                    max_identity_risk_overlap_p95=(
                        None
                        if delivery_config.get("max_identity_risk_overlap_p95")
                        is None
                        else float(
                            delivery_config["max_identity_risk_overlap_p95"]
                        )
                    ),
                    identity_path=effective["identity"],
                ),
                mgrs,
            ),
        ]
    )
    return stages


def _cross_identity_input(config: dict, granules: list[dict]) -> dict:
    return {
        "schema_version": 1,
        "granules": [
            {
                "mgrs": row["mgrs"],
                "identity_plan": str(paths_for(config, row)["identity"]),
                "granule_manifest": str(paths_for(config, row)["manifest"]),
                "s2_meta": str(paths_for(config, row)["meta"]),
            }
            for row in sorted(granules, key=lambda item: item["mgrs"])
        ],
    }


def _write_revised_manifests(config: dict, granules: list[dict]) -> None:
    layout = _cross_identity_layout(config, granules)
    for row in granules:
        original = paths_for(config, row)
        revised_dir = layout["out"] / row["mgrs"]
        revised_path = revised_dir / "identity_plan.json"
        if not revised_path.is_file():
            continue
        plan = _json(revised_path)
        manifest = _json(original["manifest"])
        assignment = plan.get("assignment") or {}
        independent = plan.get("independent") or {}
        source_by_id = {
            tile["tile_id"]: tile for tile in manifest.get("tiles") or []
        }
        if set(assignment) != set(source_by_id):
            raise ValueError(
                f"{revised_path}: revised assignment differs from granule manifest"
            )
        tiles = []
        for tile_id in sorted(
            source_by_id,
            key=lambda value: (
                int(source_by_id[value].get("iy", 0)),
                int(source_by_id[value].get("ix", 0)),
                value,
            ),
        ):
            tile = dict(source_by_id[tile_id])
            tile["force_base_date"] = assignment[tile_id]
            tiles.append(tile)
        common = {key: value for key, value in manifest.items() if key != "tiles"}
        common.update(
            {
                "identity_plan": str(revised_path),
                "cross_mgrs_revised": True,
            }
        )
        atomic_write_json(revised_dir / "manifest_icm_all.json", {**common, "tiles": tiles})
        changed = [
            tile
            for tile in tiles
            if assignment[tile["tile_id"]] != independent.get(tile["tile_id"])
        ]
        atomic_write_json(
            revised_dir / "manifest_icm_changed.json",
            {**common, "subset": "identity_changed_only", "tiles": changed},
        )


def cross_identity_stage(config: dict, granules: list[dict]) -> Stage | None:
    settings = config.get("cross_identity") or {}
    if not bool(settings.get("enabled", False)):
        return None
    if len(granules) < 2:
        raise ValueError("cross_identity requires at least two selected granules")
    layout = _cross_identity_layout(config, granules)
    audit_only = bool(settings.get("audit_only", False))
    input_payload = _cross_identity_input(config, granules)

    def prepare() -> None:
        atomic_write_json(layout["input"], input_payload)

    command = [
        sys.executable,
        _script("plan_cross_mgrs_identity.py"),
        "--inputs-manifest",
        str(layout["input"]),
        "--common-crs",
        str(settings.get("common_crs", "EPSG:3857")),
        "--adjacency-tolerance-m",
        str(settings.get("adjacency_tolerance_m", 1.0)),
        "--slack-cloud",
        str(settings.get("slack_cloud", 0.05)),
        "--center-date",
        str(settings.get("center_date", "2025-07-15")),
    ]
    if audit_only:
        command += ["--audit-only", "--report", str(layout["audit"])]
        expected = (layout["input"], layout["audit"])
    else:
        command += ["--out-dir", str(layout["out"])]
        outputs = [layout["input"], layout["audit"]]
        for row in sorted(granules, key=lambda item: item["mgrs"]):
            effective = effective_identity_paths(config, row, granules)
            outputs.extend(
                [
                    effective["identity"],
                    effective["identity_manifest"],
                    effective["identity_changed_manifest"],
                ]
            )
        expected = tuple(outputs)

    def validate() -> None:
        if _json(layout["input"]) != input_payload:
            raise ValueError("cross-identity input manifest does not match selection")
        audit = _json(layout["audit"])
        if int((audit.get("summary") or {}).get("n_granules", -1)) != len(granules):
            raise ValueError("cross-identity audit granule count is incomplete")
        if bool((audit.get("configuration") or {}).get("audit_only")) != audit_only:
            raise ValueError("cross-identity audit mode differs from configuration")
        if not audit_only:
            for row in granules:
                effective = effective_identity_paths(config, row, granules)
                _validate_identity(effective["identity"], paths_for(config, row)["manifest"])

    return Stage(
        "national:cross_identity",
        "cross_identity",
        command,
        expected,
        validate,
        prepare=prepare,
        finalize=(
            None if audit_only else lambda: _write_revised_manifests(config, granules)
        ),
    )


def cross_stage(config: dict, granules: list[dict]) -> Stage:
    sources = [paths_for(config, row)["mosaic"] for row in granules]
    cross_config = config.get("cross") or {}
    selected_ids = [row["mgrs"] for row in granules]
    all_ids = [row["mgrs"] for row in config["_granules"]]
    subset_suffix = ""
    if selected_ids != all_ids:
        subset_hash = hashlib.sha256(",".join(selected_ids).encode()).hexdigest()[:10]
        subset_suffix = f"_subset_{subset_hash}"
    out = _resolve(
        cross_config.get("out"),
        config["_paths"]["mosaics"]
        / f"{config['_country_slug']}{subset_suffix}_sr_2p5m.tif",
    )
    if subset_suffix and cross_config.get("out"):
        out = out.with_name(f"{out.stem}{subset_suffix}{out.suffix}")
    command = [
        sys.executable,
        _script("mosaic_national_sr.py"),
        "--sources", *(str(path) for path in sources),
        "--out", str(out),
        "--dst-crs", str(cross_config.get("dst_crs", "EPSG:3035")),
        "--resolution", str(cross_config.get("resolution", 2.5)),
    ]
    max_overlap_mae = cross_config.get("max_overlap_mae")
    max_nodata_fraction = cross_config.get("max_nodata_fraction")
    if max_overlap_mae is not None:
        command += ["--max-overlap-mae", str(float(max_overlap_mae))]
    if max_nodata_fraction is not None:
        command += ["--max-nodata-fraction", str(float(max_nodata_fraction))]
    return Stage(
        "national:cross", "cross", command,
        (out, out.with_suffix(out.suffix + ".json")),
        lambda: _validate_cross(
            out,
            sources,
            max_overlap_mae=(
                None if max_overlap_mae is None else float(max_overlap_mae)
            ),
            max_nodata_fraction=(
                None if max_nodata_fraction is None else float(max_nodata_fraction)
            ),
        ),
    )


def _outputs_match(record: dict, expected: tuple[Path, ...]) -> bool:
    saved = record.get("expected_outputs") or []
    if record.get("status") not in {"success", "resumed", "adopted"} or len(saved) != len(expected):
        return False
    by_path = {item.get("path"): item for item in saved}
    try:
        return all(
            path.is_file()
            and by_path.get(str(path), {}).get("sha256") == sha256_file(path)
            for path in expected
        )
    except OSError:
        return False


def _record_blocked(state: dict, state_path: Path, stage: Stage, reason: str) -> None:
    previous = state["stages"].get(stage.key, {})
    state["stages"][stage.key] = {
        **previous,
        "name": stage.name,
        "mgrs": stage.mgrs,
        "status": "blocked",
        "reason": reason,
        "command": stage.command,
        "command_shell": shlex.join(stage.command),
        "expected_outputs": [{"path": str(path), "sha256": None} for path in stage.expected],
        "updated_utc": utc_now(),
    }
    atomic_write_json(state_path, state)


def run_stage(
    stage: Stage,
    *,
    state: dict,
    state_path: Path,
    dry_run: bool,
    resume: bool,
    retries: int,
    backoff: float,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    sleeper: Callable[[float], None] = time.sleep,
) -> bool:
    previous = state["stages"].get(stage.key, {})
    base = {
        "name": stage.name,
        "mgrs": stage.mgrs,
        "command": stage.command,
        "command_shell": shlex.join(stage.command),
        "expected_outputs": [{"path": str(path), "sha256": None} for path in stage.expected],
        "attempts": list(previous.get("attempts") or []),
    }
    if stage.prepare is not None:
        started = utc_now()
        try:
            stage.prepare()
        except Exception as exc:  # noqa: BLE001
            attempt = {
                "number": 0,
                "phase": "prepare",
                "status": "failed",
                "started_utc": started,
                "finished_utc": utc_now(),
                "command": stage.command,
                "command_shell": base["command_shell"],
                "error": f"{type(exc).__name__}: {exc}",
            }
            base["attempts"].append(attempt)
            state["stages"][stage.key] = {
                **base,
                "status": "failed",
                "error": attempt["error"],
                "updated_utc": utc_now(),
            }
            atomic_write_json(state_path, state)
            return False
    if resume and _outputs_match(previous, stage.expected):
        try:
            stage.validate()
        except Exception:
            pass
        else:
            state["stages"][stage.key] = {
                **base,
                "status": "resumed",
                "resume_reason": "prior success, hashes unchanged, validation passed",
                "expected_outputs": previous["expected_outputs"],
                "updated_utc": utc_now(),
            }
            atomic_write_json(state_path, state)
            print(f"RESUME {stage.key}")
            return True
    if resume and not previous and all(path.is_file() for path in stage.expected):
        try:
            stage.validate()
            outputs = [file_fingerprint(path) for path in stage.expected]
        except Exception:
            pass
        else:
            state["stages"][stage.key] = {
                **base,
                "status": "adopted",
                "adoption_reason": (
                    "expected outputs existed before this state and validation passed"
                ),
                "expected_outputs": outputs,
                "updated_utc": utc_now(),
            }
            atomic_write_json(state_path, state)
            print(f"ADOPT {stage.key}")
            return True
    if dry_run:
        state["stages"][stage.key] = {
            **base,
            "status": "planned",
            "updated_utc": utc_now(),
        }
        atomic_write_json(state_path, state)
        print(f"PLAN {stage.key}: {base['command_shell']}")
        return True

    total_attempts = retries + 1
    for number in range(1, total_attempts + 1):
        started = utc_now()
        tick = time.monotonic()
        attempt = {
            "number": number,
            "started_utc": started,
            "command": stage.command,
            "command_shell": base["command_shell"],
        }
        try:
            completed = runner(stage.command, cwd=ROOT, check=True)
            if stage.finalize is not None:
                stage.finalize()
            stage.validate()
            outputs = [file_fingerprint(path) for path in stage.expected]
            attempt.update(
                {
                    "status": "success",
                    "returncode": completed.returncode,
                    "finished_utc": utc_now(),
                    "duration_s": round(time.monotonic() - tick, 3),
                    "expected_outputs": outputs,
                }
            )
            base["attempts"].append(attempt)
            state["stages"][stage.key] = {
                **base,
                "status": "success",
                "expected_outputs": outputs,
                "updated_utc": utc_now(),
            }
            atomic_write_json(state_path, state)
            print(f"DONE {stage.key}", flush=True)
            return True
        except Exception as exc:  # noqa: BLE001
            attempt.update(
                {
                    "status": "failed",
                    "returncode": getattr(exc, "returncode", None),
                    "error": f"{type(exc).__name__}: {exc}",
                    "finished_utc": utc_now(),
                    "duration_s": round(time.monotonic() - tick, 3),
                }
            )
            base["attempts"].append(attempt)
            state["stages"][stage.key] = {
                **base,
                "status": "failed",
                "error": attempt["error"],
                "updated_utc": utc_now(),
            }
            atomic_write_json(state_path, state)
            print(
                f"FAIL {stage.key} attempt {number}/{total_attempts}: {exc}",
                flush=True,
            )
            if number < total_attempts:
                delay = backoff * (2 ** (number - 1))
                state["stages"][stage.key]["next_retry_delay_s"] = delay
                atomic_write_json(state_path, state)
                sleeper(delay)
    return False


def cleanup_delivery_sources(
    config: dict,
    row: dict,
    *,
    state: dict,
    state_path: Path,
    dry_run: bool,
    granules: list[dict] | None = None,
) -> None:
    """Remove only per-tile QGIS exports after complete delivery validation."""
    paths = paths_for(config, row)
    effective = effective_identity_paths(
        config,
        row,
        config.get("_granules", [row]) if granules is None else granules,
    )
    delivery_config = {**config.get("delivery", {}), **row.get("delivery", {})}
    _validate_delivery(
        paths["mosaic"],
        effective["identity_manifest"],
        ramp_px=int(delivery_config.get("ramp_px", 128)),
        require_harmonization=bool(delivery_config.get("harmonize", False)),
        max_harmonized_edge_p95=(
            None
            if delivery_config.get("max_harmonized_edge_p95") is None
            else float(delivery_config["max_harmonized_edge_p95"])
        ),
        identity_path=effective["identity"],
    )
    manifest, tiles = _manifest_tiles(effective["identity_manifest"])
    production_config = {
        **config.get("production", {}),
        **row.get("production", {}),
    }
    run_prefix = str(production_config.get("run_prefix", "prod_k4_icm"))
    parent = manifest["parent"]
    targets = []
    for tile in tiles:
        qgis = (
            ROOT / "single_samples" / parent / "sample"
            / f"{run_prefix}_{tile['tile_id']}" / "qgis"
        )
        if qgis.is_dir():
            targets.append(qgis)
    record = {
        "name": "cleanup",
        "mgrs": row["mgrs"],
        "status": "planned" if dry_run else "success",
        "validation": "complete delivery mosaic and zero missing sources",
        "targets": [str(path) for path in targets],
        "target_files": [
            file_fingerprint(file)
            for path in targets
            for file in sorted(path.rglob("*"))
            if file.is_file()
        ],
        "updated_utc": utc_now(),
    }
    if not dry_run:
        for path in targets:
            shutil.rmtree(path)
        record["removed_count"] = len(targets)
    state["stages"][f"{row['mgrs']}:cleanup"] = record
    atomic_write_json(state_path, state)


def parse_stage_selection(raw: str) -> set[str]:
    if raw.strip() == "all":
        return set(STAGES)
    selected = {item.strip() for item in raw.split(",") if item.strip()}
    unknown = selected - set(STAGES)
    if unknown or not selected:
        raise ValueError(f"invalid --stages values: {sorted(unknown) or raw!r}")
    return selected


def select_granules(
    inventory: list[dict],
    *,
    requested: str | None,
    wave_size: int,
    wave_index: int,
) -> list[dict]:
    """Select an inventory-ordered explicit subset and/or zero-based wave."""
    if wave_size < 0 or wave_index < 0:
        raise ValueError("--wave-size and --wave-index must be non-negative")
    selected = list(inventory)
    if requested:
        names = [value.strip().upper().lstrip("T") for value in requested.split(",")]
        if any(not value for value in names) or len(names) != len(set(names)):
            raise ValueError("--granules must contain unique, non-empty MGRS IDs")
        wanted = set(names)
        available = {row["mgrs"] for row in inventory}
        unknown = sorted(wanted - available)
        if unknown:
            raise ValueError(f"--granules not present in inventory: {unknown}")
        # Preserve inventory order regardless of the CLI order.
        selected = [row for row in inventory if row["mgrs"] in wanted]
    if wave_size == 0:
        if wave_index != 0:
            raise ValueError("--wave-index requires a positive --wave-size")
        return selected
    start = wave_index * wave_size
    selected = selected[start : start + wave_size]
    if not selected:
        raise ValueError(
            f"wave {wave_index} is empty for {len(inventory)} inventory granules"
        )
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, default=None)
    parser.add_argument("--state", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-backoff", type=float, default=5.0)
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--gpu-offset", type=int, default=0)
    parser.add_argument("--stages", default="all", help="Comma-separated stages or all")
    parser.add_argument(
        "--granules",
        default=None,
        help="Comma-separated MGRS subset; execution preserves inventory order.",
    )
    parser.add_argument(
        "--wave-size",
        type=int,
        default=0,
        help="Deterministic inventory wave size (0 disables waves).",
    )
    parser.add_argument(
        "--wave-index",
        type=int,
        default=0,
        help="Zero-based wave index; requires --wave-size.",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--cleanup-delivery-only",
        action="store_true",
        help="After validated delivery, remove per-tile qgis exports only.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.retries < 0 or args.retry_backoff < 0:
        raise SystemExit("--retries and --retry-backoff must be non-negative")
    if args.gpus < 1 or args.gpu_offset < 0:
        raise SystemExit("--gpus must be positive and --gpu-offset non-negative")
    try:
        selected = parse_stage_selection(args.stages)
        config = load_country_config(args.config, args.inventory)
        granules = select_granules(
            config["_granules"],
            requested=args.granules,
            wave_size=args.wave_size,
            wave_index=args.wave_index,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc

    state_path = _resolve(
        args.state,
        config["_paths"]["production"] / "run_state.json",
    )
    digest = config_digest(config, granules)
    if state_path.is_file() and args.resume:
        state = _json(state_path)
        if state.get("config_sha256") != digest:
            raise SystemExit(
                f"{state_path}: config/inventory changed; use a new --state "
                "or rerun with --no-resume"
            )
    else:
        state = {
            "schema_version": 1,
            "country": config["country"],
            "config": str(config["_config_path"]),
            "config_sha256": digest,
            "selected_granules": [row["mgrs"] for row in granules],
            "created_utc": utc_now(),
            "stages": {},
        }
    state.update(
        {
            "dry_run": bool(args.dry_run),
            "selected_stages": sorted(selected, key=STAGES.index),
            "selected_granules": [row["mgrs"] for row in granules],
            "last_started_utc": utc_now(),
            "status": "running",
        }
    )
    atomic_write_json(state_path, state)

    failed = False
    ready = {row["mgrs"]: True for row in granules}
    stage_maps = {
        row["mgrs"]: {
            stage.name: stage
            for stage in stages_for_granule(
                config,
                row,
                gpus=args.gpus,
                gpu_offset=args.gpu_offset,
                granules=granules,
            )
        }
        for row in granules
    }

    def execute(stage: Stage) -> bool:
        nonlocal failed
        ok = run_stage(
            stage,
            state=state,
            state_path=state_path,
            dry_run=args.dry_run,
            resume=args.resume,
            retries=args.retries,
            backoff=args.retry_backoff,
        )
        failed = failed or not ok
        if not ok and not args.continue_on_error:
            state["status"] = "failed"
            state["finished_utc"] = utc_now()
            atomic_write_json(state_path, state)
        return ok

    # Stage-major ordering is deliberate: cross-MGRS coordination must see all
    # local identity plans before any production process reads an assignment.
    for name in ("plan", "fetch", "tile", "identity"):
        if name not in selected:
            continue
        for row in granules:
            mgrs = row["mgrs"]
            stage = stage_maps[mgrs][name]
            if not ready[mgrs]:
                _record_blocked(state, state_path, stage, "earlier selected stage failed")
                failed = True
                continue
            ready[mgrs] = execute(stage)
            if not ready[mgrs] and not args.continue_on_error:
                return 1

    cross_identity_ok = True
    coordinator = (
        cross_identity_stage(config, granules)
        if "cross_identity" in selected
        else None
    )
    if coordinator is not None:
        blocked_granules = sorted(mgrs for mgrs, ok in ready.items() if not ok)
        if blocked_granules:
            _record_blocked(
                state,
                state_path,
                coordinator,
                f"per-granule identity prerequisites failed: {blocked_granules}",
            )
            cross_identity_ok = False
            failed = True
        else:
            cross_identity_ok = execute(coordinator)
            if not cross_identity_ok and not args.continue_on_error:
                return 1

    if coordinator is not None and not cross_identity_ok:
        for mgrs in ready:
            ready[mgrs] = False

    for name in ("production", "delivery"):
        if name not in selected:
            continue
        for row in granules:
            mgrs = row["mgrs"]
            stage = stage_maps[mgrs][name]
            if not ready[mgrs]:
                _record_blocked(
                    state,
                    state_path,
                    stage,
                    "earlier per-granule or cross-identity stage failed",
                )
                failed = True
                continue
            ready[mgrs] = execute(stage)
            if not ready[mgrs] and not args.continue_on_error:
                return 1

    if args.cleanup_delivery_only and "delivery" in selected:
        for row in granules:
            if not ready[row["mgrs"]]:
                continue
            try:
                cleanup_delivery_sources(
                    config,
                    row,
                    state=state,
                    state_path=state_path,
                    dry_run=args.dry_run,
                    granules=granules,
                )
            except Exception as exc:  # noqa: BLE001
                failed = True
                ready[row["mgrs"]] = False
                state["stages"][f"{row['mgrs']}:cleanup"] = {
                    "name": "cleanup",
                    "mgrs": row["mgrs"],
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "updated_utc": utc_now(),
                }
                atomic_write_json(state_path, state)
                if not args.continue_on_error:
                    state["status"] = "failed"
                    state["finished_utc"] = utc_now()
                    atomic_write_json(state_path, state)
                    return 1

    if "cross" in selected:
        stage = cross_stage(config, granules)
        sources = [paths_for(config, row)["mosaic"] for row in granules]
        unavailable = [str(path) for path in sources if not path.is_file()]
        blocked_granules = sorted(mgrs for mgrs, ok in ready.items() if not ok)
        if blocked_granules:
            _record_blocked(
                state,
                state_path,
                stage,
                f"delivery prerequisites failed: {blocked_granules}",
            )
            failed = True
        elif unavailable and not args.dry_run:
            _record_blocked(
                state,
                state_path,
                stage,
                f"missing inventory delivery mosaics: {unavailable}",
            )
            failed = True
        else:
            ok = run_stage(
                stage,
                state=state,
                state_path=state_path,
                dry_run=args.dry_run,
                resume=args.resume,
                retries=args.retries,
                backoff=args.retry_backoff,
            )
            failed = failed or not ok

    state["status"] = "failed" if failed else ("planned" if args.dry_run else "success")
    state["finished_utc"] = utc_now()
    atomic_write_json(state_path, state)
    print(f"Run state -> {state_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
