"""Load and validate country-specific national inventory configuration."""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ISO_A2 = re.compile(r"^[A-Z]{2}$")
ISO_A3 = re.compile(r"^[A-Z]{3}$")
MGRS_TILE = re.compile(r"^[0-9]{2}[C-HJ-NP-X][A-HJ-NP-Z]{2}$")


class CountryConfigError(ValueError):
    """Raised when a country configuration is incomplete or invalid."""


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_object(
    config: dict, key: str, errors: list[str], *, allowed: set[str]
) -> dict | None:
    value = config.get(key)
    if value is None:
        return None
    if not isinstance(value, dict):
        errors.append(f"{key} must be an object")
        return None
    unknown = sorted(set(value) - allowed)
    if unknown:
        errors.append(f"{key} has unsupported fields: {unknown}")
    return value


def _non_empty_string(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _validate_date(value, field: str, errors: list[str]) -> date | None:
    if not _non_empty_string(value):
        errors.append(f"{field} must be an ISO date")
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        errors.append(f"{field} must be an ISO date")
        return None


def _validate_fraction(value, field: str, errors: list[str], *, minimum: float = 0.0) -> None:
    if not _is_number(value) or not minimum <= float(value) <= 1.0:
        errors.append(f"{field} must be between {minimum:g} and 1")


def resolve_root_path(value: str | Path, *, root: Path = ROOT) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def validate_country_config(config: dict) -> None:
    errors: list[str] = []
    allowed_top_level = {
        "$schema",
        "schema_version",
        "country",
        "inputs",
        "inventory",
        "paths",
        "roots",
        "recipe",
        "production",
        "delivery",
        "cross_identity",
        "cross",
        "territory_scope",
        "inventory_exceptions",
    }
    unknown_top_level = sorted(set(config) - allowed_top_level)
    if unknown_top_level:
        errors.append(f"unsupported top-level fields: {unknown_top_level}")
    if config.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    country = config.get("country")
    if not isinstance(country, dict):
        errors.append("country must be an object")
        country = {}
    else:
        unknown = sorted(set(country) - {"name", "iso_a2", "iso_a3"})
        if unknown:
            errors.append(f"country has unsupported fields: {unknown}")
    if not isinstance(country.get("name"), str) or not country.get("name", "").strip():
        errors.append("country.name must be a non-empty string")
    if not ISO_A2.fullmatch(str(country.get("iso_a2", ""))):
        errors.append("country.iso_a2 must be two uppercase letters")
    if not ISO_A3.fullmatch(str(country.get("iso_a3", ""))):
        errors.append("country.iso_a3 must be three uppercase letters")

    inputs = config.get("inputs")
    if not isinstance(inputs, dict):
        errors.append("inputs must be an object")
        inputs = {}
    else:
        unknown = sorted(set(inputs) - {"land_outline_geojson", "land_outline_provenance"})
        if unknown:
            errors.append(f"inputs has unsupported fields: {unknown}")
    if not isinstance(inputs.get("land_outline_geojson"), str) or not inputs.get(
        "land_outline_geojson", ""
    ).strip():
        errors.append("inputs.land_outline_geojson must be a non-empty path")
    provenance = inputs.get("land_outline_provenance")
    if provenance is not None:
        if not isinstance(provenance, dict):
            errors.append("inputs.land_outline_provenance must be an object")
        else:
            allowed = {"dataset", "scale", "source_url", "source_sha256", "scope_note"}
            unknown = sorted(set(provenance) - allowed)
            if unknown:
                errors.append(
                    f"inputs.land_outline_provenance has unsupported fields: {unknown}"
                )
            for key in ("dataset", "source_url", "source_sha256"):
                if not _non_empty_string(provenance.get(key)):
                    errors.append(f"inputs.land_outline_provenance.{key} is required")
            source_url = provenance.get("source_url")
            if _non_empty_string(source_url) and not source_url.startswith(("https://", "http://")):
                errors.append("inputs.land_outline_provenance.source_url must be HTTP(S)")
            source_sha = provenance.get("source_sha256")
            if _non_empty_string(source_sha) and not re.fullmatch(r"[0-9a-f]{64}", source_sha):
                errors.append(
                    "inputs.land_outline_provenance.source_sha256 must be lowercase SHA-256"
                )
            for key in ("scale", "scope_note"):
                if key in provenance and not _non_empty_string(provenance[key]):
                    errors.append(f"inputs.land_outline_provenance.{key} must be non-empty")

    inventory = config.get("inventory")
    if not isinstance(inventory, dict):
        errors.append("inventory must be an object")
        inventory = {}
    else:
        unknown = sorted(
            set(inventory)
            - {
                "mgrs_list_json",
                "mgrs_bboxes_json",
                "min_tile_land_fraction",
                "include_bbox",
                "source_role",
                "expected_runnable_count",
                "expected_source_count",
                "diagnostic_mgrs_list_json",
                "limitations",
            }
        )
        if unknown:
            errors.append(f"inventory has unsupported fields: {unknown}")
    for key in ("mgrs_list_json", "mgrs_bboxes_json"):
        if not isinstance(inventory.get(key), str) or not inventory.get(key, "").strip():
            errors.append(f"inventory.{key} must be a non-empty path")
    min_fraction = inventory.get("min_tile_land_fraction", 0.0)
    _validate_fraction(
        min_fraction, "inventory.min_tile_land_fraction", errors
    )
    include_bbox = inventory.get("include_bbox")
    if include_bbox is not None and (
        not isinstance(include_bbox, list)
        or len(include_bbox) != 4
        or not all(_is_number(value) for value in include_bbox)
        or include_bbox[0] >= include_bbox[2]
        or include_bbox[1] >= include_bbox[3]
    ):
        errors.append("inventory.include_bbox must be [west,south,east,north]")
    source_role = inventory.get("source_role")
    if source_role is not None and source_role not in {"curated_operational", "outline_derived"}:
        errors.append(
            "inventory.source_role must be 'curated_operational' or 'outline_derived'"
        )
    expected_count = inventory.get("expected_runnable_count")
    if expected_count is not None and (
        not isinstance(expected_count, int)
        or isinstance(expected_count, bool)
        or expected_count < 1
    ):
        errors.append("inventory.expected_runnable_count must be a positive integer")
    expected_source_count = inventory.get("expected_source_count")
    if expected_source_count is not None and (
        not isinstance(expected_source_count, int)
        or isinstance(expected_source_count, bool)
        or expected_source_count < 1
    ):
        errors.append("inventory.expected_source_count must be a positive integer")
    if (
        isinstance(expected_count, int)
        and not isinstance(expected_count, bool)
        and isinstance(expected_source_count, int)
        and not isinstance(expected_source_count, bool)
        and expected_source_count < expected_count
    ):
        errors.append(
            "inventory.expected_source_count cannot be less than expected_runnable_count"
        )
    diagnostic = inventory.get("diagnostic_mgrs_list_json")
    if diagnostic is not None and not _non_empty_string(diagnostic):
        errors.append("inventory.diagnostic_mgrs_list_json must be a non-empty path")
    limitations = inventory.get("limitations")
    if limitations is not None and (
        not isinstance(limitations, list)
        or not limitations
        or any(not _non_empty_string(item) for item in limitations)
    ):
        errors.append("inventory.limitations must be a non-empty string array")
    if source_role == "curated_operational" and not _non_empty_string(diagnostic):
        errors.append(
            "curated operational inventory requires inventory.diagnostic_mgrs_list_json"
        )

    paths = config.get("paths")
    if not isinstance(paths, dict):
        errors.append("paths must be an object")
        paths = {}
    else:
        unknown = sorted(set(paths) - {"plan_root", "s2_root"})
        if unknown:
            errors.append(f"paths has unsupported fields: {unknown}")
    for key in ("plan_root", "s2_root"):
        if not isinstance(paths.get(key), str) or not paths.get(key, "").strip():
            errors.append(f"paths.{key} must be a non-empty path")

    roots = _validate_object(
        config,
        "roots",
        errors,
        allowed={"production", "plans", "data", "tiles", "mosaics", "results"},
    )
    if roots is not None:
        if not roots:
            errors.append("roots must contain at least one path")
        for key, value in roots.items():
            if not _non_empty_string(value):
                errors.append(f"roots.{key} must be a non-empty path")

    recipe = _validate_object(
        config,
        "recipe",
        errors,
        allowed={
            "date",
            "start_date",
            "end_date",
            "side",
            "max_frames",
            "max_snow_frac",
            "max_cloud_frac",
            "min_valid_frac",
            "max_stac_items",
            "max_stac_cloud",
            "min_land_frac",
            "mainland_only",
            "land_mask",
            "iters",
            "run_prefix",
            "overlap_frac",
        },
    )
    if recipe is not None:
        parsed_dates = {
            key: _validate_date(recipe[key], f"recipe.{key}", errors)
            for key in ("start_date", "date", "end_date")
            if key in recipe
        }
        if all(parsed_dates.get(key) for key in ("start_date", "date", "end_date")):
            if not (
                parsed_dates["start_date"] <= parsed_dates["date"] <= parsed_dates["end_date"]
            ):
                errors.append("recipe dates must satisfy start_date <= date <= end_date")
        for key in ("side", "max_frames", "max_stac_items", "iters"):
            if key in recipe and (
                not isinstance(recipe[key], int)
                or isinstance(recipe[key], bool)
                or recipe[key] < 1
            ):
                errors.append(f"recipe.{key} must be a positive integer")
        for key in (
            "max_cloud_frac",
            "min_valid_frac",
            "min_land_frac",
            "overlap_frac",
        ):
            if key in recipe:
                _validate_fraction(recipe[key], f"recipe.{key}", errors)
        if "overlap_frac" in recipe and _is_number(recipe["overlap_frac"]):
            if float(recipe["overlap_frac"]) >= 1:
                errors.append("recipe.overlap_frac must be less than 1")
        if "max_snow_frac" in recipe:
            _validate_fraction(
                recipe["max_snow_frac"], "recipe.max_snow_frac", errors, minimum=-1.0
            )
        if "max_stac_cloud" in recipe and (
            not _is_number(recipe["max_stac_cloud"])
            or not 0 <= float(recipe["max_stac_cloud"]) <= 100
        ):
            errors.append("recipe.max_stac_cloud must be between 0 and 100")
        if "mainland_only" in recipe and not isinstance(recipe["mainland_only"], bool):
            errors.append("recipe.mainland_only must be boolean")
        for key in ("land_mask", "run_prefix"):
            if key in recipe and not _non_empty_string(recipe[key]):
                errors.append(f"recipe.{key} must be non-empty")

    production = _validate_object(
        config,
        "production",
        errors,
        allowed={"scope", "run_prefix", "fallback_prefix"},
    )
    if production is not None:
        if "scope" in production and production["scope"] not in {"all", "identity_changed"}:
            errors.append("production.scope must be 'all' or 'identity_changed'")
        for key in ("run_prefix", "fallback_prefix"):
            if key in production and not _non_empty_string(production[key]):
                errors.append(f"production.{key} must be non-empty")

    delivery = _validate_object(
        config,
        "delivery",
        errors,
        allowed={
            "ramp_px",
            "harmonize",
            "harmonize_strip_px",
            "harmonize_segments",
            "harmonize_regularization",
            "harmonize_max_offset",
            "harmonize_smoothness",
            "max_harmonized_edge_p95",
            "max_same_identity_overlap_p95",
            "max_identity_risk_overlap_p95",
        },
    )
    if delivery is not None:
        if "ramp_px" in delivery and (
            not isinstance(delivery["ramp_px"], int)
            or isinstance(delivery["ramp_px"], bool)
            or delivery["ramp_px"] < 0
        ):
            errors.append("delivery.ramp_px must be a non-negative integer")
        if "harmonize" in delivery and not isinstance(delivery["harmonize"], bool):
            errors.append("delivery.harmonize must be boolean")
        for key, minimum in (
            ("harmonize_strip_px", 1),
            ("harmonize_segments", 2),
        ):
            if key in delivery and (
                not isinstance(delivery[key], int)
                or isinstance(delivery[key], bool)
                or delivery[key] < minimum
            ):
                errors.append(f"delivery.{key} must be an integer >= {minimum}")
        for key in ("harmonize_regularization", "harmonize_smoothness"):
            if key in delivery and (
                not _is_number(delivery[key]) or float(delivery[key]) < 0
            ):
                errors.append(f"delivery.{key} must be non-negative")
        for key in (
            "harmonize_max_offset",
            "max_harmonized_edge_p95",
            "max_same_identity_overlap_p95",
            "max_identity_risk_overlap_p95",
        ):
            if key in delivery and (
                not _is_number(delivery[key]) or not 0 <= float(delivery[key]) <= 1
            ):
                errors.append(f"delivery.{key} must be between 0 and 1")

    cross_identity = _validate_object(
        config,
        "cross_identity",
        errors,
        allowed={
            "enabled",
            "audit_only",
            "common_crs",
            "adjacency_tolerance_m",
            "slack_cloud",
            "center_date",
            "root",
            "out_dir",
            "inputs_manifest",
            "report",
        },
    )
    if cross_identity is not None:
        for key in ("enabled", "audit_only"):
            if key in cross_identity and not isinstance(cross_identity[key], bool):
                errors.append(f"cross_identity.{key} must be boolean")
        for key in ("common_crs", "root", "out_dir", "inputs_manifest", "report"):
            if key in cross_identity and not _non_empty_string(cross_identity[key]):
                errors.append(f"cross_identity.{key} must be non-empty")
        tolerance = cross_identity.get("adjacency_tolerance_m")
        if tolerance is not None and (not _is_number(tolerance) or tolerance < 0):
            errors.append("cross_identity.adjacency_tolerance_m must be non-negative")
        if "slack_cloud" in cross_identity:
            _validate_fraction(
                cross_identity["slack_cloud"], "cross_identity.slack_cloud", errors
            )
        if "center_date" in cross_identity:
            _validate_date(
                cross_identity["center_date"], "cross_identity.center_date", errors
            )

    cross = _validate_object(
        config,
        "cross",
        errors,
        allowed={
            "dst_crs",
            "resolution",
            "out",
            "max_overlap_mae",
            "max_nodata_fraction",
        },
    )
    if cross is not None:
        for key in ("dst_crs", "out"):
            if key in cross and not _non_empty_string(cross[key]):
                errors.append(f"cross.{key} must be non-empty")
        if "resolution" in cross and (
            not _is_number(cross["resolution"]) or cross["resolution"] <= 0
        ):
            errors.append("cross.resolution must be positive")
        if "max_overlap_mae" in cross and (
            not _is_number(cross["max_overlap_mae"])
            or cross["max_overlap_mae"] < 0
        ):
            errors.append("cross.max_overlap_mae must be non-negative")
        if "max_nodata_fraction" in cross:
            _validate_fraction(
                cross["max_nodata_fraction"], "cross.max_nodata_fraction", errors
            )

    territory_scope = _validate_object(
        config, "territory_scope", errors, allowed={"included", "excluded"}
    )
    if territory_scope is not None:
        if not _non_empty_string(territory_scope.get("included")):
            errors.append("territory_scope.included must be non-empty")
        excluded = territory_scope.get("excluded")
        if not isinstance(excluded, list) or any(
            not _non_empty_string(item) for item in excluded
        ):
            errors.append("territory_scope.excluded must be a string array")
        elif len(excluded) != len(set(excluded)):
            errors.append("territory_scope.excluded must not contain duplicates")

    exceptions = config.get("inventory_exceptions")
    if exceptions is not None:
        if not isinstance(exceptions, list):
            errors.append("inventory_exceptions must be an array")
        else:
            seen_exceptions = set()
            for index, exception in enumerate(exceptions):
                field = f"inventory_exceptions[{index}]"
                if not isinstance(exception, dict):
                    errors.append(f"{field} must be an object")
                    continue
                unknown = sorted(set(exception) - {"mgrs", "status", "reason", "evidence"})
                if unknown:
                    errors.append(f"{field} has unsupported fields: {unknown}")
                mgrs = exception.get("mgrs")
                if not _non_empty_string(mgrs) or not MGRS_TILE.fullmatch(mgrs):
                    errors.append(f"{field}.mgrs must be a five-character MGRS tile")
                elif mgrs in seen_exceptions:
                    errors.append(f"duplicate inventory exception: {mgrs}")
                else:
                    seen_exceptions.add(mgrs)
                if exception.get("status") not in {"unavailable", "excluded"}:
                    errors.append(f"{field}.status must be 'unavailable' or 'excluded'")
                if not _non_empty_string(exception.get("reason")):
                    errors.append(f"{field}.reason must be non-empty")
                if "evidence" in exception and not _non_empty_string(exception["evidence"]):
                    errors.append(f"{field}.evidence must be non-empty")
    if errors:
        raise CountryConfigError("; ".join(errors))


def load_country_config(path: Path, *, root: Path = ROOT) -> dict:
    config_path = resolve_root_path(path, root=root)
    try:
        config = json.loads(config_path.read_text())
    except FileNotFoundError as exc:
        raise CountryConfigError(f"country config does not exist: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise CountryConfigError(f"invalid JSON in {config_path}: {exc}") from exc
    if not isinstance(config, dict):
        raise CountryConfigError("country config root must be an object")
    validate_country_config(config)
    config["_config_path"] = str(config_path)
    return config


def configured_path(config: dict, section: str, key: str, *, root: Path = ROOT) -> Path:
    return resolve_root_path(config[section][key], root=root)
