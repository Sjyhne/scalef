from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import run_national_production
from scripts.country_config import CountryConfigError, load_country_config, validate_country_config
from scripts.derive_country_mgrs_inventory import (
    InventoryInputError,
    derive_inventory,
    load_outline,
    run,
)


def _feature_collection(ring):
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            }
        ],
    }


def _config(tmp_path: Path, outline: Path) -> dict:
    return {
        "schema_version": 1,
        "country": {"name": "Synthetic", "iso_a2": "ZZ", "iso_a3": "ZZZ"},
        "inputs": {"land_outline_geojson": str(outline)},
        "inventory": {
            "mgrs_list_json": str(tmp_path / "mgrs.json"),
            "mgrs_bboxes_json": str(tmp_path / "bboxes.json"),
            "min_tile_land_fraction": 0.0,
        },
        "paths": {
            "plan_root": str(tmp_path / "plans"),
            "s2_root": str(tmp_path / "revisits"),
        },
    }


def test_synthetic_polygon_inventory_is_deterministic_and_has_land_bbox(tmp_path):
    # Entirely within one Denmark-area 100 km square; this is synthetic geometry,
    # not a statement about Denmark's national inventory.
    ring = [[12.40, 55.60], [12.45, 55.60], [12.45, 55.65], [12.40, 55.65], [12.40, 55.60]]
    outline = tmp_path / "outline.geojson"
    outline.write_text(json.dumps(_feature_collection(ring)))
    polygons, _ = load_outline(outline)

    first = derive_inventory(polygons)
    second = derive_inventory(polygons)

    assert first == second
    ids, bboxes, fractions = first
    assert len(ids) == 1
    bbox = bboxes[ids[0]]
    assert bbox[0] <= 12.40 < 12.45 <= bbox[2]
    assert bbox[1] <= 55.60 < 55.65 <= bbox[3]
    assert 0 < fractions[ids[0]] < 1


def test_run_writes_reproducible_outputs_from_supplied_outline(tmp_path):
    ring = [[9.9, 59.9], [10.0, 59.9], [10.0, 60.0], [9.9, 60.0], [9.9, 59.9]]
    outline = tmp_path / "outline.geojson"
    outline.write_text(json.dumps(_feature_collection(ring)))
    config_path = tmp_path / "country.json"
    config_path.write_text(json.dumps(_config(tmp_path, outline)))

    result = run(config_path)

    inventory = json.loads((tmp_path / "mgrs.json").read_text())
    bboxes = json.loads((tmp_path / "bboxes.json").read_text())
    assert result["n"] == inventory["n"] == len(inventory["mgrs"])
    assert sorted(bboxes) == inventory["mgrs"]
    assert inventory["provenance"]["source_sha256"]
    assert inventory["provenance"]["method"] == "offline_utm_100km_polygon_clip_v1"


def test_missing_outline_is_a_preflight_failure_with_blocker(tmp_path):
    missing = tmp_path / "authoritative-outline.geojson"
    config_path = tmp_path / "country.json"
    config_path.write_text(json.dumps(_config(tmp_path, missing)))
    blocker_path = tmp_path / "INPUT_REQUIRED.json"

    with pytest.raises(InventoryInputError, match="authoritative-outline.geojson"):
        run(config_path, blocker_out=blocker_path)

    blocker = json.loads(blocker_path.read_text())
    assert blocker["status"] == "blocked"
    assert blocker["blocker"] == "missing_authoritative_country_land_outline_geojson"
    assert blocker["required_input"]["path"] == str(missing)


def test_config_validation_rejects_missing_required_paths(tmp_path):
    config_path = tmp_path / "invalid.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "country": {"name": "Synthetic", "iso_a2": "ZZ", "iso_a3": "ZZZ"},
                "inputs": {},
                "inventory": {},
                "paths": {},
            }
        )
    )
    with pytest.raises(CountryConfigError, match="land_outline_geojson"):
        load_country_config(config_path)


def test_outline_validation_rejects_unclosed_polygon(tmp_path):
    outline = tmp_path / "bad.geojson"
    outline.write_text(
        json.dumps(_feature_collection([[12.0, 55.0], [12.1, 55.0], [12.1, 55.1], [12.0, 55.1]]))
    )
    with pytest.raises(InventoryInputError, match="closed"):
        load_outline(outline)


@pytest.mark.parametrize("country", ["norway", "denmark"])
def test_operational_country_configs_validate_and_load(country):
    root = Path(__file__).resolve().parent.parent
    config_path = root / "configs" / "countries" / f"{country}.json"

    validated = load_country_config(config_path)
    operational = run_national_production.load_country_config(config_path)

    assert operational["_granules"]
    assert len(operational["_granules"]) == validated["inventory"]["expected_runnable_count"]
    assert operational["territory_scope"]["included"]
    assert operational["_paths"]["tiles"] == root / validated["roots"]["tiles"]
    assert operational["_paths"]["production"] == root / validated["roots"]["production"]


def test_norway_curated_inventory_is_distinct_from_diagnostic_derivation():
    root = Path(__file__).resolve().parent.parent
    config = load_country_config(root / "configs/countries/norway.json")
    operational_path = root / config["inventory"]["mgrs_list_json"]
    diagnostic_path = root / config["inventory"]["diagnostic_mgrs_list_json"]
    operational = json.loads(operational_path.read_text())
    diagnostic = json.loads(diagnostic_path.read_text())
    exceptions = config["inventory_exceptions"]

    assert config["inventory"]["source_role"] == "curated_operational"
    assert operational["n"] == 73
    assert diagnostic["n"] == 62
    assert config["inventory"]["expected_source_count"] == 74
    assert [row["mgrs"] for row in exceptions] == ["33WVL"]
    assert "33WVL" not in operational["mgrs"]
    assert operational["inventory_accounting"]["source_n"] == 74
    assert operational["inventory_accounting"]["runnable_n"] + len(exceptions) == 74


def test_denmark_natural_earth_provenance_and_scope_are_explicit():
    root = Path(__file__).resolve().parent.parent
    config = load_country_config(root / "configs/countries/denmark.json")
    outline = json.loads((root / config["inputs"]["land_outline_geojson"]).read_text())
    provenance = config["inputs"]["land_outline_provenance"]

    assert config["inventory"]["source_role"] == "outline_derived"
    assert provenance["dataset"] == "Natural Earth Admin-0 Countries"
    assert provenance["source_sha256"] == outline["source"]["download_sha256"]
    assert config["territory_scope"]["excluded"] == outline["scope"]["excluded"]


def test_operational_section_validation_preserves_safety(tmp_path):
    config = _config(tmp_path, tmp_path / "outline.geojson")
    config.update(
        {
            "recipe": {
                "start_date": "2025-08-01",
                "date": "2025-07-15",
                "end_date": "2025-08-29",
                "max_cloud_frac": 1.5,
            },
            "production": {"scope": "unsafe_unknown_scope"},
            "inventory_exceptions": [
                {"mgrs": "INVALID", "status": "ignored", "reason": ""}
            ],
        }
    )

    with pytest.raises(
        CountryConfigError,
        match="recipe dates.*max_cloud_frac.*production.scope.*inventory_exceptions",
    ):
        validate_country_config(config)


def test_operational_loader_enforces_inventory_accounting(tmp_path):
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(json.dumps({"mgrs": ["32VNM"]}))
    config = _config(tmp_path, tmp_path / "outline.geojson")
    config["inventory"].update(
        {
            "mgrs_list_json": str(inventory_path),
            "expected_runnable_count": 1,
        }
    )
    config["inventory_exceptions"] = [
        {"mgrs": "32VNM", "status": "unavailable", "reason": "synthetic conflict"}
    ]
    config_path = tmp_path / "country.json"
    config_path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match="must not also be runnable"):
        run_national_production.load_country_config(config_path)

    config["inventory_exceptions"][0]["mgrs"] = "33WVL"
    config["inventory"]["expected_runnable_count"] = 2
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="expected_runnable_count"):
        run_national_production.load_country_config(config_path)
