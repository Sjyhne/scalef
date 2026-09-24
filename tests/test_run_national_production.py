import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_national_production as national


def _write(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, (dict, list)):
        path.write_text(json.dumps(value))
    else:
        path.write_text(value)
    return path


def test_country_config_builds_complete_strict_pipeline(tmp_path):
    config_path = _write(
        tmp_path / "country.json",
        {
            "country": "testland",
            "roots": {
                "data": str(tmp_path / "data"),
                "production": str(tmp_path / "production"),
                "tiles": str(tmp_path / "tiles"),
            },
            "inventory": [{"mgrs": "T32VNM", "identity": {"grid": [21, 21]}}],
        },
    )
    config = national.load_country_config(config_path)
    stages = national.stages_for_granule(
        config, config["_granules"][0], gpus=3, gpu_offset=2
    )
    commands = {stage.name: stage.command for stage in stages}

    assert list(commands) == [
        "plan",
        "fetch",
        "tile",
        "identity",
        "production",
        "delivery",
    ]
    assert commands["plan"][1].endswith("plan_national_mgrs.py")
    assert commands["fetch"][1].endswith("fetch_national_mgrs.py")
    assert "--fetch-only" in commands["fetch"]
    assert commands["tile"][1].endswith("make_granule_tiles.py")
    assert "--cell-plan" in commands["tile"]
    assert commands["identity"][1].endswith("plan_identity_icm.py")
    assert commands["production"][commands["production"].index("--gpus") + 1] == "3"
    assert commands["production"][commands["production"].index("--gpu-offset") + 1] == "2"
    assert "--identity-plan" in commands["production"]
    assert commands["production"][commands["production"].index("--manifest") + 1].endswith(
        "manifest_icm_changed.json"
    )
    assert commands["delivery"][1].endswith("mosaic_seam_ramp.py")
    assert "--identity-plan" in commands["delivery"]
    assert "--mosaic" in commands["delivery"]
    assert "--no-preview" in commands["delivery"]
    assert commands["delivery"][commands["delivery"].index("--run-prefix") + 1] == (
        "prod_k4_icm"
    )
    assert commands["delivery"][commands["delivery"].index("--fallback-prefix") + 1] == (
        "prod_k4"
    )
    assert "--allow-missing" not in commands["delivery"]
    assert str(tmp_path / "data" / "32VNM") in commands["fetch"]


def test_country_pipeline_enables_harmonized_delivery_from_config(tmp_path):
    config_path = _write(
        tmp_path / "country.json",
        {
            "country": "testland",
            "roots": {
                "data": str(tmp_path / "data"),
                "production": str(tmp_path / "production"),
                "tiles": str(tmp_path / "tiles"),
            },
            "inventory": [{"mgrs": "32VNM"}],
            "delivery": {
                "ramp_px": 96,
                "harmonize": True,
                "harmonize_strip_px": 192,
                "harmonize_segments": 12,
                "harmonize_regularization": 0.03,
                "harmonize_max_offset": 0.04,
                "max_harmonized_edge_p95": 0.005,
            },
        },
    )
    config = national.load_country_config(config_path)
    command = {
        stage.name: stage.command
        for stage in national.stages_for_granule(
            config, config["_granules"][0], gpus=1, gpu_offset=0
        )
    }["delivery"]

    assert "--harmonize" in command
    assert command[command.index("--harmonize-strip-px") + 1] == "192"
    assert command[command.index("--harmonize-segments") + 1] == "12"
    assert command[command.index("--max-harmonized-edge-p95") + 1] == "0.005"


def test_overlap_pipeline_uses_manifest_identity_and_streaming_feather(tmp_path):
    config_path = _write(
        tmp_path / "country.json",
        {
            "country": "testland",
            "roots": {
                "data": str(tmp_path / "data"),
                "production": str(tmp_path / "production"),
                "tiles": str(tmp_path / "tiles"),
            },
            "inventory": [{"mgrs": "32VNM"}],
            "recipe": {"overlap_frac": 0.125},
            "production": {
                "scope": "all",
                "run_prefix": "prod_k4_ovl12_icm",
                "fallback_prefix": "prod_k4_ovl12_icm",
            },
            "delivery": {
                "harmonize": True,
                "max_same_identity_overlap_p95": 0.006,
                "max_identity_risk_overlap_p95": 0.01,
            },
        },
    )
    config = national.load_country_config(config_path)
    stages = national.stages_for_granule(
        config, config["_granules"][0], gpus=2, gpu_offset=0
    )
    commands = {stage.name: stage.command for stage in stages}

    assert "--overlap-frac" in commands["tile"]
    assert "--manifest-cells" in commands["identity"]
    assert commands["delivery"][1].endswith("mosaic_granule_sr.py")
    assert "--method" in commands["delivery"]
    assert "--harmonize" in commands["delivery"]
    assert "--max-same-identity-overlap-p95" in commands["delivery"]
    assert national.paths_for(config, config["_granules"][0])[
        "manifest"
    ].name.endswith("_ovl12_manifest.json")


def test_overlap_tile_validation_uses_source_plan_coverage(tmp_path):
    plan = _write(
        tmp_path / "plan.json",
        {
            "cells": [
                {"iy": 0, "ix": 0, "n_frames": 1},
                {"iy": 0, "ix": 1, "n_frames": 1},
            ]
        },
    )
    manifest = _write(
        tmp_path / "manifest.json",
        {
            "overlap_frac": 0.125,
            "tiles": [
                {
                    "tile_id": "ovl-a",
                    "iy": 0,
                    "ix": 0,
                    "dates": ["2025-07-12"],
                    "source_plan_cells": [
                        {"iy": 0, "ix": 0},
                        {"iy": 0, "ix": 1},
                    ],
                }
            ],
        },
    )

    national._validate_tile(manifest, plan)


def test_established_country_schema_and_derived_inventory_are_supported(tmp_path):
    inventory = _write(
        tmp_path / "inventory.json",
        {"schema_version": 1, "mgrs": ["32VNM", "32VNN"], "n": 2},
    )
    config_path = _write(
        tmp_path / "norway.json",
        {
            "schema_version": 1,
            "country": {"name": "Norway", "iso_a2": "NO", "iso_a3": "NOR"},
            "inputs": {"land_outline_geojson": str(tmp_path / "norway.geojson")},
            "inventory": {
                "mgrs_list_json": str(inventory),
                "mgrs_bboxes_json": str(tmp_path / "bboxes.json"),
            },
            "paths": {
                "plan_root": str(tmp_path / "plans"),
                "s2_root": str(tmp_path / "s2"),
            },
        },
    )

    config = national.load_country_config(config_path)

    assert [row["mgrs"] for row in config["_granules"]] == ["32VNM", "32VNN"]
    assert config["_paths"]["plans"] == tmp_path / "plans"
    assert config["_paths"]["data"] == tmp_path / "s2"
    assert config["_paths"]["production"] == tmp_path
    assert config["_country_slug"] == "norway"
    assert config["recipe"]["land_mask"] == str(tmp_path / "norway.geojson")
    assert national.paths_for(config, config["_granules"][0])["plan_dir"] == (
        tmp_path / "plans" / "32VNM"
    )


def test_dry_run_records_exact_command_without_execution(tmp_path):
    output = tmp_path / "would-be-output"
    stage = national.Stage(
        key="32VNM:plan",
        name="plan",
        command=["never-run", "--value", "with spaces"],
        expected=(output,),
        validate=lambda: None,
        mgrs="32VNM",
    )
    state_path = tmp_path / "state.json"
    state = {"stages": {}}

    assert national.run_stage(
        stage,
        state=state,
        state_path=state_path,
        dry_run=True,
        resume=True,
        retries=2,
        backoff=1,
        runner=lambda *args, **kwargs: pytest.fail("dry run executed a command"),
    )

    saved = json.loads(state_path.read_text())
    record = saved["stages"]["32VNM:plan"]
    assert record["status"] == "planned"
    assert record["command"] == stage.command
    assert record["command_shell"] == "never-run --value 'with spaces'"
    assert record["expected_outputs"] == [{"path": str(output), "sha256": None}]
    assert record["attempts"] == []


def test_retry_history_and_resume_require_unchanged_hash(tmp_path):
    output = tmp_path / "result.json"
    state_path = tmp_path / "state.json"
    state = {"stages": {}}
    calls = []
    sleeps = []

    def runner(command, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            raise RuntimeError("transient")
        _write(output, {"complete": True})
        return SimpleNamespace(returncode=0)

    stage = national.Stage(
        "x:tile",
        "tile",
        ["tool", "--run"],
        (output,),
        lambda: national._json(output),
        "x",
    )
    assert national.run_stage(
        stage,
        state=state,
        state_path=state_path,
        dry_run=False,
        resume=True,
        retries=2,
        backoff=0.25,
        runner=runner,
        sleeper=sleeps.append,
    )
    assert len(calls) == 2
    assert sleeps == [0.25]
    record = state["stages"]["x:tile"]
    assert [attempt["status"] for attempt in record["attempts"]] == ["failed", "success"]
    assert record["expected_outputs"][0]["sha256"] == national.sha256_file(output)

    assert national.run_stage(
        stage,
        state=state,
        state_path=state_path,
        dry_run=False,
        resume=True,
        retries=0,
        backoff=0,
        runner=lambda *args, **kwargs: pytest.fail("unchanged output was rerun"),
    )
    assert state["stages"]["x:tile"]["status"] == "resumed"

    output.write_text('{"complete": false}')
    reruns = []

    def rerun(command, **kwargs):
        reruns.append(command)
        _write(output, {"complete": True})
        return SimpleNamespace(returncode=0)

    assert national.run_stage(
        stage,
        state=state,
        state_path=state_path,
        dry_run=False,
        resume=True,
        retries=0,
        backoff=0,
        runner=rerun,
    )
    assert reruns == [["tool", "--run"]]


def test_resume_adopts_valid_outputs_without_prior_state(tmp_path):
    output = _write(tmp_path / "result.json", {"complete": True})
    state = {"stages": {}}
    stage = national.Stage(
        "x:plan",
        "plan",
        ["never-run"],
        (output,),
        lambda: national._json(output),
        "x",
    )

    assert national.run_stage(
        stage,
        state=state,
        state_path=tmp_path / "state.json",
        dry_run=False,
        resume=True,
        retries=0,
        backoff=0,
        runner=lambda *args, **kwargs: pytest.fail("adopted output was rerun"),
    )
    record = state["stages"]["x:plan"]
    assert record["status"] == "adopted"
    assert record["expected_outputs"][0]["sha256"] == national.sha256_file(output)


def test_production_scope_all_uses_full_icm_manifest_and_same_delivery_prefix(tmp_path):
    config_path = _write(
        tmp_path / "country.json",
        {
            "country": "testland",
            "roots": {"production": str(tmp_path / "production")},
            "production": {
                "scope": "all",
                "run_prefix": "fresh_icm",
            },
            "inventory": ["32VNM"],
        },
    )
    config = national.load_country_config(config_path)
    stages = national.stages_for_granule(
        config, config["_granules"][0], gpus=1, gpu_offset=0
    )
    commands = {stage.name: stage.command for stage in stages}

    assert commands["production"][commands["production"].index("--manifest") + 1].endswith(
        "manifest_icm_all.json"
    )
    assert commands["delivery"][commands["delivery"].index("--run-prefix") + 1] == (
        "fresh_icm"
    )
    assert commands["delivery"][commands["delivery"].index("--fallback-prefix") + 1] == (
        "fresh_icm"
    )


def test_granule_subset_and_waves_are_deterministic_and_digest_scoped(tmp_path):
    inventory = [{"mgrs": value} for value in ("A0001", "A0002", "A0003", "A0004")]
    wave = national.select_granules(
        inventory,
        requested="A0004,A0002,A0003",
        wave_size=2,
        wave_index=0,
    )
    assert [row["mgrs"] for row in wave] == ["A0002", "A0003"]
    next_wave = national.select_granules(
        inventory,
        requested=None,
        wave_size=2,
        wave_index=1,
    )
    assert [row["mgrs"] for row in next_wave] == ["A0003", "A0004"]

    config_path = _write(
        tmp_path / "country.json",
        {"country": "x", "inventory": ["A0001", "A0002", "A0003", "A0004"]},
    )
    config = national.load_country_config(config_path)
    assert national.config_digest(config, wave) != national.config_digest(config, next_wave)


def test_cross_identity_stage_builds_deterministic_input_and_cli(tmp_path):
    config_path = _write(
        tmp_path / "country.json",
        {
            "country": "testland",
            "roots": {"production": str(tmp_path / "production")},
            "cross_identity": {
                "enabled": True,
                "audit_only": True,
                "common_crs": "EPSG:3035",
                "adjacency_tolerance_m": 2.5,
                "slack_cloud": 0.04,
                "center_date": "2025-07-16",
            },
            "inventory": ["32VNN", "32VNM"],
        },
    )
    config = national.load_country_config(config_path)
    stage = national.cross_identity_stage(config, config["_granules"])
    assert stage is not None

    stage.prepare()
    payload = json.loads(Path(stage.command[stage.command.index("--inputs-manifest") + 1]).read_text())

    assert stage.name == "cross_identity"
    assert [row["mgrs"] for row in payload["granules"]] == ["32VNM", "32VNN"]
    assert stage.command[1].endswith("plan_cross_mgrs_identity.py")
    assert "--audit-only" in stage.command
    assert stage.command[stage.command.index("--common-crs") + 1] == "EPSG:3035"
    assert stage.command[stage.command.index("--adjacency-tolerance-m") + 1] == "2.5"
    assert stage.command[stage.command.index("--slack-cloud") + 1] == "0.04"


def test_cross_identity_revision_routes_production_and_audit_only_does_not(tmp_path):
    config_path = _write(
        tmp_path / "country.json",
        {
            "country": "testland",
            "roots": {"production": str(tmp_path / "production")},
            "cross_identity": {"enabled": True, "audit_only": False},
            "inventory": ["32VNM", "32VNN"],
        },
    )
    config = national.load_country_config(config_path)
    row = config["_granules"][0]
    revised = national.effective_identity_paths(config, row, config["_granules"])
    stages = national.stages_for_granule(
        config,
        row,
        gpus=1,
        gpu_offset=0,
        granules=config["_granules"],
    )
    commands = {stage.name: stage.command for stage in stages}
    assert commands["production"][commands["production"].index("--manifest") + 1] == str(
        revised["identity_changed_manifest"]
    )
    assert commands["delivery"][commands["delivery"].index("--identity-plan") + 1] == str(
        revised["identity"]
    )

    config["cross_identity"]["audit_only"] = True
    original = national.paths_for(config, row)
    audit_stages = national.stages_for_granule(
        config,
        row,
        gpus=1,
        gpu_offset=0,
        granules=config["_granules"],
    )
    audit_commands = {stage.name: stage.command for stage in audit_stages}
    assert audit_commands["production"][
        audit_commands["production"].index("--manifest") + 1
    ] == str(original["identity_changed_manifest"])
    assert audit_commands["delivery"][
        audit_commands["delivery"].index("--identity-plan") + 1
    ] == str(original["identity"])


def test_cross_identity_materializes_revised_all_and_changed_manifests(tmp_path):
    config_path = _write(
        tmp_path / "country.json",
        {
            "country": "testland",
            "roots": {
                "production": str(tmp_path / "production"),
                "data": str(tmp_path / "data"),
            },
            "cross_identity": {"enabled": True, "audit_only": False},
            "inventory": ["32VNM", "32VNN"],
        },
    )
    config = national.load_country_config(config_path)
    layout = national._cross_identity_layout(config, config["_granules"])
    for index, row in enumerate(config["_granules"]):
        paths = national.paths_for(config, row)
        tile_id = f"{row['mgrs']}_t512_y00_x00"
        _write(
            paths["manifest"],
            {
                "parent": row["mgrs"],
                "tiles": [
                    {
                        "tile_id": tile_id,
                        "parent": row["mgrs"],
                        "iy": 0,
                        "ix": 0,
                    }
                ],
            },
        )
        before = "2025-07-12"
        after = "2025-07-18" if index == 0 else before
        _write(
            layout["out"] / row["mgrs"] / "identity_plan.json",
            {
                "assignment": {tile_id: after},
                "independent": {tile_id: before},
                "cells": [{"tile_id": tile_id, "icm_date": after}],
            },
        )

    national._write_revised_manifests(config, config["_granules"])

    first = national.effective_identity_paths(
        config, config["_granules"][0], config["_granules"]
    )
    all_manifest = json.loads(first["identity_manifest"].read_text())
    changed_manifest = json.loads(first["identity_changed_manifest"].read_text())
    assert all_manifest["tiles"][0]["force_base_date"] == "2025-07-18"
    assert [tile["tile_id"] for tile in changed_manifest["tiles"]] == [
        "32VNM_t512_y00_x00"
    ]
    second = national.effective_identity_paths(
        config, config["_granules"][1], config["_granules"]
    )
    assert json.loads(second["identity_changed_manifest"].read_text())["tiles"] == []


def test_main_executes_stage_major_with_cross_identity_barrier(tmp_path, monkeypatch):
    config_path = _write(
        tmp_path / "country.json",
        {
            "country": "testland",
            "roots": {"production": str(tmp_path / "production")},
            "cross_identity": {"enabled": True, "audit_only": True},
            "inventory": ["32VNM", "32VNN"],
        },
    )
    order = []

    def fake_run(stage, **kwargs):
        order.append((stage.name, stage.mgrs))
        return True

    monkeypatch.setattr(national, "run_stage", fake_run)
    result = national.main(
        [
            "--config",
            str(config_path),
            "--state",
            str(tmp_path / "state.json"),
            "--dry-run",
        ]
    )

    assert result == 0
    assert order == [
        ("plan", "32VNM"),
        ("plan", "32VNN"),
        ("fetch", "32VNM"),
        ("fetch", "32VNN"),
        ("tile", "32VNM"),
        ("tile", "32VNN"),
        ("identity", "32VNM"),
        ("identity", "32VNN"),
        ("cross_identity", None),
        ("production", "32VNM"),
        ("production", "32VNN"),
        ("delivery", "32VNM"),
        ("delivery", "32VNN"),
        ("cross", None),
    ]


def test_delivery_validation_rejects_any_silent_omission(tmp_path):
    manifest = _write(
        tmp_path / "manifest.json",
        {"parent": "32VNM", "tiles": [{"tile_id": "a"}, {"tile_id": "b"}]},
    )
    mosaic = _write(tmp_path / "delivery.tif", "pixels")
    _write(
        mosaic.with_suffix(".tif.json"),
        {"n_missing": 1, "mosaic": {"n_sources": 1}},
    )
    with pytest.raises(ValueError, match="missing tiles"):
        national._validate_delivery(mosaic, manifest)


def test_seam_ramp_delivery_validates_tile_count_ramp_and_date_cuts(tmp_path):
    manifest = _write(
        tmp_path / "manifest.json",
        {"parent": "32VNM", "tiles": [{"tile_id": "a"}, {"tile_id": "b"}]},
    )
    mosaic = _write(tmp_path / "delivery.tif", "pixels")
    sidecar = mosaic.with_suffix(".tif.json")
    _write(
        sidecar,
        {"n_tiles": 2, "ramp_px": 128, "date_cuts_only": True},
    )
    national._validate_delivery(mosaic, manifest, ramp_px=128)

    _write(sidecar, {"n_tiles": 2, "ramp_px": 64, "date_cuts_only": True})
    with pytest.raises(ValueError, match="ramp_px"):
        national._validate_delivery(mosaic, manifest, ramp_px=128)


def test_harmonized_delivery_requires_metrics_and_passing_gate(tmp_path):
    manifest = _write(
        tmp_path / "manifest.json",
        {"parent": "32VNM", "tiles": [{"tile_id": "a"}, {"tile_id": "b"}]},
    )
    mosaic = _write(tmp_path / "delivery.tif", "pixels")
    sidecar = mosaic.with_suffix(".tif.json")
    record = {
        "n_tiles": 2,
        "ramp_px": 128,
        "date_cuts_only": False,
        "ramp_scope": "identity_risk_edges_after_global_harmonization",
        "harmonization": {
            "metrics": {"identity_risk_edges_after_global": {"p95": 0.003}}
        },
        "qa": {"passed": True},
    }
    _write(sidecar, record)

    national._validate_delivery(
        mosaic,
        manifest,
        ramp_px=128,
        require_harmonization=True,
        max_harmonized_edge_p95=0.005,
    )

    record["harmonization"]["metrics"]["identity_risk_edges_after_global"]["p95"] = 0.008
    _write(sidecar, record)
    with pytest.raises(ValueError, match="exceeds"):
        national._validate_delivery(
            mosaic,
            manifest,
            ramp_px=128,
            require_harmonization=True,
            max_harmonized_edge_p95=0.005,
        )


def test_overlap_delivery_validates_pairwise_qa(tmp_path):
    manifest = _write(
        tmp_path / "manifest.json",
        {"parent": "32VNM", "tiles": [{"tile_id": "a"}, {"tile_id": "b"}]},
    )
    mosaic = _write(tmp_path / "delivery.tif", "pixels")
    sidecar = mosaic.with_suffix(".tif.json")
    record = {
        "n_missing": 0,
        "merge_method": "feather",
        "mosaic": {"n_sources": 2},
        "harmonization": {"metrics": {}},
        "overlap_qa": {
            "same_identity": {"p95": 0.004},
            "identity_risk": {"p95": 0.008},
        },
        "qa": {"passed": True},
    }
    _write(sidecar, record)

    national._validate_delivery(
        mosaic,
        manifest,
        require_harmonization=True,
        max_same_identity_overlap_p95=0.006,
        max_identity_risk_overlap_p95=0.01,
    )
    record["overlap_qa"]["same_identity"]["p95"] = 0.007
    _write(sidecar, record)
    with pytest.raises(ValueError, match="same-identity"):
        national._validate_delivery(
            mosaic,
            manifest,
            require_harmonization=True,
            max_same_identity_overlap_p95=0.006,
        )


def test_cross_validation_rejects_delivery_source_changed_after_mosaic(tmp_path):
    source = _write(tmp_path / "source.tif", "source")
    output = _write(tmp_path / "national.tif", "national")
    stat = source.stat()
    sidecar = output.with_suffix(".tif.json")
    _write(
        sidecar,
        {
            "sources": [str(source.resolve())],
            "source_details": [
                {
                    "path": str(source.resolve()),
                    "size_bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            ],
        },
    )
    national._validate_cross(output, [source])

    source.write_text("changed source")
    with pytest.raises(ValueError, match="source size changed"):
        national._validate_cross(output, [source])


def test_cleanup_requires_validated_delivery_and_only_removes_qgis(tmp_path, monkeypatch):
    monkeypatch.setattr(national, "ROOT", tmp_path)
    manifest = _write(
        tmp_path / "stack" / "manifest_icm_all.json",
        {
            "parent": "32VNM",
            "tiles": [{"tile_id": "cell", "iy": 0, "ix": 0}],
        },
    )
    _write(tmp_path / "stack" / "identity_plan.json", {"assignment": {}})
    mosaic = _write(tmp_path / "mosaics" / "32VNM.tif", "pixels")
    _write(
        mosaic.with_suffix(".tif.json"),
        {"n_missing": 0, "mosaic": {"n_sources": 1}},
    )
    qgis = (
        tmp_path
        / "single_samples"
        / "32VNM"
        / "sample"
        / "prod_k4_cell"
        / "qgis"
    )
    _write(qgis / "sr_pred.tif", "tile")
    keep = _write(qgis.parent / "metrics.json", "{}")
    config = {
        "_paths": {
            "production": tmp_path / "production",
            "plans": tmp_path / "production" / "plans",
            "data": tmp_path / "data",
            "tiles": tmp_path / "tiles",
            "mosaics": tmp_path / "mosaics",
            "results": tmp_path / "results",
        },
        "recipe": {},
        "identity": {},
        "production": {"run_prefix": "prod_k4"},
        "delivery": {},
    }
    row = {
        "mgrs": "32VNM",
        "manifest": str(manifest),
        "identity_dir": str(tmp_path / "stack"),
        "delivery_mosaic": str(mosaic),
    }
    state = {"stages": {}}

    national.cleanup_delivery_sources(
        config,
        row,
        state=state,
        state_path=tmp_path / "state.json",
        dry_run=False,
    )

    assert not qgis.exists()
    assert keep.is_file()
    assert state["stages"]["32VNM:cleanup"]["removed_count"] == 1
