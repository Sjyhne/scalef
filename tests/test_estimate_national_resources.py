import json
from pathlib import Path

from scripts import estimate_national_resources as estimator


def _write(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def _config(tmp_path: Path, inventory) -> Path:
    return _write(
        tmp_path / "country.json",
        {
            "country": "testland",
            "roots": {
                "production": str(tmp_path / "production"),
                "data": str(tmp_path / "data"),
                "tiles": str(tmp_path / "data"),
                "mosaics": str(tmp_path / "mosaics"),
                "results": str(tmp_path / "results"),
            },
            "recipe": {"side": 16},
            "production": {"scope": "all", "run_prefix": "test_run"},
            "cross": {"dst_crs": "EPSG:3857", "resolution": 10},
            "inventory": inventory,
        },
    )


def test_exact_estimate_honors_wave_counts_outputs_and_overrides(tmp_path, monkeypatch):
    monkeypatch.setattr(estimator, "ROOT", tmp_path)
    config_path = _config(tmp_path, ["A0001", "A0002", "A0003"])
    for index, mgrs in enumerate(("A0001", "A0002", "A0003")):
        plan = {
            "bbox_wgs84": [10 + index, 60, 11 + index, 61],
            "cells": [
                {"iy": 0, "ix": 0, "n_frames": 2},
                {"iy": 0, "ix": 1, "n_frames": 3},
            ],
            "skipped_no_pass": [{"iy": 1, "ix": 0}],
        }
        _write(tmp_path / "production/plans" / mgrs / "plan.json", plan)
        _write(
            tmp_path / "data" / mgrs / "granule_tiles_lr16_manifest.json",
            {
                "parent": mgrs,
                "tiles": [
                    {"tile_id": f"{mgrs}_t16_y00_x00", "parent": mgrs},
                    {"tile_id": f"{mgrs}_t16_y00_x01", "parent": mgrs},
                ],
            },
        )

    completed = (
        tmp_path
        / "single_samples/A0002/sample/test_run_A0002_t16_y00_x00"
    )
    _write(completed / "metrics.json", {"training_time_seconds": 1})
    _write(completed / "qgis/sr_pred.tif", {"not": "read"})
    state = _write(
        tmp_path / "production/run_state.json",
        {"stages": {"A0002:production": {"status": "running"}}},
    )
    out = tmp_path / "estimate.json"
    argv = [
        "--config",
        str(config_path),
        "--out",
        str(out),
        "--state",
        str(state),
        "--wave-size",
        "1",
        "--wave-index",
        "1",
        "--gpus",
        "2",
        "--gpu-hours-low",
        "1",
        "--gpu-hours-high",
        "2",
    ]

    assert estimator.main(argv) == 0
    first = out.read_bytes()
    assert estimator.main(argv) == 0
    assert out.read_bytes() == first

    result = json.loads(first)
    assert result["inventory"]["selected_granules"] == ["A0002"]
    assert result["totals"]["planned_no_pass_cells"] == 1
    assert result["totals"]["training_target_cells"] == {"low": 2, "high": 2}
    assert result["totals"]["already_complete_training_cells"] == 1
    assert result["totals"]["already_complete_cog_cells"] == 1
    assert result["totals"]["remaining_training_cells"] == {"low": 1, "high": 1}
    assert result["totals"]["serial_gpu_hours"] == {"low": 1.0, "high": 2.0}
    assert result["totals"]["wall_hours_for_gpus"] == {
        "low": 0.5,
        "high": 1.0,
        "gpus": 2,
    }
    assert result["granules"][0]["state_production_status"] == "running"
    assert result["totals"]["final_mosaic"]["pixels"] is not None
    assert result["config"]["file_sha256"]
    assert result["inventory"]["selected_sha256"]
    assert any(item["path"] == str(state) for item in result["artifacts"])


def test_missing_identity_manifest_and_plan_emit_ranges_and_availability_unknowns(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(estimator, "ROOT", tmp_path)
    availability = _write(
        tmp_path / "production/cloud_availability/july2025_pm45/mgrs_availability.json",
        {
            "tiles": [
                {"mgrs_tile": "A0001", "n_scenes": 5, "bbox": [10, 60, 11, 61]},
                {"mgrs_tile": "A0002", "n_scenes": 0, "bbox": [11, 60, 12, 61]},
            ]
        },
    )
    config_path = _config(tmp_path, ["A0001", "A0002"])
    config = json.loads(config_path.read_text())
    config["production"]["scope"] = "identity_changed"
    config["inventory_exceptions"] = [
        {"mgrs": "A0003", "status": "unavailable", "reason": "no scenes"}
    ]
    _write(config_path, config)
    _write(
        tmp_path / "production/plans/A0001/plan.json",
        {
            "cells": [
                {"iy": 0, "ix": 0, "n_frames": 1},
                {"iy": 0, "ix": 1, "n_frames": 1},
                {"iy": 0, "ix": 2, "n_frames": 1},
            ],
            "n_cells_no_pass": 2,
        },
    )
    args = estimator.build_parser().parse_args(
        [
            "--config",
            str(config_path),
            "--out",
            str(tmp_path / "estimate.json"),
            "--unplanned-cells-low",
            "10",
            "--unplanned-cells-high",
            "20",
        ]
    )

    result = estimator.estimate(args)

    first, second = result["granules"]
    assert first["training_target_cells"] == {
        "low": 0,
        "high": 3,
        "basis": "range_identity_changed_manifest_absent",
    }
    assert first["plan"]["planned_cells"] == 5
    assert second["training_target_cells"] == {
        "low": 0,
        "high": 0,
        "basis": "no_scenes_in_availability_artifact",
    }
    assert result["inventory"]["exception_granules"] == 1
    assert result["totals"]["training_target_cells"] == {"low": 0, "high": 3}
    assert any(item["path"] == str(availability) for item in result["artifacts"])
    assert result["assumptions"]["calibration"]["sample_count"] == 0
    assert (
        result["assumptions"]["gpu_hours_per_cell"]["low"]
        < result["assumptions"]["gpu_hours_per_cell"]["high"]
    )


def test_measured_gpu_defaults_and_summary_auto_calibration(tmp_path, monkeypatch):
    monkeypatch.setattr(estimator, "ROOT", tmp_path)
    assert estimator.DEFAULTS["gpu_hours_per_cell"] == [0.018, 0.023]
    config_path = _config(tmp_path, ["A0001"])
    _write(
        tmp_path / "production/plans/A0001/plan.json",
        {"cells": [{"iy": 0, "ix": index, "n_frames": 1} for index in range(4)]},
    )
    _write(
        tmp_path / "data/A0001/granule_tiles_lr16_manifest.json",
        {
            "parent": "A0001",
            "tiles": [
                {"tile_id": f"A0001_t16_y00_x{index:02d}", "parent": "A0001"}
                for index in range(4)
            ],
        },
    )
    summary = _write(
        tmp_path / "results/production_A0001_lr16.json",
        {
            "side": 16,
            "tiles": [
                {"training_time_s": 68.0},
                {"training_time_s": 70.0},
                {"training_time_s": 72.0},
                {"training_time_s": 74.0},
            ],
        },
    )
    args = estimator.build_parser().parse_args(
        ["--config", str(config_path), "--out", str(tmp_path / "estimate.json"), "--gpus", "2"]
    )

    result = estimator.estimate(args)

    calibration = result["assumptions"]["calibration"]
    assert calibration["sample_count"] == 4
    assert calibration["source_files"][0]["path"] == str(summary)
    assert calibration["source_files"][0]["sha256"]
    assert calibration["source_files"][0]["sample_count"] == 4
    assert result["assumptions"]["gpu_hours_per_cell_defaults"] == {
        "low": 0.018,
        "high": 0.023,
    }
    serial = result["totals"]["serial_gpu_hours"]
    wall = result["totals"]["wall_hours_for_gpus"]
    assert wall["low"] == round(serial["low"] / 2, 3)
    assert wall["high"] == round(serial["high"] / 2, 3)

    no_auto = estimator.build_parser().parse_args(
        [
            "--config",
            str(config_path),
            "--out",
            str(tmp_path / "estimate.json"),
            "--no-auto-calibrate",
        ]
    )
    defaults = estimator.estimate(no_auto)
    assert defaults["assumptions"]["gpu_hours_per_cell"] == {
        "low": 0.018,
        "high": 0.023,
    }
