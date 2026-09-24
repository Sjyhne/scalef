import json
from pathlib import Path
from types import SimpleNamespace

from scripts import plan_national_waves as planner
from scripts import run_national_production as national
from scripts import run_national_waves as runner


def _write(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def _country_config(tmp_path: Path, ids: list[str], bboxes: dict) -> Path:
    inventory = _write(tmp_path / "inventory.json", {"mgrs": ids})
    bbox_path = _write(tmp_path / "bboxes.json", bboxes)
    return _write(
        tmp_path / "country.json",
        {
            "schema_version": 1,
            "country": {"name": "Synthetic", "iso_a2": "ZZ", "iso_a3": "ZZZ"},
            "inputs": {"land_outline_geojson": str(tmp_path / "outline.geojson")},
            "inventory": {
                "mgrs_list_json": str(inventory),
                "mgrs_bboxes_json": str(bbox_path),
                "expected_runnable_count": len(ids),
            },
            "paths": {
                "plan_root": str(tmp_path / "production" / "plans"),
                "s2_root": str(tmp_path / "data"),
            },
            "roots": {
                "production": str(tmp_path / "production"),
                "mosaics": str(tmp_path / "mosaics"),
            },
            "cross": {"out": str(tmp_path / "mosaics" / "country.tif")},
        },
    )


def test_geographic_planner_uses_bboxes_not_alphabetical_order(tmp_path):
    ids = ["A0001", "A0002", "A0003", "A0004", "A0005", "A0006"]
    bboxes = {
        "A0001": [0, 50, 1, 51],
        "A0002": [0, 0, 1, 1],
        "A0003": [1, 50, 2, 51],
        "A0004": [1, 0, 2, 1],
        "A0005": [2, 50, 3, 51],
        "A0006": [2, 0, 3, 1],
    }
    config = _country_config(tmp_path, ids, bboxes)

    first = planner.build_wave_plan(config, completed=[], excluded=[], minimum=3, maximum=3)
    second = planner.build_wave_plan(config, completed=[], excluded=[], minimum=3, maximum=3)

    assert [wave["granules"] for wave in first["waves"]] == [
        ["A0002", "A0004", "A0006"],
        ["A0001", "A0003", "A0005"],
    ]
    assert [wave["granules"] for wave in first["waves"]] == [
        wave["granules"] for wave in second["waves"]
    ]


def test_operational_norway_and_denmark_have_complete_balanced_accounting():
    root = Path(__file__).resolve().parents[1]
    completed = ["32VNM", "32VNL", "32VNN", "32VPL", "32VPM", "32VPN"]
    norway = planner.build_wave_plan(
        root / "configs/countries/norway.json",
        completed=completed,
        excluded=[],
    )
    denmark = planner.build_wave_plan(
        root / "configs/countries/denmark.json",
        completed=[],
        excluded=[],
    )

    assert norway["accounting"] == {
        "configured_runnable": 73,
        "completed": 6,
        "explicitly_excluded_runnable": 0,
        "scheduled": 67,
        "inventory_exceptions": 1,
        "final_mosaic_granules": 73,
        "source_total": 74,
    }
    assert norway["inventory_exceptions"][0]["mgrs"] == "33WVL"
    assert all(6 <= wave["n_granules"] <= 9 for wave in norway["waves"])
    assert [wave["n_granules"] for wave in denmark["waves"]] == [7, 7, 6]
    assert denmark["accounting"]["source_total"] == 20


def test_wave_runner_dry_run_uses_separate_states_and_delivery_then_cross(tmp_path):
    ids = [f"A{i:04d}" for i in range(1, 7)]
    bboxes = {mgrs: [i, 0, i + 0.5, 0.5] for i, mgrs in enumerate(ids)}
    config = _country_config(tmp_path, ids, bboxes)
    plan = planner.build_wave_plan(config, completed=[], excluded=[])
    plan_path = _write(tmp_path / "waves.json", plan)
    state_dir = tmp_path / "states"
    log_dir = tmp_path / "logs"

    assert (
        runner.main(
            [
                "--plan",
                str(plan_path),
                "--state-dir",
                str(state_dir),
                "--log-dir",
                str(log_dir),
                "--dry-run",
            ]
        )
        == 0
    )

    state = json.loads((state_dir / "wave_runner.json").read_text())
    wave = state["jobs"]["wave:000"]
    final = state["jobs"]["final:cross"]
    assert wave["child_state"].endswith("wave_000.json")
    assert wave["log"].endswith("wave_000.log")
    assert wave["command"][wave["command"].index("--stages") + 1] == runner.WAVE_STAGES
    assert "cross" not in runner.WAVE_STAGES.split(",")
    assert final["child_state"].endswith("final_cross.json")
    assert final["command"][final["command"].index("--stages") + 1] == "cross"
    assert "--cleanup-delivery-only" not in final["command"]


def test_wave_job_retries_when_child_success_is_not_validated(tmp_path):
    child_state = tmp_path / "child.json"
    state_path = tmp_path / "runner.json"
    state = {"jobs": {}}
    calls = []

    def fake_subprocess(command, **kwargs):
        calls.append(command)
        stages = {} if len(calls) == 1 else {"A0001:delivery": {"status": "success"}}
        _write(
            child_state,
            {
                "status": "success",
                "selected_granules": ["A0001"],
                "stages": stages,
            },
        )
        return SimpleNamespace(returncode=0)

    assert runner.execute_job(
        key="wave:000",
        command=["national"],
        child_state=child_state,
        log_path=tmp_path / "wave.log",
        validate=runner.validate_wave_state,
        granules=["A0001"],
        state=state,
        state_path=state_path,
        retries=1,
        retry_backoff=0,
        dry_run=False,
        runner=fake_subprocess,
        sleeper=lambda _: None,
    )
    assert len(calls) == 2
    assert [attempt["status"] for attempt in state["jobs"]["wave:000"]["attempts"]] == [
        "failed",
        "success",
    ]


def test_cross_stage_never_reuses_country_output_for_subset(tmp_path):
    config_path = _write(
        tmp_path / "country.json",
        {
            "country": "testland",
            "inventory": ["A0001", "A0002"],
            "roots": {"production": str(tmp_path / "production")},
            "cross": {"out": str(tmp_path / "mosaics" / "testland.tif")},
        },
    )
    config = national.load_country_config(config_path)
    full = national.cross_stage(config, config["_granules"])
    subset = national.cross_stage(config, config["_granules"][:1])
    full_out = Path(full.command[full.command.index("--out") + 1])
    subset_out = Path(subset.command[subset.command.index("--out") + 1])

    assert full_out == tmp_path / "mosaics" / "testland.tif"
    assert subset_out != full_out
    assert "_subset_" in subset_out.stem
