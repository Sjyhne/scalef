from argparse import Namespace
import json
from pathlib import Path

import numpy as np

from scripts.compare_norway_season_windows import _probe_day_stats
from scripts.run_norway_national_demo import _run_stage, commands_for_tile
from scripts.select_norway_mgrs_block import (
    build_adjacency,
    build_manifest,
    score_records,
    select_connected,
)


def _season():
    tiles = []
    for y in range(3):
        for x in range(3):
            tiles.append(
                {
                    "mgrs_tile": f"G{x}{y}",
                    "clear_days": 10 - x - y,
                    "bbox": [x, y, x + 1, y + 1],
                }
            )
    return {
        "recommendation": {"top": "summer"},
        "stac": {
            "windows": [
                {
                    "name": "summer",
                    "date_range": "2025-06-01/2025-08-31",
                    "tiles": tiles,
                }
            ]
        },
    }


def _heat():
    return [
        {
            "mgrs_tile": f"G{x}{y}",
            "grid_x": x,
            "grid_y": y,
            "bbox": [x, y, x + 1, y + 1],
            "mainland_cells_total": 100,
            "mainland_cells_ge_min_clear": 90 - 5 * (x + y),
            "clear_count_mean_mainland": 8.0,
        }
        for y in range(3)
        for x in range(3)
    ]


def test_block_selection_is_deterministic_connected_and_auditable():
    first = build_manifest(
        season_payload=_season(),
        window_name=None,
        heatmap_rows=_heat(),
        count=6,
        mode="block",
        min_clear=6,
        adjacency_factor=1.65,
    )
    second = build_manifest(
        season_payload=_season(),
        window_name=None,
        heatmap_rows=list(reversed(_heat())),
        count=6,
        mode="block",
        min_clear=6,
        adjacency_factor=1.65,
    )
    assert first["selected_mgrs"] == second["selected_mgrs"]
    assert len(first["selected_mgrs"]) == 6
    assert all(row["reasons"] for row in first["tiles"])
    selected = set(first["selected_mgrs"])
    seen = {next(iter(selected))}
    while True:
        grown = seen | {
            neighbor
            for tile in seen
            for neighbor in first["adjacency"][tile]
            if neighbor in selected
        }
        if grown == seen:
            break
        seen = grown
    assert seen == selected


def test_connected_selector_rejects_disconnected_requested_size():
    season = {row["mgrs_tile"]: row for row in _season()["stac"]["windows"][0]["tiles"]}
    rows = score_records(_heat()[:2], season, min_clear=6)
    graph = {row["mgrs_tile"]: set() for row in rows}
    try:
        select_connected(rows, graph, count=2, mode="connected")
    except ValueError as exc:
        assert "no connected set" in str(exc)
    else:
        raise AssertionError("expected disconnected selection to fail")


def test_grid_adjacency_is_four_neighbor_only():
    graph = build_adjacency(_heat())
    assert graph["G11"] == {"G01", "G10", "G12", "G21"}
    assert "G11" not in graph["G00"]


def test_scl_probe_reports_shadow_snow_vegetation_and_ndvi():
    scl = np.array([[4, 3], [11, 5]], dtype=np.uint8)
    red = np.full((2, 2), 0.2, dtype=np.float32)
    nir = np.full((2, 2), 0.6, dtype=np.float32)
    stats = _probe_day_stats(scl, red, nir, include_shadow=True)
    assert stats["shadow_frac"] == 0.25
    assert stats["snow_frac"] == 0.25
    assert stats["veg_frac"] == 0.25
    assert stats["cloud_frac"] == 0.25
    assert abs(stats["ndvi_clear_land_mean"] - 0.5) < 1e-6


def test_driver_reuses_pipeline_scripts_and_exact_window(tmp_path):
    args = Namespace(
        date_range="2025-06-01/2025-08-31",
        side=512,
        num_samples=8,
        cloud_method="omnicloudmask",
        max_cloud_frac=0.15,
        min_valid_frac=0.85,
        max_stac_cloud=60.0,
        cloud_device="cpu",
        include_shadow=True,
        min_land_frac=0.1,
        heatmap_dir=tmp_path,
        min_clear=6,
        iters=5000,
        gpus=2,
        gpu_offset=1,
    )
    stages = commands_for_tile(
        {"mgrs_tile": "32VNM", "bbox": [9.0, 59.0, 12.0, 61.0]},
        args,
    )
    names = [name for name, _, _ in stages]
    assert names == ["fetch", "tile", "train", "granule_mosaic"]
    commands = {name: command for name, command, _ in stages}
    assert commands["fetch"][1].endswith("fetch_s2_revisits.py")
    assert commands["fetch"][commands["fetch"].index("--start-date") + 1] == "2025-06-01"
    assert "--include-shadow" in commands["fetch"]
    assert commands["tile"][1].endswith("make_granule_tiles.py")
    assert "--mainland-only" in commands["tile"]
    assert commands["train"][1].endswith("run_production.py")
    assert "--skip-existing" in commands["train"]
    assert commands["granule_mosaic"][1].endswith("mosaic_granule_sr.py")


def test_dry_run_records_command_without_executing(tmp_path):
    log_path = tmp_path / "run.json"
    log = {"stages": {}}
    ok = _run_stage(
        key="demo:fetch",
        command=["definitely-not-an-executable", "--flag"],
        expected=tmp_path / "missing",
        log=log,
        log_path=log_path,
        dry_run=True,
        resume=True,
    )
    assert ok
    saved = json.loads(log_path.read_text())
    assert saved["stages"]["demo:fetch"]["status"] == "planned"
    assert saved["stages"]["demo:fetch"]["command"] == [
        "definitely-not-an-executable",
        "--flag",
    ]
