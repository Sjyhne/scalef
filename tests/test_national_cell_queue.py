from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np

from scripts.national_cell_queue import (
    UINT16_NODATA,
    build_cell_plan,
    cell_lookup,
    date_span_days,
    filter_frames_for_cell,
    filter_stac_items_for_plan,
    frame_day,
    load_dates_file,
    load_plan_stac_ids,
    rank_day_indices,
)


def test_date_span_is_chronological_not_rank_order():
    days = ["2025-07-15", "2025-05-31", "2025-08-29"]
    passed = np.array([True, True, True])
    idxs = rank_day_indices(days, passed, center=date(2025, 7, 15), max_frames=16)
    ranked = [days[i] for i in idxs]
    assert ranked[0] == "2025-07-15"
    assert date_span_days(ranked) == 90
    assert date_span_days([]) == 0
    assert date_span_days(["2025-07-15"]) == 0


def test_rank_keeps_thin_stack_and_caps_at_16():
    days = [f"2025-07-{d:02d}" for d in range(1, 21)]  # 20 days
    passed = np.ones(len(days), dtype=bool)
    idxs = rank_day_indices(days, passed, center=date(2025, 7, 15), max_frames=16)
    selected = [days[i] for i in idxs]
    assert len(selected) == 16
    assert "2025-07-15" in selected
    assert "2025-07-01" not in selected  # farther than the closest 16
    assert "2025-07-20" in selected

    thin = np.zeros(len(days), dtype=bool)
    thin[0] = True
    thin[3] = True
    thin[10] = True
    idxs3 = rank_day_indices(days, thin, center=date(2025, 7, 15), max_frames=16)
    assert len(idxs3) == 3


def test_plan_union_and_qa_exclude_ocean_and_empty_cells():
    days = ["2025-07-01", "2025-07-15", "2025-07-20", "2025-08-01"]
    n_days, n_y, n_x = 4, 2, 2
    passed = np.zeros((n_days, n_y, n_x), dtype=bool)
    cloud = np.full((n_days, n_y, n_x), 0.04, dtype=np.float32)
    snow = np.full((n_days, n_y, n_x), 0.01, dtype=np.float32)
    # land cell (0,0): only 2 pass days (thin stack kept)
    passed[[1, 2], 0, 0] = True
    snow[1, 0, 0] = 0.02
    snow[2, 0, 0] = 0.08
    cloud[1, 0, 0] = 0.10
    cloud[2, 0, 0] = 0.05
    # land cell (0,1): different two days (shows union is per-cell, not granule 16)
    passed[[0, 3], 0, 1] = True
    # ocean cell (1,0): would pass but must be excluded from union
    passed[:, 1, 0] = True
    land = np.array([[True, True], [False, True]], dtype=bool)
    # land cell (1,1): never passes
    plan = build_cell_plan(
        days,
        passed,
        cloud,
        snow,
        land,
        center="2025-07-15",
        max_frames=16,
        side=512,
    )
    assert plan["n_land_cells"] == 3
    assert plan["n_cells_with_frames"] == 2
    assert plan["n_cells_no_pass"] == 1
    assert plan["union_dates"] == ["2025-07-01", "2025-07-15", "2025-07-20", "2025-08-01"]
    lookup = cell_lookup(plan)
    thin = lookup[(0, 0)]
    assert thin["dates"] == ["2025-07-15", "2025-07-20"]
    assert thin["n_frames"] == 2
    assert thin["date_span_days"] == 5
    assert abs(thin["mean_snow_used"] - 0.05) < 1e-6
    assert abs(thin["mean_cloud_used"] - 0.075) < 1e-6
    other = lookup[(0, 1)]
    assert other["dates"] == ["2025-07-01", "2025-08-01"]
    qa = plan["qa"]
    assert qa["n_frames"][1, 0] == UINT16_NODATA  # ocean
    assert qa["n_frames"][1, 1] == 0  # land, no pass
    assert qa["n_frames"][0, 0] == 2


def test_filter_frames_for_cell_and_dates_file(tmp_path: Path):
    frames = [
        {"path": "001_20250701.tif", "datetime": "2025-07-01T10:00:00+00:00"},
        {"path": "002_20250715.tif", "datetime": "2025-07-15T10:00:00+00:00"},
        {"path": "003_20250801.tif", "datetime": "2025-08-01T10:00:00+00:00"},
    ]
    kept = filter_frames_for_cell(frames, ["2025-07-15", "2025-07-01"])
    assert [frame_day(f) for f in kept] == ["2025-07-01", "2025-07-15"]

    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps({"union_dates": ["2025-07-15", "2025-07-01"], "mgrs_tile": "32VNM"})
    )
    assert load_dates_file(plan_path) == ["2025-07-01", "2025-07-15"]
    list_path = tmp_path / "dates.json"
    list_path.write_text(json.dumps(["2025-07-20"]))
    assert load_dates_file(list_path) == ["2025-07-20"]


class _Item:
    def __init__(self, item_id: str, day: str, cloud: float):
        from datetime import datetime, timezone

        self.id = item_id
        self.datetime = datetime.fromisoformat(day).replace(tzinfo=timezone.utc)
        self.properties = {"eo:cloud_cover": cloud}


def test_load_plan_stac_ids_and_pin(tmp_path: Path):
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "union_dates": ["2025-07-15", "2025-07-12"],
                "days_scored_items": [
                    {"date": "2025-07-15", "stac_id": "ITEM_CLEAR"},
                    {"date": "2025-07-12", "stac_id": "ITEM_12"},
                ],
            }
        )
    )
    ids = load_plan_stac_ids(plan_path)
    assert ids["2025-07-15"] == "ITEM_CLEAR"
    cloudy = _Item("ITEM_CLOUDY", "2025-07-15", 64.0)
    clear = _Item("ITEM_CLEAR", "2025-07-15", 12.0)
    other = _Item("ITEM_12", "2025-07-12", 5.0)
    # First-hit would keep ITEM_CLOUDY; pin keeps ITEM_CLEAR.
    selected, missing = filter_stac_items_for_plan(
        [cloudy, clear, other],
        {"2025-07-15", "2025-07-12"},
        ids,
    )
    assert {it.id for it in selected} == {"ITEM_CLEAR", "ITEM_12"}
    assert missing == []
    # No ids → lowest scene cloud per day.
    selected2, missing2 = filter_stac_items_for_plan(
        [cloudy, clear, other],
        {"2025-07-15"},
        {},
    )
    assert [it.id for it in selected2] == ["ITEM_CLEAR"]
    assert missing2 == []


def test_score_scl_cell_rejects_majority_cloud():
    from eval.s2_cloud_mask import SCL_CLOUD_HIGH, score_scl_cell

    scl = np.full((4, 4), SCL_CLOUD_HIGH, dtype=np.uint8)
    scl[:2, :2] = 4  # veg
    ok, cloud, _snow, valid = score_scl_cell(
        scl, max_cloud_frac=0.15, min_valid_frac=0.85
    )
    assert not ok
    assert cloud == 0.75
    assert valid == 1.0
    clear = np.full((4, 4), 4, dtype=np.uint8)
    ok2, cloud2, _, _ = score_scl_cell(
        clear, max_cloud_frac=0.15, min_valid_frac=0.85
    )
    assert ok2
    assert cloud2 == 0.0
