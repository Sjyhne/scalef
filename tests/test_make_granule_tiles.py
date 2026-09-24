from scripts.make_granule_tiles import _plan_record_for_window


def test_overlap_window_unions_intersecting_plan_dates():
    cells = [
        {
            "iy": 0,
            "ix": 0,
            "row_off": 0,
            "col_off": 0,
            "side": 512,
            "dates": ["2025-07-12"],
        },
        {
            "iy": 0,
            "ix": 1,
            "row_off": 0,
            "col_off": 512,
            "side": 512,
            "dates": ["2025-07-18"],
        },
        {
            "iy": 1,
            "ix": 0,
            "row_off": 512,
            "col_off": 0,
            "side": 512,
            "dates": ["2025-08-12"],
        },
    ]

    record = _plan_record_for_window(
        cells, row_off=448, col_off=448, side=512
    )

    assert record is not None
    assert record["dates"] == ["2025-07-12", "2025-07-18", "2025-08-12"]
    assert len(record["source_plan_cells"]) == 3


def test_overlap_window_without_planned_dates_is_dropped():
    assert (
        _plan_record_for_window(
            [
                {
                    "iy": 0,
                    "ix": 0,
                    "row_off": 0,
                    "col_off": 0,
                    "side": 512,
                    "dates": [],
                }
            ],
            row_off=448,
            col_off=448,
            side=512,
        )
        is None
    )
