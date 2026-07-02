"""Unit tests for MGRS subtile clipping + AOI nodata helpers."""

import numpy as np
import pytest

from download_s2_earth_search import (
    _mgrs_grid_code,
    _north_is_low_y_index,
    _west_is_low_x_index,
    aoi_nodata_fractions,
    best_valid_window,
    clip_stack_mgrs_subtile,
    filter_stack_by_aoi_nodata,
    normalize_mgrs_tile,
)


class _FakeStack:
    def __init__(self, y: np.ndarray, x: np.ndarray, data: np.ndarray):
        self.y = type("Y", (), {"values": y})()
        self.x = type("X", (), {"values": x})()
        self._data = data
        self.sizes = {"y": data.shape[0], "x": data.shape[1], "time": 1, "band": 3}

    def isel(self, **kw):
        y_sl = kw.get("y", slice(None))
        x_sl = kw.get("x", slice(None))
        sub = self._data[y_sl, x_sl]
        y = self.y.values[y_sl]
        x = self.x.values[x_sl]
        return _FakeStack(y, x, sub)


def test_normalize_mgrs_grid_code():
    assert normalize_mgrs_tile("mgrs-19kdt") == "19KDT"
    assert _mgrs_grid_code("19KDT") == "MGRS-19KDT"


def test_north_west_quarter_north_up():
    # Row 0 = north (y decreases with row index, common for UTM north-up rasters).
    y = np.linspace(7000000, 6990000, 100)
    x = np.linspace(500000, 510000, 80)
    data = np.arange(100 * 80).reshape(100, 80)
    stack = _FakeStack(y, x, data)
    clipped = clip_stack_mgrs_subtile(stack, "nw")
    assert clipped.sizes["y"] == 50
    assert clipped.sizes["x"] == 40
    assert _north_is_low_y_index(y)
    assert _west_is_low_x_index(x)


def test_unknown_subtile_raises():
    y = np.linspace(0, 1, 4)
    x = np.linspace(0, 1, 4)
    stack = _FakeStack(y, x, np.zeros((4, 4)))
    with pytest.raises(ValueError, match="Unknown mgrs subtile"):
        clip_stack_mgrs_subtile(stack, "bad")


def _xr_stack(frames: list[np.ndarray]):
    """Build a (time, band, y, x) DataArray from a list of HWC RGB frames."""
    xr = pytest.importorskip("xarray")
    arr = np.stack([np.transpose(f, (2, 0, 1)) for f in frames], axis=0)
    t, b, h, w = arr.shape
    return xr.DataArray(
        arr,
        dims=("time", "band", "y", "x"),
        coords={
            "time": np.arange(t),
            "band": ["red", "green", "blue"][:b],
            "y": np.linspace(10, 0, h),
            "x": np.linspace(0, 10, w),
        },
    )


def test_aoi_nodata_fractions_counts_nan_pixels():
    clean = np.ones((4, 4, 3), dtype=np.float32)
    half_nan = np.ones((4, 4, 3), dtype=np.float32)
    half_nan[:2, :, :] = np.nan  # top half is nodata -> 50%
    stack = _xr_stack([clean, half_nan])
    fracs = aoi_nodata_fractions(stack, ("red", "green", "blue"))
    assert fracs[0] == pytest.approx(0.0)
    assert fracs[1] == pytest.approx(50.0)


def test_filter_drops_partial_coverage_scene():
    clean = np.ones((4, 4, 3), dtype=np.float32)
    mostly_nan = np.full((4, 4, 3), np.nan, dtype=np.float32)
    mostly_nan[0, 0, :] = 1.0  # ~94% nodata
    stack = _xr_stack([clean, mostly_nan])
    kept, fracs = filter_stack_by_aoi_nodata(
        stack, max_aoi_nodata_pct=1.0, assets=("red", "green", "blue")
    )
    assert int(kept.sizes["time"]) == 1
    assert fracs == pytest.approx([0.0])


def test_filter_raises_when_all_partial():
    mostly_nan = np.full((4, 4, 3), np.nan, dtype=np.float32)
    stack = _xr_stack([mostly_nan])
    with pytest.raises(RuntimeError, match="No scenes with AOI nodata"):
        filter_stack_by_aoi_nodata(
            stack, max_aoi_nodata_pct=1.0, assets=("red", "green", "blue")
        )


def test_best_valid_window_picks_clean_corner():
    # 10x10 nodata-count map; the bottom-right 4x4 is fully clean (zeros).
    invalid = np.ones((10, 10), dtype=np.float32)
    invalid[6:10, 6:10] = 0.0
    r, c, score = best_valid_window(invalid, 4, 4)
    assert score == 0.0
    assert r == 6 and c == 6


def test_best_valid_window_minimizes_when_no_clean_window():
    invalid = np.ones((6, 6), dtype=np.float32)
    invalid[0, 0] = 0.0  # only one clean pixel
    r, c, score = best_valid_window(invalid, 3, 3)
    # Best 3x3 window includes the single zero -> score = 8 (9 cells - 1 clean).
    assert score == 8.0
    assert (r, c) == (0, 0)
