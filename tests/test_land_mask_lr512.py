"""Land overlap must follow the window footprint, not only a coarse point grid."""

from __future__ import annotations

import numpy as np
from matplotlib.path import Path as MplPath
from rasterio.crs import CRS
from rasterio.transform import Affine

from scripts.land_mask_lr512 import assess_window_land, land_hit_frac, window_lonlat_samples
from scripts.recover_shore_tiles import choose_shore_date


def _lonlat_transform() -> Affine:
    # Pixel (col, row) -> (lon, lat). One pixel is 0.1 degrees.
    return Affine(0.1, 0.0, 0.0, 0.0, -0.1, 1.0)


def test_thin_shore_misses_the_sample_grid_but_still_counts_as_land():
    # Land occupies only the western 0.02 degrees of a 1-degree window.
    # The 9x9 sample centers start 1/18 degree in, so every sample is offshore.
    ring = np.array(
        [[-1.0, -1.0], [0.02, -1.0], [0.02, 2.0], [-1.0, 2.0], [-1.0, -1.0]],
        dtype=float,
    )
    paths = [MplPath(ring)]
    transform = _lonlat_transform()
    crs = CRS.from_epsg(4326)
    samples = window_lonlat_samples(transform, crs, row0=0, col0=0, side=10, n=9)
    assert land_hit_frac(paths, samples) == 0.0

    frac, keep = assess_window_land(
        paths,
        transform,
        crs,
        row0=0,
        col0=0,
        side=10,
        sample_n=9,
        min_land_frac=0.0,
    )
    assert keep
    assert frac > 0.0


def test_new_shore_cell_joins_the_granule_date_without_moving_others():
    # The shared date is clear on this cell, so the hole uses it.
    assert (
        choose_shore_date(
            {"2025-07-02": 0.01, "2025-08-01": 0.0},
            primary="2025-07-02",
            neighbor_dates=["2025-08-01"],
        )
        == "2025-07-02"
    )
    # Primary is not in the stack. A clear neighbour date is used.
    assert (
        choose_shore_date(
            {"2025-06-22": 0.01, "2025-08-29": 0.0},
            primary="2025-07-02",
            neighbor_dates=["2025-06-22", "2025-06-22"],
        )
        == "2025-06-22"
    )


def test_open_water_window_is_dropped():
    ring = np.array(
        [[-1.0, -1.0], [-0.2, -1.0], [-0.2, 2.0], [-1.0, 2.0], [-1.0, -1.0]],
        dtype=float,
    )
    frac, keep = assess_window_land(
        [MplPath(ring)],
        _lonlat_transform(),
        CRS.from_epsg(4326),
        row0=0,
        col0=0,
        side=10,
        sample_n=9,
        min_land_frac=0.0,
    )
    assert not keep
    assert frac == 0.0
