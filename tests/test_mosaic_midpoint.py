"""Midpoint mosaic with a per-tile overlap gain."""

import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.mosaic_midpoint import solve_overlap_gains, write_midpoint


def _write(path: Path, value: float, *, east: float, north: float, size: int = 32) -> None:
    rgb = np.full((3, size, size), value, dtype=np.float32)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=size,
        height=size,
        count=3,
        dtype="float32",
        crs="EPSG:32632",
        transform=from_origin(east, north, 2.5, 2.5),
    ) as dst:
        dst.write(rgb)


def test_off_date_tile_is_scaled_to_the_primary_overlap(tmp_path: Path):
    west = tmp_path / "west.tif"
    east = tmp_path / "east.tif"
    # 32 px tiles, 8 px (20 m) overlap. East is the brighter primary date.
    _write(west, 0.02, east=0.0, north=80.0)
    _write(east, 0.04, east=60.0, north=80.0)
    sources = [(west, 0, 0), (east, 0, 1)]
    dates = {(0, 0): "2025-07-18", (0, 1): "2025-07-12"}
    gains = solve_overlap_gains(sources, dates, "2025-07-12", min_valid=50)
    assert gains[(0, 1)] == 1.0
    assert gains[(0, 0)] == np.float32(2.0) or abs(gains[(0, 0)] - 2.0) < 0.02

    out = tmp_path / "mosaic.tif"
    write_midpoint(sources, out, cog=False, block_size=16, gains=gains)
    with rasterio.open(out) as src:
        dark_side = src.read(window=rasterio.windows.Window(4, 8, 8, 8))
        bright_side = src.read(window=rasterio.windows.Window(44, 8, 8, 8))
    assert abs(float(dark_side.mean()) - 0.04) < 0.002
    assert abs(float(bright_side.mean()) - 0.04) < 0.002


def test_primary_tiles_keep_their_own_brightness(tmp_path: Path):
    west = tmp_path / "west.tif"
    east = tmp_path / "east.tif"
    _write(west, 0.02, east=0.0, north=80.0)
    _write(east, 0.04, east=60.0, north=80.0)
    sources = [(west, 0, 0), (east, 0, 1)]
    dates = {(0, 0): "2025-07-12", (0, 1): "2025-07-12"}
    gains = solve_overlap_gains(sources, dates, "2025-07-12", min_valid=50)
    assert gains[(0, 0)] == 1.0
    assert gains[(0, 1)] == 1.0
