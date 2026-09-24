import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.mosaic_granule_sr import (  # noqa: E402
    _stream_feather,
    estimate_date_affine_harmonization,
    measure_overlap_qa,
)


def _write_tile(path: Path, value: float, x0: float) -> None:
    profile = {
        "driver": "GTiff",
        "width": 16,
        "height": 16,
        "count": 3,
        "dtype": "float32",
        "crs": "EPSG:32632",
        "transform": from_origin(x0, 20.0, 2.5, 2.5),
        "nodata": 0.0,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.full((3, 16, 16), value, dtype=np.float32))


def _write_gradient_tile(
    path: Path,
    *,
    x0: float,
    gain: np.ndarray,
    offset: np.ndarray,
) -> None:
    xs = x0 + (np.arange(24, dtype=np.float32) + 0.5) * 2.5
    ys = (np.arange(16, dtype=np.float32) + 0.5) * 2.5
    base = 0.08 + 0.0015 * xs[None, :] + 0.001 * ys[:, None]
    rgb = np.stack([base, 0.9 * base, 0.75 * base])
    rgb = gain[:, None, None] * rgb + offset[:, None, None]
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=24,
        height=16,
        count=3,
        dtype="float32",
        crs="EPSG:32632",
        transform=from_origin(x0, 20.0, 2.5, 2.5),
        nodata=0.0,
    ) as dst:
        dst.write(rgb)


def test_stream_feather_is_windowed_cog_and_applies_offsets(tmp_path):
    first = tmp_path / "first.tif"
    second = tmp_path / "second.tif"
    out = tmp_path / "mosaic.tif"
    _write_tile(first, 0.2, 0.0)
    _write_tile(second, 0.3, 20.0)

    meta = _stream_feather(
        [first, second],
        out,
        cog=True,
        feather_px=4,
        corrections={second: np.full(3, -0.1, dtype=np.float32)},
        block_size=128,
    )

    with rasterio.open(out) as src:
        data = src.read()
        assert src.width == 24
        assert src.height == 16
        assert src.tags(ns="IMAGE_STRUCTURE").get("LAYOUT") == "COG"
        assert np.allclose(data[data > 0], 0.2, atol=1e-6)
        assert np.all(src.dataset_mask() > 0)
    assert meta["streaming"] is True
    assert meta["overlap_pixels"] == 128
    assert meta["nodata_fraction"] == 0.0


def test_overlap_qa_splits_identity_risk_pairs(tmp_path):
    first = tmp_path / "first.tif"
    second = tmp_path / "second.tif"
    _write_tile(first, 0.2, 0.0)
    _write_tile(second, 0.3, 20.0)
    grid = {(0, 0): first, (0, 1): second}

    qa = measure_overlap_qa(
        grid,
        corrections={(0, 1): np.full(3, -0.1, dtype=np.float32)},
        dates={(0, 0): "2025-07-12", (0, 1): "2025-08-12"},
        independent_dates={(0, 0): "2025-07-12", (0, 1): "2025-08-12"},
        margin_px=0,
    )

    assert qa["same_identity"]["n"] == 0
    assert qa["identity_risk"]["n"] == 1
    assert qa["identity_risk"]["p95"] < 1e-6


def test_date_affine_harmonization_reduces_cross_date_overlap_error(tmp_path):
    first = tmp_path / "first.tif"
    second = tmp_path / "second.tif"
    _write_gradient_tile(
        first,
        x0=0.0,
        gain=np.ones(3, dtype=np.float32),
        offset=np.zeros(3, dtype=np.float32),
    )
    _write_gradient_tile(
        second,
        x0=20.0,
        gain=np.array([1.08, 0.94, 1.04], dtype=np.float32),
        offset=np.array([0.012, -0.006, 0.008], dtype=np.float32),
    )
    grid = {(0, 0): first, (0, 1): second}
    dates = {(0, 0): "2025-07-12", (0, 1): "2025-08-12"}

    before = measure_overlap_qa(grid, dates=dates, margin_px=0)
    harmony = estimate_date_affine_harmonization(
        grid,
        dates=dates,
        margin_px=0,
        samples_per_pair=256,
        gain_regularization=1e-5,
        offset_regularization=1e-5,
        max_gain_delta=0.2,
        max_offset=0.05,
    )
    after = measure_overlap_qa(
        grid, corrections=harmony["corrections"], dates=dates, margin_px=0
    )

    assert after["identity_risk"]["p95"] < before["identity_risk"]["p95"] * 0.1
    assert harmony["configuration"]["reference_date"] == "2025-08-12"
    reference = harmony["configuration"]["date_transforms"]["2025-08-12"]
    np.testing.assert_allclose(reference["gain"], 1.0, atol=1e-3)
    np.testing.assert_allclose(reference["offset"], 0.0, atol=1e-3)
