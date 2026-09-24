import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import mosaic_national_sr as national  # noqa: E402


def _write_raster(
    path: Path,
    value: int,
    *,
    x0: float,
    y0: float,
    pixel_size: float = 1.0,
    crs: str = "EPSG:3857",
    shape: tuple[int, int] = (16, 16),
    nodata: int | None = None,
) -> None:
    data = np.full((1, *shape), value, dtype=np.uint16)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=shape[1],
        height=shape[0],
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=from_origin(x0, y0, pixel_size, pixel_size),
        nodata=nodata,
    ) as dst:
        dst.write(data)


def test_disjoint_sources_stream_destination_windows(tmp_path, monkeypatch):
    left = tmp_path / "left.tif"
    right = tmp_path / "right.tif"
    _write_raster(left, 11, x0=0, y0=16)
    _write_raster(right, 22, x0=24, y0=16)
    out = tmp_path / "national.tif"

    writes = []
    original_write_window = national._write_window

    def recording_write(destination, data, valid, window):
        writes.append(window)
        original_write_window(destination, data, valid, window)

    monkeypatch.setattr(national, "_write_window", recording_write)
    metadata = national.mosaic_sources(
        [left, right],
        out,
        dst_crs="EPSG:3857",
        resolution=1,
        block_size=16,
    )

    with rasterio.open(out) as src:
        data = src.read(1)
        mask = src.dataset_mask()
        assert src.width == 40
        assert src.height == 16
        assert src.tags(ns="IMAGE_STRUCTURE")["LAYOUT"] == "COG"
    np.testing.assert_array_equal(data[:, :16], 11)
    np.testing.assert_array_equal(data[:, 16:24], 0)
    np.testing.assert_array_equal(data[:, 24:], 22)
    np.testing.assert_array_equal(mask[:, 16:24], 0)
    assert len(writes) == 3
    assert all(window.width <= 16 and window.height <= 16 for window in writes)
    assert metadata["streaming"]["windows_written"] == len(writes)
    assert metadata["coverage"]["valid_pixels"] == 16 * 16 * 2
    verification = metadata["verification"]
    assert verification["total_source_valid_pixels"] == 16 * 16 * 2
    assert verification["unique_output_valid_pixels"] == 16 * 16 * 2
    assert verification["duplicate_valid_observations"] == 0
    assert verification["overlap_pixels"] == 0
    assert verification["nodata_gap_pixels"] == 16 * 8
    assert [item["contributed_valid_pixels"] for item in verification["per_source"]] == [
        256,
        256,
    ]
    validation = national.validate_output(out)
    assert validation["checks"]["output_checksum"] is True
    assert validation["checks"]["cog_layout"] is True
    assert validation["checks"]["source_checksums"] is True
    sidecar = out.with_suffix(".tif.json")
    sidecar_metadata = json.loads(sidecar.read_text())
    assert sidecar_metadata["output_sha256"] == national._sha256(out)
    sidecar_metadata["output_sha256"] = "invalid"
    sidecar.write_text(json.dumps(sidecar_metadata))
    with pytest.raises(RuntimeError, match="output_checksum"):
        national.validate_output(out)
    cli_validation = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "mosaic_national_sr.py"),
            "--out",
            str(out),
            "--validate-only",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert cli_validation.returncode != 0
    assert "output_checksum" in cli_validation.stderr


def test_overlap_uses_first_valid_source_and_resumes(tmp_path):
    first = tmp_path / "first.tif"
    second = tmp_path / "second.tif"
    _write_raster(first, 3, x0=0, y0=16, nodata=0)
    _write_raster(second, 9, x0=0, y0=16)
    with rasterio.open(first, "r+") as dst:
        data = dst.read(1)
        data[4:8, 4:8] = 0
        dst.write(data, 1)
    out = tmp_path / "overlap.tif"

    initial = national.mosaic_sources(
        [first, second],
        out,
        dst_crs="EPSG:3857",
        resolution=1,
        block_size=16,
    )
    resumed = national.mosaic_sources(
        [first, second],
        out,
        dst_crs="EPSG:3857",
        resolution=1,
        block_size=16,
    )

    with rasterio.open(out) as src:
        data = src.read(1)
    np.testing.assert_array_equal(data[:4], 3)
    np.testing.assert_array_equal(data[4:8, 4:8], 9)
    np.testing.assert_array_equal(data[8:], 3)
    assert initial["resumed"] is False
    assert resumed["resumed"] is True
    verification = initial["verification"]
    assert verification["total_source_valid_pixels"] == 496
    assert verification["unique_output_valid_pixels"] == 256
    assert verification["duplicate_valid_observations"] == 240
    assert verification["overlap_pixels"] == 240
    assert verification["overlap_source_observations"] == 480
    assert verification["cross_source_overlap_band_observations"] == 240
    assert verification["cross_source_overlap_mae"] == 6
    assert verification["nodata_gap_pixels"] == 0
    assert verification["per_source"][0]["contributed_valid_pixels"] == 240
    assert verification["per_source"][1]["contributed_valid_pixels"] == 16
    assert national.validate_output(out)["checks"]["coverage_matches_sidecar"] is True
    with pytest.raises(RuntimeError, match="overlap_mae_threshold"):
        national.validate_output(out, max_overlap_mae=5)
    with second.open("ab") as changed:
        changed.write(b"changed")
    with pytest.raises(RuntimeError, match="source_checksums"):
        national.validate_output(out)


def test_reprojects_crs_onto_requested_resolution_grid(tmp_path):
    source = tmp_path / "latlon.tif"
    second = tmp_path / "mercator.tif"
    _write_raster(
        source,
        7,
        x0=0.0,
        y0=0.02,
        pixel_size=0.001,
        crs="EPSG:4326",
        shape=(20, 20),
    )
    _write_raster(
        second,
        8,
        x0=0,
        y0=2226.4,
        pixel_size=111.32,
        crs="EPSG:3857",
        shape=(20, 20),
    )
    out = tmp_path / "mercator.tif"

    metadata = national.mosaic_sources(
        [source, second],
        out,
        dst_crs="EPSG:3857",
        resolution=1000,
        block_size=16,
    )

    with rasterio.open(out) as src:
        assert src.crs == rasterio.crs.CRS.from_epsg(3857)
        assert abs(src.transform.a) == 1000
        assert abs(src.transform.e) == 1000
        assert src.width == 3
        assert src.height == 3
        assert np.count_nonzero(src.dataset_mask()) > 0
    assert metadata["width"] == 3
    assert metadata["height"] == 3
    assert metadata["source_details"][0]["crs"] == "EPSG:4326"
    assert metadata["source_crs"]["all_same"] is False
    assert metadata["source_crs"]["unique_crs"] == ["EPSG:3857", "EPSG:4326"]
