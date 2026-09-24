import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import mosaic_seam_ramp as seam


def _write_rgb(path: Path, value: float, *, x0: float, y0: float) -> None:
    rgb = np.full((3, 4, 4), value, dtype=np.float32)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=4,
        height=4,
        count=3,
        dtype="float32",
        crs="EPSG:32632",
        transform=from_origin(x0, y0, 2.5, 2.5),
    ) as dst:
        dst.write(rgb)


def test_is_date_cut_only_when_assignment_differs():
    dates = {(0, 0): "2025-07-12", (0, 1): "2025-07-12", (0, 2): "2025-07-18"}
    assert not seam._is_date_cut(dates, (0, 0), (0, 1))
    assert seam._is_date_cut(dates, (0, 1), (0, 2))
    assert seam._is_date_cut(None, (0, 0), (0, 1))


def test_balance_cut_includes_icm_joined_independent_base_change():
    assigned = {(0, 0): "2025-07-12", (0, 1): "2025-07-12"}
    independent = {(0, 0): "2025-07-12", (0, 1): "2025-07-18"}

    assert seam._is_balance_cut(assigned, independent, (0, 0), (0, 1))
    independent[(0, 1)] = "2025-07-12"
    assert not seam._is_balance_cut(assigned, independent, (0, 0), (0, 1))


def test_streaming_mosaic_uses_grid_origin_and_ramps_date_cuts(tmp_path, monkeypatch):
    monkeypatch.setattr(seam, "CELL_HR", 4)
    monkeypatch.setattr(seam, "EDGE", 1)
    left = tmp_path / "left.tif"
    right = tmp_path / "right.tif"
    _write_rgb(left, 0.1, x0=100.0, y0=200.0)
    _write_rgb(right, 0.3, x0=110.0, y0=200.0)
    grid = {(2, 3): left, (2, 4): right}
    out = tmp_path / "mosaic.tif"

    rec = seam.write_mosaic_strips(
        {"parent": "32VNM"},
        grid,
        ramp_px=1,
        out=out,
        dates={(2, 3): "2025-07-12", (2, 4): "2025-07-18"},
    )

    with rasterio.open(out) as src:
        assert src.width == 20
        assert src.height == 12
        assert src.transform == from_origin(70.0, 220.0, 2.5, 2.5)
        data = src.read()
    np.testing.assert_allclose(data[:, 8:12, 12:15], 0.1)
    np.testing.assert_allclose(data[:, 8:12, 15:17], 0.2)
    np.testing.assert_allclose(data[:, 8:12, 17:20], 0.3)
    assert rec["n_ramped_edges"] == 1
    assert rec["date_cuts_only"] is True
    assert json.loads(out.with_suffix(".tif.json").read_text())["n_tiles"] == 2


def test_global_harmonization_balances_complete_cell_interiors(tmp_path, monkeypatch):
    monkeypatch.setattr(seam, "CELL_HR", 4)
    monkeypatch.setattr(seam, "EDGE", 1)
    left = tmp_path / "left.tif"
    right = tmp_path / "right.tif"
    _write_rgb(left, 0.1, x0=100.0, y0=200.0)
    _write_rgb(right, 0.3, x0=110.0, y0=200.0)
    grid = {(0, 0): left, (0, 1): right}

    harmony = seam.estimate_harmonization(
        grid,
        strip_px=2,
        segments=4,
        regularization=0.001,
        max_offset=0.2,
    )

    np.testing.assert_allclose(harmony["corrections"][(0, 0)], 0.1, atol=5e-4)
    np.testing.assert_allclose(harmony["corrections"][(0, 1)], -0.1, atol=5e-4)
    assert harmony["metrics"]["all_edges_after_global"]["p95"] < 0.001

    out = tmp_path / "harmonized.tif"
    rec = seam.write_mosaic_strips(
        {"parent": "32VNM"},
        grid,
        ramp_px=1,
        out=out,
        dates={(0, 0): "2025-07-12", (0, 1): "2025-07-18"},
        harmonize=True,
        harmonize_strip_px=2,
        harmonize_segments=4,
        harmonize_regularization=0.001,
        harmonize_max_offset=0.2,
        max_harmonized_edge_p95=0.001,
    )

    with rasterio.open(out) as src:
        data = src.read()
    np.testing.assert_allclose(data[:, :, :4], 0.2, atol=5e-4)
    np.testing.assert_allclose(data[:, :, 4:], 0.2, atol=5e-4)
    assert rec["ramp_scope"] == "identity_risk_edges_after_global_harmonization"
    assert rec["qa"]["passed"] is True
