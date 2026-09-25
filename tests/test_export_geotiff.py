import types

import numpy as np
import rasterio
from rasterio.transform import from_origin

from eval.export_geotiff import export_qgis_layers, write_rgb_geotiff


def _read(path):
    with rasterio.open(path) as src:
        return src.read()


def test_default_clip_is_unit_range(tmp_path):
    img = np.full((4, 4, 3), -0.05, np.float32)
    img[0, 0] = 1.2
    p = write_rgb_geotiff(tmp_path / "a.tif", img, transform=from_origin(0, 4, 1, 1), crs="EPSG:32632")
    data = _read(p)
    assert data.min() == 0.0 and data.max() == 1.0


def test_offset_removed_layers_keep_negative_reflectance(tmp_path):
    ds = types.SimpleNamespace(
        eval_reflectance_offset=0.1,
        get_geo_meta=lambda: {"crs": "EPSG:32632", "hr_transform": from_origin(0, 4, 1, 1)},
    )
    pred = np.full((4, 4, 3), -0.03, np.float32)
    pred[0, 0] = -0.4
    written = export_qgis_layers(tmp_path, sr_pred_hwc=pred, s2_bilinear_hwc=pred, dataset=ds)
    data = _read(written["sr_pred.tif"])
    assert np.isclose(data[:, 1, 1], -0.03).all()
    assert np.isclose(data[:, 0, 0], -0.1).all()
