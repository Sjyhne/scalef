"""SCL / per-frame cloud-mask helpers for the reconstruction loss."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.lr_holdout import EarlyStopState, apply_lr_clear_masks, compute_holdout_val_loss
from eval.s2_cloud_mask import (
    SCL_CIRRUS,
    SCL_CLOUD_HIGH,
    SCL_CLOUD_MED,
    SCL_SHADOW,
    apply_clear_to_holdout_masks,
    clear_from_cloud_mask,
    cloudy_from_scl,
    frame_clear_fractions,
)
from s2_dataset import _read_frame_clear


def test_cloudy_from_scl_classes():
    scl = np.array([[4, SCL_CLOUD_MED], [SCL_CIRRUS, SCL_SHADOW]], dtype=np.uint8)
    cloudy = cloudy_from_scl(scl)
    assert cloudy.tolist() == [[False, True], [True, False]]
    cloudy_sh = cloudy_from_scl(scl, include_shadow=True)
    assert cloudy_sh.tolist() == [[False, True], [True, True]]
    assert cloudy_from_scl(np.array([[SCL_CLOUD_HIGH]]))[0, 0]


def test_clear_from_cloud_mask_nonzero_is_cloudy():
    mask = np.array([[0, 1], [2, 0]], dtype=np.uint8)
    assert clear_from_cloud_mask(mask).tolist() == [[True, False], [False, True]]
    stacked = mask[None, ...]
    assert np.array_equal(clear_from_cloud_mask(stacked), clear_from_cloud_mask(mask))


def test_apply_clear_to_holdout_masks_ands():
    hold = [torch.ones(1, 2, 2, 1, dtype=torch.bool)]
    hold[0][0, 0, 0, 0] = False
    clear = torch.tensor([[[True, False], [True, True]]])
    out = apply_clear_to_holdout_masks(hold, clear)
    assert out[0][0, 0, 0, 0] == False
    assert out[0][0, 0, 1, 0] == False
    assert out[0][0, 1, 0, 0] == True
    assert hold[0][0, 0, 1, 0] == True  # input unchanged


def test_apply_lr_clear_masks_keeps_spatial_for_val():
    spatial = torch.ones(1, 4, 4, 1, dtype=torch.bool)
    spatial[:, 0:2, 0:2, :] = False
    clear = torch.ones(1, 4, 4, dtype=torch.bool)
    clear[0, 0, 0] = False
    clear[0, 3, 3] = False
    state = EarlyStopState(
        train_masks=[spatial.clone()],
        spatial_masks=[spatial.clone()],
        val_ids=[0],
        patience=0,
        min_iters=0,
        holdout_block=2,
    )
    state = apply_lr_clear_masks(state, clear)
    assert state.train_masks[0][0, 3, 3, 0] == False
    assert state.train_masks[0][0, 0, 0, 0] == False
    assert state.spatial_masks[0][0, 3, 3, 0] == True
    assert state.spatial_masks[0][0, 0, 0, 0] == False
    assert frame_clear_fractions(clear)[0] == 14 / 16


def test_holdout_val_excludes_cloudy_pixels():
    class _DS:
        lr_height = 16
        lr_width = 16

        def get_hr_coordinates(self):
            return torch.zeros(64, 64, 2)

        def get_lr_sample_hwc(self, idx):
            return torch.ones(16, 16, 3)

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, coords, sample_idx=None, lr_frames=None, **kwargs):
            return torch.zeros_like(lr_frames)

    spatial = torch.ones(1, 16, 16, 1, dtype=torch.bool)
    spatial[:, 0:8, 0:8, :] = False
    args = type(
        "A",
        (),
        {
            "recon_loss": "mae",
            "holdout_block": 8,
            "holdout_patch_batch": 64,
            "holdout_psf_pad_lr": 0,
        },
    )()
    model = _Model()
    clear_all = torch.ones(1, 16, 16, dtype=torch.bool)
    state = apply_lr_clear_masks(
        EarlyStopState(
            train_masks=[spatial.clone()],
            spatial_masks=[spatial.clone()],
            val_ids=[0],
            patience=0,
            min_iters=0,
            holdout_block=8,
        ),
        clear_all,
    )
    loss_clear = compute_holdout_val_loss(model, _DS(), state, args, torch.device("cpu"))
    assert abs(loss_clear - 1.0) < 1e-5

    clear_none = torch.ones(1, 16, 16, dtype=torch.bool)
    clear_none[0, 0:8, 0:8] = False
    state_cloud = apply_lr_clear_masks(
        EarlyStopState(
            train_masks=[spatial.clone()],
            spatial_masks=[spatial.clone()],
            val_ids=[0],
            patience=0,
            min_iters=0,
            holdout_block=8,
        ),
        clear_none,
    )
    loss_cloud = compute_holdout_val_loss(
        model, _DS(), state_cloud, args, torch.device("cpu")
    )
    assert loss_cloud == 0.0


def test_read_frame_clear_missing_is_all_clear(tmp_path):
    s2_dir = Path(tmp_path)
    clear = _read_frame_clear(
        s2_dir, {"cloud_mask": None}, None, height=3, width=4
    )
    assert clear.shape == (3, 4)
    assert bool(clear.all())
    clear2 = _read_frame_clear(
        s2_dir, {"cloud_mask": "nope.tif"}, None, height=3, width=4
    )
    assert bool(clear2.all())


def test_read_frame_clear_reprojects_aoi_mask(tmp_path):
    """City OmniCloudMask is already cropped; do not apply the granule window."""
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.windows import Window

    path = Path(tmp_path) / "001_aoi_cloud.tif"
    cloudy = np.ones((4, 4), dtype=np.uint8)
    profile = {
        "driver": "GTiff",
        "height": 4,
        "width": 4,
        "count": 1,
        "dtype": "uint8",
        "crs": "EPSG:32632",
        "transform": from_origin(20, 60, 10, 10),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(cloudy, 1)
    # Granule-style window that does not fit the 4×4 AOI mask.
    win = Window(10, 10, 8, 8)
    clear = _read_frame_clear(
        Path(tmp_path),
        {"cloud_mask": "001_aoi_cloud.tif"},
        win,
        height=8,
        width=8,
        dst_transform=from_origin(0, 80, 10, 10),
        dst_crs="EPSG:32632",
    )
    assert clear.shape == (8, 8)
    assert bool(clear[:2, :].all())
    assert bool(clear[:, :2].all())
    assert not bool(clear[2:6, 2:6].any())


def test_read_frame_clear_crops_window(tmp_path):
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.windows import Window

    path = Path(tmp_path) / "001_aoi_cloud.tif"
    cloudy = np.zeros((8, 8), dtype=np.uint8)
    cloudy[2:6, 1:5] = 1
    profile = {
        "driver": "GTiff",
        "height": 8,
        "width": 8,
        "count": 1,
        "dtype": "uint8",
        "crs": "EPSG:32632",
        "transform": from_origin(0, 80, 10, 10),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(cloudy, 1)
    win = Window(1, 2, 4, 4)
    clear = _read_frame_clear(
        Path(tmp_path),
        {"cloud_mask": "001_aoi_cloud.tif"},
        win,
        height=4,
        width=4,
    )
    assert clear.shape == (4, 4)
    assert not bool(clear.any())


def test_propagate_tiles_patches_cell_meta(tmp_path):
    from scripts.attach_scl_masks import propagate_tiles

    parent = Path(tmp_path) / "32VNM"
    parent.mkdir()
    (parent / "001_aoi_cloud.tif").write_bytes(b"x")
    (parent / "001_scl.tif").write_bytes(b"y")
    meta = {
        "frames": [
            {
                "path": "001.tif",
                "cloud_mask": "001_aoi_cloud.tif",
                "scl_path": "001_scl.tif",
                "mask_source": "SCL",
                "cloud_frac": 0.2,
            }
        ]
    }
    cell = Path(tmp_path) / "32VNM_t512_y00_x00"
    cell.mkdir()
    (cell / "meta.json").write_text(
        json.dumps({"frames": [{"path": "001.tif", "cloud_mask": None}]}) + "\n"
    )
    n = propagate_tiles(parent, meta, [])
    assert n == 1
    tmeta = json.loads((cell / "meta.json").read_text())
    assert tmeta["frames"][0]["cloud_mask"] == "001_aoi_cloud.tif"
    assert tmeta["frames"][0]["scl_path"] == "001_scl.tif"
    assert (cell / "001_aoi_cloud.tif").is_symlink()
    assert (cell / "001_scl.tif").is_symlink()
