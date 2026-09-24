import json
from pathlib import Path

import numpy as np
import pytest
import torch

from eval.common_footprint import (
    bilinear_scores_are_invariant,
    common_center_hw,
    crop_bchw_to_hw,
    crop_hwc_frac,
    nest_window_frac,
    score_common_footprint,
)
from s2_dataset import load_hr_eval_shift
from scripts.freeze_confirmatory_eval import build_frozen_manifest


def _entry(dy: float, dx: float, *, s2_dir_name: str, lr: int = 512) -> dict:
    return {
        "df": 4,
        "s2_dir_name": s2_dir_name,
        "lr_size": [lr, lr],
        "hr_size": [lr * 4, lr * 4],
        "base_frame_index": 0,
        "base_frame_date": "2024-05-14",
        "nib_acquisition_date": "2024-05-15",
        "hr_shift_hr_px": {"dy": dy, "dx": dx},
    }


def test_load_hr_eval_shift_prefers_tile_key_over_city(tmp_path: Path):
    payload = {
        "cities": {
            "asker": _entry(-0.1, -0.2, s2_dir_name="asker"),
        },
        "tiles": {
            "asker_lr512": _entry(-0.8, -0.3, s2_dir_name="asker_lr512"),
        },
    }
    path = tmp_path / "spatial_alignment.json"
    path.write_text(json.dumps(payload))
    assert load_hr_eval_shift(
        "asker", path=path, current_df=4, s2_dir=tmp_path / "asker_lr512"
    ) == pytest.approx((-0.8, -0.3))
    # Size-ladder children inherit the parent LR512 tile, not the native crop.
    assert load_hr_eval_shift(
        "asker", path=path, current_df=4, s2_dir=tmp_path / "asker_lr64"
    ) == pytest.approx((-0.8, -0.3))
    # City-only lookup still falls back to the named city key.
    assert load_hr_eval_shift("asker", path=path, current_df=4) == pytest.approx((-0.1, -0.2))


def test_load_hr_eval_shift_prefers_explicit_parent_patch(tmp_path: Path):
    payload = {
        "cities": {"asker": _entry(-0.1, -0.2, s2_dir_name="asker")},
        "tiles": {
            "asker_lr512": _entry(-0.8, -0.3, s2_dir_name="asker_lr512"),
            "asker_p02_02": _entry(0.25, 0.75, s2_dir_name="asker_p02_02"),
        },
    }
    path = tmp_path / "spatial_alignment.json"
    path.write_text(json.dumps(payload))
    assert load_hr_eval_shift(
        "asker",
        path=path,
        current_df=4,
        s2_dir=tmp_path / "asker_nest64_p02_02_y00_x00",
        parent_tile_id="asker_p02_02",
    ) == pytest.approx((0.25, 0.75))


def test_load_hr_eval_shift_scales_with_df(tmp_path: Path):
    payload = {"cities": {"asker": _entry(-0.8, 0.4, s2_dir_name="asker_lr512")}}
    path = tmp_path / "spatial_alignment.json"
    path.write_text(json.dumps(payload))
    assert load_hr_eval_shift("asker", path=path, current_df=2) == pytest.approx((-0.4, 0.2))


def test_freeze_overlays_lr512_tile_shift(tmp_path: Path):
    native = _entry(-0.1, -0.2, s2_dir_name="asker", lr=256)
    tile = _entry(-0.8, -0.3, s2_dir_name="asker_lr512")
    source = tmp_path / "spatial_alignment.json"
    source.write_text(
        json.dumps({"version": 1, "method": "alignment", "cities": {"asker": native}, "tiles": {"asker_lr512": tile}})
    )
    data_root = tmp_path / "s2"
    meta_dir = data_root / "asker_lr512"
    meta_dir.mkdir(parents=True)
    (meta_dir / "meta.json").write_text(
        json.dumps({"frames": [{"path": "000.tif", "id": "a"}, {"path": "001.tif", "id": "b"}]})
    )
    frozen = build_frozen_manifest(source, data_root=data_root)
    assert frozen["cities"]["asker"]["hr_shift_hr_px"] == {"dy": -0.8, "dx": -0.3}
    assert frozen["cities"]["asker"]["s2_dir_name"] == "asker_lr512"
    assert frozen["cities"]["asker"]["confirmatory_s2_dir_name"] == "asker_lr512"
    assert frozen["cities"]["asker"]["lr_size"] == [512, 512]


def test_nest_window_frac_maps_lr64_into_parent():
    assert nest_window_frac(0, 0, 64, 512) == (0.0, 0.0, 0.125, 0.125)
    assert nest_window_frac(7, 3, 64, 512) == (0.875, 0.375, 0.125, 0.125)
    assert nest_window_frac(1, 1, 64, 128) == (0.5, 0.5, 0.5, 0.5)
    img = np.arange(64, dtype=np.float32).reshape(8, 8, 1)
    crop = crop_hwc_frac(img, 0.5, 0.25, 0.25, 0.25)
    assert crop.shape[:2] == (2, 2)


def test_common_center_hw_is_intersection():
    assert common_center_hw([(2048, 2048), (1024, 1024), (256, 256)]) == (256, 256)


def test_score_common_footprint_bilinear_is_crop_invariant():
    gt = torch.zeros(1, 3, 32, 32)
    gt[:, :, 8:24, 8:24] = 0.8
    bilinear = gt.clone()
    bilinear[:, :, 8:24, 8:24] = 0.5
    pred_large = bilinear.clone()
    pred_small = crop_bchw_to_hw(pred_large, 16, 16)
    gt_small = crop_bchw_to_hw(gt, 16, 16)
    bil_small = crop_bchw_to_hw(bilinear, 16, 16)
    large = score_common_footprint(pred_large, gt, bilinear, target_hw=(16, 16))
    small = score_common_footprint(pred_small, gt_small, bil_small)
    bilinear_scores_are_invariant([large, small])
    assert large["hr_hw"] == [16, 16]
    assert small["hr_hw"] == [16, 16]


def test_bilinear_invariant_detects_crop_dependent_baseline():
    a = {"bilinear_psnr": 30.0, "bilinear_mse": 0.001}
    b = {"bilinear_psnr": 31.5, "bilinear_mse": 0.001}
    with pytest.raises(ValueError, match="bilinear invariant failed"):
        bilinear_scores_are_invariant([a, b])
