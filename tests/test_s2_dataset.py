import sys
from datetime import date
from pathlib import Path

import numpy as np

# Ensure repo root is importable when pytest changes cwd/import mode.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from s2_dataset import (
    _base_frame_index_for_nib,
    _build_hr_eval_mask,
    _center_crop_hw,
    _fit_affine_1d,
    _frame_acquisition_date,
    _harmonize_hr_histogram_match,
    _index_to_front,
    _is_nib_focus_location,
    _largest_ones_rectangle,
    _nib_acquisition_date,
    standardize_rgb_masked,
)


def test_masked_standardization_ignores_cloud_pixels():
    import torch

    rng = np.random.default_rng(0)
    ground = rng.uniform(0.02, 0.08, size=(32, 32, 3)).astype(np.float32)
    rgb = ground.copy()
    cloud = np.zeros((32, 32), dtype=bool)
    cloud[:8] = True
    rgb[cloud] = 0.9

    _, mean, std, used = standardize_rgb_masked(torch.from_numpy(rgb), ~cloud)
    clear = ground[~cloud]
    assert used
    np.testing.assert_allclose(mean.view(-1).numpy(), clear.mean(axis=0), rtol=1e-5)
    np.testing.assert_allclose(std.view(-1).numpy(), clear.std(axis=0, ddof=1), rtol=1e-4)

    _, mean_all, _, _ = standardize_rgb_masked(torch.from_numpy(rgb), None)
    assert float(mean_all.view(-1)[0]) > float(mean.view(-1)[0]) + 0.1


def test_masked_standardization_falls_back_when_mostly_cloudy():
    import torch

    rgb = torch.rand(32, 32, 3)
    mask = np.zeros((32, 32), dtype=bool)
    mask[0, :10] = True
    standardized, mean, _, used = standardize_rgb_masked(rgb, mask)
    assert not used
    torch.testing.assert_close(mean.view(-1), rgb.mean(dim=(0, 1)))
    assert standardized.shape == rgb.shape


def test_largest_ones_rectangle_full():
    mask = np.ones((4, 5), dtype=bool)
    assert _largest_ones_rectangle(mask) == (0, 4, 0, 5)


def test_largest_ones_rectangle_hole():
    mask = np.ones((5, 5), dtype=bool)
    mask[0, :] = False
    mask[:, 0] = False
    r0, r1, c0, c1 = _largest_ones_rectangle(mask)
    assert (r0, r1, c0, c1) == (1, 5, 1, 5)


def test_center_crop_hw():
    r0, r1, c0, c1 = _center_crop_hw(10, 20, 4, 4)
    assert (r1 - r0, c1 - c0) == (4, 4)
    assert (r0, c0) == (3, 8)


def test_harmonize_hr_histogram_match_preserves_valid_mask():
    rng = np.random.default_rng(1)
    base_lr = rng.uniform(0.02, 0.2, size=(8, 8, 3)).astype(np.float32)
    hr = rng.uniform(0.3, 0.9, size=(32, 32, 3)).astype(np.float32)
    hr[0:4, :, :] = 0.0
    matched = _harmonize_hr_histogram_match(hr, base_lr)
    assert matched.shape == hr.shape
    assert np.all(matched[0:4, :, :] == 0.0)
    assert float(matched[4:, :, :].mean()) < float(hr[4:, :, :].mean())


def test_fit_affine_1d_recovers_known_affine():
    rng = np.random.default_rng(0)
    x = rng.uniform(0.05, 0.95, size=200_000).astype(np.float64)
    a_true = 1.7
    b_true = -0.15
    y = a_true * x + b_true
    y = y + rng.normal(scale=0.01, size=y.shape)
    a, b = _fit_affine_1d(x, y, trim_percentiles=(1.0, 99.0))
    assert abs(a - a_true) < 0.03
    assert abs(b - b_true) < 0.03


def test_nib_and_frame_date_parsing():
    assert _nib_acquisition_date(Path("trondheim_2017-06-30_2.5m.tif")).isoformat() == "2017-06-30"
    frame = {"path": "003_20170723.tif", "datetime": "2017-07-23T10:56:19.027000+00:00"}
    assert _frame_acquisition_date(frame).isoformat() == "2017-07-23"


def test_build_hr_eval_mask_respects_coverage_and_s2():
    df = 4
    hr_cover = np.ones((8, 8), dtype=bool)
    hr_cover[:, -2:] = False
    s2_valid_lr = np.ones((2, 2), dtype=bool)
    s2_valid_lr[0, 1] = False
    hr_rgb = np.ones((8, 8, 3), dtype=np.float32)
    hr_rgb[0, :, :] = 0.0
    mask = _build_hr_eval_mask(hr_cover, s2_valid_lr, hr_rgb, df, erode_hr=0)
    assert mask.shape == (8, 8)
    assert not mask[0, 0]
    assert not mask[0, 4]
    assert mask[4, 4]


def test_is_nib_focus_location_from_meta():
    assert _is_nib_focus_location({"focus_project_folder": "01_asker_akershus"}, Path("asker_lr512"))
    assert not _is_nib_focus_location({}, Path("demo"))


def test_apply_hr_spatial_shift_moves_content():
    from s2_dataset import _apply_hr_spatial_shift

    hr = np.zeros((8, 8, 3), dtype=np.float32)
    hr[2:4, 2:4, :] = 0.8
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:4, 2:4] = True
    hr2, mask2 = _apply_hr_spatial_shift(hr, mask, dy=1.0, dx=0.0)
    assert mask2[3, 2] or mask2[3, 3]
    assert float(hr2[3, 2:4].mean()) > float(hr2[1, 2:4].mean())


def test_base_frame_index_for_nib_picks_closest_date():
    frames = [
        {"path": "001_20170630.tif", "datetime": "2017-06-30T10:50:29+00:00"},
        {"path": "002_20170720.tif", "datetime": "2017-07-20T10:50:29+00:00"},
        {"path": "003_20170516.tif", "datetime": "2017-05-16T10:50:31+00:00"},
    ]
    nib_date = _nib_acquisition_date(Path("trondheim_2017-06-30_2.5m.tif"))
    assert _base_frame_index_for_nib(frames, nib_date) == 0
    assert _base_frame_index_for_nib(frames[1:], nib_date) == 0  # 2017-07-20 vs 2017-05-16


def test_base_frame_skips_closer_cloudy_day():
    frames = [
        {"path": "000_20250605.tif", "datetime": "2025-06-05T10:00:00+00:00"},
        {"path": "001_20250715.tif", "datetime": "2025-07-15T10:00:00+00:00"},
        {"path": "002_20250712.tif", "datetime": "2025-07-12T10:00:00+00:00"},
    ]
    center = date(2025, 7, 15)
    # July 15 is closest but 12% cloud; July 12 is clear.
    assert (
        _base_frame_index_for_nib(
            frames, center, cloud_fracs=[0.13, 0.12, 0.0], max_cloud_frac=0.02
        )
        == 2
    )


def test_base_frame_falls_back_to_clearest():
    frames = [
        {"path": "000_20250605.tif", "datetime": "2025-06-05T10:00:00+00:00"},
        {"path": "001_20250715.tif", "datetime": "2025-07-15T10:00:00+00:00"},
    ]
    center = date(2025, 7, 15)
    assert (
        _base_frame_index_for_nib(
            frames, center, cloud_fracs=[0.04, 0.11], max_cloud_frac=0.02
        )
        == 0
    )


def test_index_to_front():
    assert _index_to_front(2, 4) == [2, 0, 1, 3]
    assert _index_to_front(0, 3) == [0, 1, 2]


def test_force_base_date_picks_requested_day_despite_cloud():
    frames = [
        {"path": "000_20250712.tif", "datetime": "2025-07-12T10:00:00+00:00"},
        {"path": "001_20250812.tif", "datetime": "2025-08-12T10:00:00+00:00"},
    ]
    center = date(2025, 7, 15)
    # Default 2% rule would skip July 12 (5% cloud) and freeze August.
    assert (
        _base_frame_index_for_nib(
            frames, center, cloud_fracs=[0.05, 0.0], max_cloud_frac=0.02
        )
        == 1
    )
    hits = [
        i
        for i, fr in enumerate(frames)
        if _frame_acquisition_date(fr) == date(2025, 7, 12)
    ]
    assert hits == [0]
