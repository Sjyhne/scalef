import numpy as np

from s2_dataset import (
    L2A_REFLECTANCE_CAP,
    _apply_hr_spatial_shift,
    _harmonize_hr_histogram_match,
)


def _scene(seed: int = 0):
    rng = np.random.default_rng(seed)
    hr = rng.uniform(0.05, 0.30, size=(64, 64, 3)).astype(np.float32)
    base = rng.uniform(0.10, 0.20, size=(16, 16, 3)).astype(np.float32)
    cloud = np.zeros((16, 16), bool)
    cloud[:4, :4] = True  # 6% of the frame at cloud brightness
    base[cloud] = 0.9
    return hr, base, ~cloud


def test_clear_pixel_reference_ignores_cloud_brightness():
    hr, base, clear = _scene()
    matched_all = _harmonize_hr_histogram_match(hr, base)
    matched_clear = _harmonize_hr_histogram_match(hr, base, clear)
    # matching to the whole frame lifts the brightest HR pixels to cloud level; clear-only does not
    assert matched_all.max() > 0.8
    assert matched_clear.max() < 0.25
    assert abs(matched_clear.std() - base[clear].std()) < 0.01


def test_no_mask_matches_previous_behaviour():
    hr, base, _ = _scene()
    a = _harmonize_hr_histogram_match(hr, base)
    b = _harmonize_hr_histogram_match(hr, base, None)
    assert np.array_equal(a, b)


def test_harmonize_and_shift_preserve_l2a_reflectance_cap():
    rng = np.random.default_rng(42)
    hr = rng.uniform(0.05, 0.95, size=(32, 32, 3)).astype(np.float32)
    base = rng.uniform(0.10, 1.35, size=(16, 16, 3)).astype(np.float32)
    matched = _harmonize_hr_histogram_match(hr, base)
    assert matched.max() > 1.0
    assert matched.max() <= L2A_REFLECTANCE_CAP
    shifted, mask_out = _apply_hr_spatial_shift(matched, np.ones((32, 32), dtype=bool), 0.25, -0.25)
    assert shifted[mask_out].max() > 1.0
    assert shifted[mask_out].max() <= L2A_REFLECTANCE_CAP
