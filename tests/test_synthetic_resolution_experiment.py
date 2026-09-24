"""CPU tests for the controlled synthetic resolution experiment."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from models.s2_psf_forward import DEFAULT_S2_PSF_SIGMA_M_BY_BAND
from scripts.synthetic_resolution_experiment import (
    analytic_target,
    degrade_shifted_truth,
    generate_control_shifts,
    image_metrics,
    radial_fourier_recovery,
    warp_hr_translation,
)


def test_warp_translation_uses_canonical_coordinate_convention():
    ramp = np.zeros((12, 16, 3), dtype=np.float32)
    ramp[..., 0] = np.arange(16, dtype=np.float32)[None, :] / 16
    shifted = warp_hr_translation(ramp, dx_hr_px=2.0, dy_hr_px=0.0)
    shifted = shifted[0].permute(1, 2, 0).numpy()
    # frame(x)=truth(x+dx), away from reflection-padded boundaries.
    np.testing.assert_allclose(shifted[3:-3, 2:-4, 0], ramp[3:-3, 4:-2, 0], atol=1e-6)


def test_control_shifts_are_deterministic_and_phase_specific():
    phase_a = generate_control_shifts(
        "phase_diverse", 8, seed=17, max_subpixel_lr=0.49, max_integer_lr=2
    )
    phase_b = generate_control_shifts(
        "phase_diverse", 8, seed=17, max_subpixel_lr=0.49, max_integer_lr=2
    )
    assert phase_a == phase_b
    assert phase_a[0] == (0.0, 0.0)
    assert all(abs(dx) < 0.5 and abs(dy) < 0.5 for dx, dy in phase_a)

    duplicated = generate_control_shifts(
        "duplicated", 8, seed=17, max_subpixel_lr=0.49, max_integer_lr=2
    )
    assert set(duplicated) == {(0.0, 0.0)}

    integer = generate_control_shifts(
        "integer_aligned", 8, seed=17, max_subpixel_lr=0.49, max_integer_lr=2
    )
    assert all(dx == int(dx) and dy == int(dy) for dx, dy in integer)


def test_degradation_has_expected_shape_and_preserves_constant():
    truth = np.full((64, 80, 3), 0.37, dtype=np.float32)
    lr = degrade_shifted_truth(
        truth,
        (0.0, 0.0),
        scale=4,
        native_gsd_m=10.0,
        sigma_m_by_band=DEFAULT_S2_PSF_SIGMA_M_BY_BAND,
    )
    assert lr.shape == (16, 20, 3)
    np.testing.assert_allclose(lr, 0.37, atol=1e-6)


def test_subpixel_shift_changes_degraded_frame():
    truth = analytic_target(64)
    base = degrade_shifted_truth(
        truth,
        (0.0, 0.0),
        scale=4,
        native_gsd_m=10.0,
        sigma_m_by_band=DEFAULT_S2_PSF_SIGMA_M_BY_BAND,
    )
    shifted = degrade_shifted_truth(
        truth,
        (0.31, -0.27),
        scale=4,
        native_gsd_m=10.0,
        sigma_m_by_band=DEFAULT_S2_PSF_SIGMA_M_BY_BAND,
    )
    assert not np.allclose(base, shifted)
    assert float(np.mean(np.abs(base - shifted))) > 1e-3


def test_image_metrics_are_exact_for_perfect_reconstruction():
    truth = analytic_target(64)
    metrics = image_metrics(truth.copy(), truth, use_lpips=False)
    assert metrics["mse"] == pytest.approx(0.0)
    assert metrics["psnr_db"] == float("inf")
    assert metrics["ssim"] == pytest.approx(1.0)
    assert metrics["lpips"] is None


def test_fourier_recovery_is_one_for_exact_reconstruction():
    truth = analytic_target(64, "chirp")
    recovery = radial_fourier_recovery(truth.copy(), truth, bins=16)
    values = [value for value in recovery["amplitude_recovery"] if value is not None]
    torch.testing.assert_close(torch.tensor(values), torch.ones(len(values)), atol=1e-6, rtol=1e-6)
    assert recovery["high_frequency_recovery_0p25_to_0p5"] == pytest.approx(1.0)
