import pytest

from models.training_schedule import (
    effective_lr_align_args,
    iteration_progress,
    resolve_psf_curriculum_mode,
    resolve_psf_sigma_scale,
)


class Args:
    lr_degradation = "s2_psf_m"
    schedule_horizon_iters = 3000
    schedule_boundaries = "0.27,0.53"
    psf_curriculum = "step"
    psf_sigma_schedule = "none"
    psf_sigma_min_scale = 0.0


def test_psf_curriculum_phases():
    a = Args()
    assert resolve_psf_curriculum_mode(a, 100) == "area"
    assert resolve_psf_curriculum_mode(a, 1000) == "s2_psf"
    assert resolve_psf_curriculum_mode(a, 2000) == "s2_psf_m"


def test_psf_sigma_linear():
    a = Args()
    a.psf_curriculum = "none"
    a.psf_sigma_schedule = "linear"
    assert resolve_psf_sigma_scale(a, 0) == 0.0
    assert resolve_psf_sigma_scale(a, 1500) == pytest.approx(0.5)
    assert resolve_psf_sigma_scale(a, 3000) == pytest.approx(1.0)


def test_effective_lr_align_args_sigma():
    a = Args()
    a.psf_curriculum = "none"
    a.psf_sigma_schedule = "linear"
    align = effective_lr_align_args(a, 1500)
    assert align.lr_degradation == "s2_psf_m"
    assert align.psf_sigma_scale == pytest.approx(0.5)


def test_iteration_progress():
    assert iteration_progress(1500, 3000) == 0.5
