"""Controlled checks for the sampling-geometry score."""

import numpy as np
import pytest

from eval.sampling_geometry import (
    circular_deviation,
    sampling_geometry_score,
    sampling_offsets_lr,
)


def _grid(factor: int) -> np.ndarray:
    coords = np.arange(factor, dtype=np.float64) / factor
    xx, yy = np.meshgrid(coords, coords, indexing="ij")
    return np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)


@pytest.mark.parametrize("factor", [2, 3, 4])
def test_ideal_grid_scores_100(factor):
    offsets = _grid(factor)
    usable = np.ones(offsets.shape[0], dtype=bool)
    result = sampling_geometry_score(offsets, usable, factor)
    assert result["status"] == "ok"
    assert result["score"] == pytest.approx(100.0, abs=1e-8)
    assert result["rank"] == factor * factor
    assert result["eigenvalues"].shape == (factor * factor,)
    assert np.min(result["eigenvalues"]) == pytest.approx(1.0, abs=1e-8)


@pytest.mark.parametrize("factor", [2, 3, 4])
def test_identical_offsets_are_rank_deficient(factor):
    offsets = np.tile(np.array([[0.2, 0.7]], dtype=np.float64), (factor * factor, 1))
    usable = np.ones(offsets.shape[0], dtype=bool)
    result = sampling_geometry_score(offsets, usable, factor)
    assert result["score"] == pytest.approx(0.0, abs=1e-8)
    assert result["rank"] == 1
    assert np.count_nonzero(result["eigenvalues"] <= 1e-8) == factor * factor - 1


@pytest.mark.parametrize("factor", [2, 3, 4])
def test_horizontal_offsets_do_not_constrain_2d(factor):
    offsets = np.stack(
        [
            np.arange(factor * factor, dtype=np.float64) / (factor * factor),
            np.zeros(factor * factor, dtype=np.float64),
        ],
        axis=1,
    )
    usable = np.ones(offsets.shape[0], dtype=bool)
    result = sampling_geometry_score(offsets, usable, factor)
    assert result["score"] == pytest.approx(0.0, abs=1e-8)
    assert result["rank"] < factor * factor
    assert result["eigenvalues"].shape == (factor * factor,)


@pytest.mark.parametrize("factor", [2, 3, 4])
def test_fewer_than_r_squared_leaves_a_zero_eigenvalue(factor):
    offsets = _grid(factor)[:-1]
    usable = np.ones(offsets.shape[0], dtype=bool)
    result = sampling_geometry_score(offsets, usable, factor)
    assert offsets.shape[0] == factor * factor - 1
    assert result["n_usable"] == factor * factor - 1
    assert result["score"] == pytest.approx(0.0, abs=1e-8)
    assert result["rank"] <= result["n_usable"]
    assert result["eigenvalues"].shape == (factor * factor,)
    assert np.min(result["eigenvalues"]) == pytest.approx(0.0, abs=1e-8)


@pytest.mark.parametrize("factor", [2, 3, 4])
def test_common_translation_leaves_the_score_unchanged(factor):
    offsets = _grid(factor)
    usable = np.ones(offsets.shape[0], dtype=bool)
    baseline = sampling_geometry_score(offsets, usable, factor)
    shifted = np.mod(offsets + np.array([0.13, 0.71]), 1.0)
    moved = sampling_geometry_score(shifted, usable, factor)
    assert moved["score"] == pytest.approx(baseline["score"], abs=1e-8)
    assert np.allclose(moved["eigenvalues"], baseline["eigenvalues"], atol=1e-8)


def test_no_usable_observations_are_not_a_zero_score():
    offsets = _grid(2)
    usable = np.zeros(offsets.shape[0], dtype=bool)
    result = sampling_geometry_score(offsets, usable, 2)
    assert result["status"] == "no_data"
    assert result["score"] is None
    assert result["rank"] == 0
    assert result["n_usable"] == 0


def test_normalized_translation_converts_to_lr_pixels():
    width = height = 512
    affines = np.zeros((2, 2, 3), dtype=np.float64)
    affines[:, 0, 0] = 1.0
    affines[:, 1, 1] = 1.0
    affines[1, 0, 2] = 1.5 / width
    affines[1, 1, 2] = -0.25 / height
    _, fractional = sampling_offsets_lr(affines, np.array([10, 40]), np.array([3, 9]), height, width)
    assert fractional[0, 0, 0] == pytest.approx(0.0, abs=1e-12)
    assert fractional[0, 0, 1] == pytest.approx(0.0, abs=1e-12)
    assert fractional[1, 0, 0] == pytest.approx(0.5, abs=1e-12)
    assert fractional[1, 0, 1] == pytest.approx(0.75, abs=1e-12)
    assert np.allclose(fractional[1, 0], fractional[1, 1])


def test_scale_makes_the_offset_vary_inside_a_patch():
    affines = np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=np.float64)
    affines = np.repeat(affines, 2, axis=0)
    affines[1, 0, 0] = 1.01
    rows = np.array([0, 63])
    cols = np.array([0, 63])
    displacement, _ = sampling_offsets_lr(affines, rows, cols, 512, 512)
    assert np.max(np.abs(displacement[0])) == pytest.approx(0.0, abs=1e-12)
    assert np.max(displacement[1, :, 0]) - np.min(displacement[1, :, 0]) > 0.5


def test_pixel_positions_are_not_extra_revisits():
    """Two locations stay two scores; stacking them would invent revisits."""
    first = np.array([[0.0, 0.0], [0.5, 0.0]], dtype=np.float64)
    second = np.array([[0.0, 0.5], [0.5, 0.5]], dtype=np.float64)
    usable = np.array([True, True])
    separate = [
        sampling_geometry_score(first, usable, 2)["score"],
        sampling_geometry_score(second, usable, 2)["score"],
    ]
    stacked = sampling_geometry_score(np.vstack([first, second]), np.ones(4, dtype=bool), 2)
    assert stacked["n_usable"] == 4
    assert stacked["score"] == pytest.approx(100.0, abs=1e-8)
    assert all(score == pytest.approx(0.0, abs=1e-8) for score in separate)


def test_circular_deviation_is_zero_for_one_offset_and_wraps():
    single = circular_deviation(np.array([[0.1, 0.9]]))
    assert np.allclose(single, 0.0)
    wrapped = circular_deviation(np.array([[0.01, 0.5], [0.99, 0.5]]))
    assert wrapped[0] == pytest.approx(0.01, abs=1e-12)
    assert wrapped[1] == pytest.approx(0.0, abs=1e-12)
