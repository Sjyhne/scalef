import math

import torch

from input_projections.hashgrid_projection import (
    HashGridProjection,
    _grid_coord,
    _hash_interp_weight,
    compute_level_footprint_weights,
    hash_level_resolutions,
)


def test_smoothstep_applies_half_voxel_offset_per_level():
    axis = torch.tensor([0.25])
    r = 17
    linear = _grid_coord(axis, r, "linear")
    smooth = _grid_coord(axis, r, "smoothstep")
    assert torch.allclose(linear, axis * (r - 1))
    assert torch.allclose(smooth, linear + 0.5)


def test_smoothstep_offset_changes_encoding():
    torch.manual_seed(0)
    coords = torch.tensor([[0.31, 0.67], [0.12, 0.88]])
    common = dict(
        n_levels=4,
        n_features_per_level=2,
        log2_hashmap_size=8,
        base_resolution=8,
        max_resolution=32,
    )
    linear = HashGridProjection(interpolation="linear", **common)
    smooth = HashGridProjection(interpolation="smoothstep", **common)
    smooth.tables.data.copy_(linear.tables.data)

    y_linear = linear(coords)
    y_smooth = smooth(coords)
    assert y_linear.shape == y_smooth.shape
    assert not torch.allclose(y_linear, y_smooth)


def test_smoothstep_derivatives_vanish_at_cell_boundaries():
    t = torch.tensor([0.0, 1.0], requires_grad=True)
    w = _hash_interp_weight(t, "smoothstep")
    grad = torch.autograd.grad(w.sum(), t, create_graph=False)[0]
    assert torch.allclose(grad, torch.zeros_like(grad))


def test_linear_weights_at_cell_boundaries():
    t = torch.tensor([0.0, 1.0])
    w = _hash_interp_weight(t, "linear")
    assert torch.allclose(w, t)


def test_footprint_weights_disabled_for_nonpositive_sigma():
    res = hash_level_resolutions(16, 2048, 16)
    assert compute_level_footprint_weights(res, 0.0) is None
    assert compute_level_footprint_weights(res, -1.0) is None


def test_footprint_weights_match_erf_and_decrease_with_resolution():
    res = hash_level_resolutions(16, 2048, 8)
    sigma = 1.0 / (math.sqrt(12.0) * 64)  # area degradation, 64 px LR side
    w = compute_level_footprint_weights(res, sigma)
    expected = torch.tensor(
        [math.erf(1.0 / (math.sqrt(8.0) * sigma * n)) for n in res], dtype=torch.float32
    )
    assert torch.allclose(w, expected, atol=1e-6)
    # Monotonically non-increasing: fine levels get suppressed.
    assert torch.all(w[:-1] >= w[1:])
    # Coarse level barely touched, finest level heavily downweighted.
    assert w[0] > 0.99
    assert w[-1] < w[0]


def test_footprint_weights_scale_hashgrid_features():
    torch.manual_seed(0)
    coords = torch.tensor([[0.31, 0.67], [0.12, 0.88]])
    sigma = 0.02
    common = dict(
        n_levels=4,
        n_features_per_level=2,
        log2_hashmap_size=8,
        base_resolution=8,
        max_resolution=64,
    )
    plain = HashGridProjection(level_sigma=0.0, **common)
    weighted = HashGridProjection(level_sigma=sigma, **common)
    weighted.tables.data.copy_(plain.tables.data)

    y_plain = plain(coords)
    y_weighted = weighted(coords)
    w = compute_level_footprint_weights(plain.resolutions.tolist(), sigma)
    expected = y_plain * w.repeat_interleave(common["n_features_per_level"])
    assert torch.allclose(y_weighted, expected, atol=1e-6)
