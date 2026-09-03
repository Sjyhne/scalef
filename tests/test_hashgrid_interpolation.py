import pytest
import torch

from input_projections.hashgrid_projection import (
    HashGridProjection,
    _grid_coord,
    _hash_interp_weight,
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


def _reference_loop_forward(grid, x):
    """Original per-level Python loop, kept as the correctness oracle."""
    x = x.reshape(-1, 2).clamp(0.0, 1.0 - 1e-6)
    outs = []
    for li in range(grid.n_levels):
        rh = int(grid.resolutions_h[li])
        rw = int(grid.resolutions_w[li])
        gx = _grid_coord(x[:, 0], rw, grid.interpolation)
        gy = _grid_coord(x[:, 1], rh, grid.interpolation)
        x0 = torch.floor(gx).to(torch.int64)
        y0 = torch.floor(gy).to(torch.int64)
        x1 = torch.clamp(x0 + 1, max=rw - 1)
        y1 = torch.clamp(y0 + 1, max=rh - 1)
        wx = _hash_interp_weight(gx - x0.to(gx.dtype), grid.interpolation).unsqueeze(-1)
        wy = _hash_interp_weight(gy - y0.to(gy.dtype), grid.interpolation).unsqueeze(-1)
        t = grid.tables[li]
        f00 = t[grid._hash(x0, y0, grid.hashmap_size)]
        f10 = t[grid._hash(x1, y0, grid.hashmap_size)]
        f01 = t[grid._hash(x0, y1, grid.hashmap_size)]
        f11 = t[grid._hash(x1, y1, grid.hashmap_size)]
        f0 = f00 * (1 - wx) + f10 * wx
        f1 = f01 * (1 - wx) + f11 * wx
        f = f0 * (1 - wy) + f1 * wy
        outs.append(f)
    return torch.cat(outs, dim=-1)


@pytest.mark.parametrize("interpolation", ["linear", "smoothstep"])
def test_batched_forward_matches_reference_loop(interpolation):
    torch.manual_seed(0)
    grid = HashGridProjection(
        n_levels=6,
        n_features_per_level=2,
        log2_hashmap_size=10,
        base_resolution_h=9,
        base_resolution_w=17,
        max_resolution_h=36,
        max_resolution_w=68,
        interpolation=interpolation,
    )
    grid.tables.data.uniform_(-0.5, 0.5)
    coords = torch.rand(64, 2)
    assert torch.allclose(grid(coords), _reference_loop_forward(grid, coords), atol=1e-6)


def test_batched_forward_gradients_match_reference_loop():
    torch.manual_seed(0)
    common = dict(
        n_levels=5,
        n_features_per_level=2,
        log2_hashmap_size=9,
        base_resolution=8,
        max_resolution=48,
        interpolation="linear",
    )
    grid = HashGridProjection(**common)
    grid.tables.data.uniform_(-0.5, 0.5)
    coords = torch.rand(48, 2)

    grid(coords).square().sum().backward()
    fast_grad = grid.tables.grad.clone()

    grid.zero_grad(set_to_none=True)
    _reference_loop_forward(grid, coords).square().sum().backward()
    assert torch.allclose(fast_grad, grid.tables.grad, atol=1e-6)


def test_output_preserves_per_level_concatenation_order():
    torch.manual_seed(0)
    grid = HashGridProjection(
        n_levels=3,
        n_features_per_level=2,
        log2_hashmap_size=8,
        base_resolution=8,
        max_resolution=32,
    )
    grid.tables.data.uniform_(-0.5, 0.5)
    coords = torch.rand(7, 2)
    out = grid(coords)
    assert out.shape == (7, 6)
    # Zeroing one level's table must only affect that level's feature slice.
    grid.tables.data[1].zero_()
    changed = (grid(coords) - out).abs().sum(dim=0) > 0
    assert not bool(changed[0]) and not bool(changed[1])
    assert bool(changed[2]) and bool(changed[3])


def test_preserves_leading_dimensions():
    grid = HashGridProjection(
        n_levels=4, n_features_per_level=2, log2_hashmap_size=8,
        base_resolution=8, max_resolution=32,
    )
    out = grid(torch.rand(2, 5, 9, 2))
    assert out.shape == (2, 5, 9, 8)


def test_base_resolution_is_honored():
    grid = HashGridProjection(
        n_levels=4,
        n_features_per_level=2,
        log2_hashmap_size=8,
        base_resolution=10,
        max_resolution=80,
    )
    assert int(grid.resolutions[0]) == 10
    assert int(grid.resolutions[-1]) == 80


def test_base_resolution_falls_back_to_quarter_of_max():
    grid = HashGridProjection(
        n_levels=4,
        n_features_per_level=2,
        log2_hashmap_size=8,
        base_resolution=0,
        max_resolution=80,
    )
    assert int(grid.resolutions[0]) == 20


def test_per_axis_base_resolution_for_rectangular_grid():
    grid = HashGridProjection(
        n_levels=4,
        n_features_per_level=2,
        log2_hashmap_size=8,
        base_resolution_h=17,
        base_resolution_w=52,
        max_resolution_h=68,
        max_resolution_w=209,
    )
    assert grid.rectangular
    assert int(grid.resolutions_h[0]) == 17
    assert int(grid.resolutions_w[0]) == 52
    assert int(grid.resolutions_h[-1]) == 68
    assert int(grid.resolutions_w[-1]) == 209


def test_base_resolution_above_max_falls_back():
    grid = HashGridProjection(
        n_levels=4,
        n_features_per_level=2,
        log2_hashmap_size=8,
        base_resolution=500,
        max_resolution=80,
    )
    assert int(grid.resolutions[0]) == 20
