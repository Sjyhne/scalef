"""tcnn's HashGrid is isotropic over the unit square, so rectangular AOIs are
handled by squeezing the short input axis instead of changing the encoding.

These tests pin the two properties that makes correct: the realised per-axis
resolution ladder tracks the PyTorch rectangular grid, and the squeezed
coordinates stay inside [0, 1] (tcnn indexes its dense coarse levels directly,
so an input above 1 would read past the end of the level).
"""

import pytest
import torch

from input_projections.hashgrid_projection import hash_level_resolutions

tcnn = pytest.importorskip(
    "tinycudann", reason="tiny-cuda-nn is optional; skip when not installed"
)

from input_projections.hashgrid_tcnn import HashGridTcnn  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="tcnn requires CUDA"
)

# (base_h, base_w, max_h, max_w) taken from real AOIs at mult=1.
BERGEN = (17, 52, 68, 209)
STAVANGER = (16, 86, 65, 347)
KRISTIANSAND = (61, 36, 244, 145)


def _build(geom):
    base_h, base_w, max_h, max_w = geom
    return HashGridTcnn(
        n_levels=16,
        n_features_per_level=2,
        log2_hashmap_size=21,
        base_resolution_h=base_h,
        base_resolution_w=base_w,
        max_resolution_h=max_h,
        max_resolution_w=max_w,
        interpolation="linear",
    )


@pytest.mark.parametrize("geom", [BERGEN, STAVANGER, KRISTIANSAND], ids=["bergen", "stavanger", "kristiansand"])
def test_finest_level_matches_requested_rectangular_resolution(geom):
    _, _, max_h, max_w = geom
    enc = _build(geom)
    assert enc.resolutions_w[-1] == max_w
    assert enc.resolutions_h[-1] == max_h


@pytest.mark.parametrize("geom", [BERGEN, STAVANGER, KRISTIANSAND], ids=["bergen", "stavanger", "kristiansand"])
def test_effective_ladder_tracks_pytorch_rectangular_ladder(geom):
    """One ladder is floored per axis, the other is floored after scaling, so
    the two may disagree by a single cell but must never drift beyond that."""
    base_h, base_w, max_h, max_w = geom
    enc = _build(geom)
    ref_h = hash_level_resolutions(base_h, max_h, 16)
    ref_w = hash_level_resolutions(base_w, max_w, 16)
    for got, ref in ((enc.resolutions_h, ref_h), (enc.resolutions_w, ref_w)):
        for g, r in zip(got, ref):
            assert abs(g - r) <= 1


@pytest.mark.parametrize("geom", [BERGEN, STAVANGER, KRISTIANSAND], ids=["bergen", "stavanger", "kristiansand"])
def test_squeezed_coordinates_stay_in_unit_square(geom):
    _, _, max_h, max_w = geom
    enc = _build(geom)
    long_max = max(max_h, max_w)
    assert enc.coord_scale.max().item() == pytest.approx(1.0)
    expected_short = min(max_h, max_w) / long_max
    assert enc.coord_scale.min().item() == pytest.approx(expected_short, rel=1e-6)

    x = torch.rand(4096, 2, device="cuda")
    scaled = x.clamp(0.0, 1.0 - 1e-6) * enc.coord_scale.cuda()
    assert scaled.max().item() <= 1.0


def test_square_grid_leaves_coordinates_untouched():
    enc = HashGridTcnn(
        n_levels=8,
        n_features_per_level=2,
        log2_hashmap_size=19,
        base_resolution=16,
        max_resolution=256,
        interpolation="linear",
    )
    assert not enc.rectangular
    assert torch.allclose(enc.coord_scale, torch.ones(2))


def test_forward_is_finite_and_differentiable():
    enc = _build(BERGEN).cuda()
    x = torch.rand(1024, 2, device="cuda", requires_grad=True)
    y = enc(x)
    assert y.shape == (1024, 32)
    assert torch.isfinite(y).all()
    y.square().sum().backward()
    params = list(enc.parameters())
    assert params and params[0].grad is not None
