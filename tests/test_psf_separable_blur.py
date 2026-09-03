"""The separable Gaussian PSF must stay bit-comparable to the per-band 2-D form.

The optimised path replaces a K x K convolution per band with two grouped 1-D
passes and caches the kernels. Both are supervision-critical, so drift here
would silently change the forward model rather than raise.
"""

import pytest
import torch
import torch.nn.functional as F

from models.s2_psf_forward import (
    _gaussian_kernel_1d,
    _separable_blur_weights,
    gaussian_blur_per_band,
)


def _reference_blur(x: torch.Tensor, sigma_px, truncate: float = 4.0) -> torch.Tensor:
    """Original implementation: per-band outer-product kernel, 2-D reflect pad."""
    c = x.shape[1]
    if hasattr(sigma_px, "__len__"):
        sigmas = list(sigma_px)
    else:
        sigmas = [float(sigma_px)] * c
    out = []
    for ch in range(c):
        k1 = _gaussian_kernel_1d(float(sigmas[ch]), truncate, x.device, x.dtype)
        k2 = k1[:, None] * k1[None, :]
        k2 = k2 / k2.sum()
        weight = k2.view(1, 1, k2.shape[0], k2.shape[1])
        plane = F.pad(x[:, ch : ch + 1], (k2.shape[0] // 2,) * 4, mode="reflect")
        out.append(F.conv2d(plane, weight))
    return torch.cat(out, dim=1)


S2_MTF_SIGMAS = [4.2 / 2.5, 3.25 / 2.5, 2.8 / 2.5]


@pytest.mark.parametrize(
    "sigmas",
    [S2_MTF_SIGMAS, [0.25, 0.25, 0.25], 2.0, [0.4, 1.9, 0.4]],
    ids=["s2_mtf", "dsen2_narrow", "scalar", "mixed_radii"],
)
def test_separable_blur_matches_per_band_reference(sigmas):
    torch.manual_seed(0)
    x = torch.randn(2, 3, 40, 61, dtype=torch.float64)
    expected = _reference_blur(x, sigmas)
    got = gaussian_blur_per_band(x, sigmas)
    assert got.shape == expected.shape
    assert torch.allclose(got, expected, atol=1e-12)


def test_separable_blur_gradients_match_reference():
    torch.manual_seed(0)
    sigmas = S2_MTF_SIGMAS
    x = torch.randn(1, 3, 32, 45, dtype=torch.float64, requires_grad=True)
    (g_ref,) = torch.autograd.grad(_reference_blur(x, sigmas).square().sum(), x)
    (g_new,) = torch.autograd.grad(gaussian_blur_per_band(x, sigmas).square().sum(), x)
    assert torch.allclose(g_new, g_ref, atol=1e-12)


def test_blur_preserves_a_constant_field():
    """Reflect padding plus normalised kernels must not darken flat regions."""
    x = torch.full((1, 3, 24, 24), 0.37, dtype=torch.float64)
    got = gaussian_blur_per_band(x, S2_MTF_SIGMAS)
    assert torch.allclose(got, x, atol=1e-12)


def test_bands_with_smaller_sigma_are_zero_padded_not_widened():
    """Mixed radii share one kernel width; the narrow band must be unaffected."""
    narrow, wide = 0.3, 2.5
    w_x, _, radius = _separable_blur_weights([narrow, wide], 4.0, torch.device("cpu"), torch.float64)
    assert radius == max(1, int(-(-4.0 * wide // 1)))
    k_narrow = _gaussian_kernel_1d(narrow, 4.0, torch.device("cpu"), torch.float64)
    off = radius - k_narrow.numel() // 2
    row = w_x[0, 0, 0]
    assert torch.allclose(row[off : off + k_narrow.numel()], k_narrow, atol=1e-12)
    assert torch.allclose(row[:off], torch.zeros(off, dtype=torch.float64), atol=0)


def test_kernel_cache_returns_identical_tensors():
    """Kernels are rebuilt every training step unless the cache actually hits."""
    a = _separable_blur_weights(S2_MTF_SIGMAS, 4.0, torch.device("cpu"), torch.float32)
    b = _separable_blur_weights(S2_MTF_SIGMAS, 4.0, torch.device("cpu"), torch.float32)
    assert a[0] is b[0] and a[1] is b[1]
