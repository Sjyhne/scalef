"""Tests for tiled HR render helpers."""

from __future__ import annotations

import torch

from eval.hr_render import iter_hr_tile_windows, resolve_hr_render_tile, render_hr_rgb_tiled
from eval.masked_metrics import _center_crop_bchw


def test_resolve_hr_render_tile_auto():
    assert resolve_hr_render_tile(512, 512, 0) == 512
    assert resolve_hr_render_tile(2048, 2048, 0) == 2048
    assert resolve_hr_render_tile(8192, 8192, 0) == 2048
    assert resolve_hr_render_tile(8192, 8192, 1024) == 1024


def test_iter_hr_tile_windows_covers():
    windows = list(iter_hr_tile_windows(5000, 3000, 2048))
    covered = torch.zeros(5000, 3000, dtype=torch.bool)
    for r0, c0, r1, c1 in windows:
        covered[r0:r1, c0:c1] = True
    assert bool(covered.all())
    assert windows[0] == (0, 0, 2048, 2048)
    assert windows[-1][2:] == (5000, 3000)


def test_center_crop_bchw():
    t = torch.zeros(1, 3, 4096, 8192)
    (cropped,) = _center_crop_bchw(t, max_side=2048)
    assert cropped.shape == (1, 3, 2048, 2048)


class _StubINR(torch.nn.Module):
    def forward(self, coords, sample_id, scale_factor=1, training=False, **kwargs):
        # coords: [1,h,w,2] → RGB from coords for stitch check
        rgb = coords[..., :1].expand(-1, -1, -1, 3).float() * 0.1 + 0.5
        return rgb, None


def test_render_hr_rgb_tiled_stitches():
    model = _StubINR()
    h, w = 300, 250
    yy = torch.linspace(0, 1, h).view(h, 1).expand(h, w)
    xx = torch.linspace(0, 1, w).view(1, w).expand(h, w)
    coords = torch.stack([xx, yy], dim=-1)
    out = render_hr_rgb_tiled(
        model, coords, torch.tensor([0]), device=torch.device("cpu"), tile=128
    )
    assert out.shape == (1, h, w, 3)
    # Corner pixels should match stub formula on those coords
    assert torch.allclose(out[0, 0, 0], coords[0, 0, :1].expand(3) * 0.1 + 0.5, atol=1e-5)
    assert torch.allclose(out[0, -1, -1], coords[-1, -1, :1].expand(3) * 0.1 + 0.5, atol=1e-5)
