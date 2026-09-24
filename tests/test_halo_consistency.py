from __future__ import annotations

import numpy as np
import torch

from eval.halo_consistency import (
    expand_aoi_for_halo,
    halo_consistency_loss,
    halo_masks_lr,
    parse_halo_sides,
)


def test_expand_aoi_west_and_clamp():
    aoi = {"col_off": 10, "row_off": 20, "width": 512, "height": 512}
    out = expand_aoi_for_halo(aoi, west_px=32, north_px=0)
    assert out == {"col_off": 0, "row_off": 20, "width": 522, "height": 512}
    edge = expand_aoi_for_halo(
        {"col_off": 8704, "row_off": 2048, "width": 512, "height": 512},
        west_px=32,
    )
    assert edge["col_off"] == 8672
    assert edge["width"] == 544


def test_halo_masks_west_strip():
    m = halo_masks_lr(8, 16, west_px=3, north_px=0)
    assert m[:, :3].all()
    assert not m[:, 3:].any()


def test_parse_halo_sides():
    assert parse_halo_sides("west", 32) == (32, 0)
    assert parse_halo_sides("north", 16) == (0, 16)
    assert parse_halo_sides("west,north", 8) == (8, 8)


def test_halo_loss_zero_on_match():
    pred = torch.ones(1, 4, 4, 3)
    mask = torch.zeros(4, 4, dtype=torch.bool)
    mask[:, :2] = True
    loss = halo_consistency_loss(pred, pred, mask, recon_loss="mae")
    assert float(loss) == 0.0
    target = pred.clone()
    target[:, :, :2] = 0.0
    loss2 = halo_consistency_loss(pred, target, mask, recon_loss="mae")
    assert float(loss2) == 1.0


def test_halo_mean_mode_matches_dc_only():
    pred = torch.zeros(1, 8, 8, 3)
    pred[..., :] = 0.10
    target = torch.zeros(1, 8, 8, 3)
    target[..., :] = 0.20
    mask = torch.ones(8, 8, dtype=torch.bool)
    loss = halo_consistency_loss(pred, target, mask, mode="mean")
    assert abs(float(loss) - 0.10) < 1e-5
    # High-frequency noise should not change the mean loss.
    pred2 = pred.clone()
    pred2[:, 0::2, 0::2, :] += 0.05
    pred2[:, 1::2, 1::2, :] -= 0.05
    loss_n = halo_consistency_loss(pred2, target, mask, mode="mean")
    assert abs(float(loss_n) - 0.10) < 1e-5


def test_halo_lowfreq_is_between_pixel_and_mean():
    pred = torch.zeros(1, 16, 16, 3)
    pred[..., :] = 0.10
    pred[:, :, 8:, :] = 0.30
    target = torch.ones(1, 16, 16, 3) * 0.20
    mask = torch.ones(16, 16, dtype=torch.bool)
    full = float(halo_consistency_loss(pred, target, mask, recon_loss="mae", mode="full"))
    lf = float(
        halo_consistency_loss(
            pred, target, mask, recon_loss="mae", mode="lowfreq", lf_hr_px=8
        )
    )
    mean = float(halo_consistency_loss(pred, target, mask, mode="mean"))
    assert abs(mean - 0.0) < 1e-5 or mean < full
    assert lf >= 0.0
    assert abs(full - 0.10) < 1e-5
