from __future__ import annotations

from types import SimpleNamespace

import torch

from models.lr_tile_sampler import (
    CrossFrameTileSampler,
    LrTileSampler,
    build_cross_frame_tile_sampler,
    build_lr_tile_sampler,
    crop_lr_hr_tensors,
    hr_window_for_lr_tile,
    infer_df,
    lr_tile_origins,
    raw_tiles_per_step,
    resolve_lr_tile_mix,
    stack_cross_frame_tiles,
)


def test_origins_nonoverlapping_512():
    origins = lr_tile_origins(512, 512, 64)
    assert len(origins) == 64
    assert origins[0] == (0, 0)
    assert origins[1] == (0, 64)
    assert origins[-1] == (448, 448)


def test_origins_128():
    origins = lr_tile_origins(512, 512, 128)
    assert len(origins) == 16
    assert origins[0] == (0, 0)
    assert origins[-1] == (384, 384)


def test_incomplete_edge_dropped():
    origins = lr_tile_origins(500, 500, 64)
    assert all(r + 64 <= 500 and c + 64 <= 500 for r, c in origins)
    assert len(origins) == 7 * 7


def test_hr_window_df4():
    assert hr_window_for_lr_tile(128, 64, 64, 4) == (512, 256, 256, 256)


def test_infer_df():
    assert infer_df(2048, 512) == 4


def test_crop_shapes():
    coords = torch.zeros(1, 2048, 2048, 2)
    lr = torch.zeros(1, 512, 512, 3)
    mask = torch.ones(1, 512, 512, 1, dtype=torch.bool)
    coords[:, 512:768, 256:512, 0] = 1.0
    lr[:, 128:192, 64:128, :] = 2.0
    c, t, m = crop_lr_hr_tensors(coords, lr, 128, 64, 64, mask=mask)
    assert c.shape == (1, 256, 256, 2)
    assert t.shape == (1, 64, 64, 3)
    assert m.shape == (1, 64, 64, 1)
    assert float(c[0, 0, 0, 0]) == 1.0
    assert float(t[0, 0, 0, 0]) == 2.0


def test_stack_tiles_cats_batch_and_repeats_ids():
    coords = torch.zeros(1, 2048, 2048, 2)
    lr = torch.zeros(1, 512, 512, 3)
    mask = torch.ones(1, 512, 512, 1, dtype=torch.bool)
    coords[:, 0:512, 0:512, 0] = 1.0
    coords[:, 0:512, 512:1024, 0] = 3.0
    lr[:, 0:128, 0:128, :] = 2.0
    lr[:, 0:128, 128:256, :] = 4.0
    sid = torch.tensor([7])
    dx = torch.tensor([0.1])
    from models.lr_tile_sampler import stack_lr_hr_tiles

    c, t, m, sid_b, dx_b, dy_b = stack_lr_hr_tiles(
        coords,
        lr,
        [(0, 0), (0, 128)],
        128,
        mask=mask,
        sample_id=sid,
        gt_dx=dx,
        gt_dy=torch.zeros(1),
    )
    assert c.shape == (2, 512, 512, 2)
    assert t.shape == (2, 128, 128, 3)
    assert m.shape == (2, 128, 128, 1)
    assert float(c[0, 0, 0, 0]) == 1.0
    assert float(c[1, 0, 0, 0]) == 3.0
    assert sid_b.tolist() == [7, 7]
    assert abs(float(dx_b[1]) - 0.1) < 1e-6


def test_sampler_cover_frame_then_reshuffle():
    s = LrTileSampler.from_shapes(512, 512, 128, tiles_per_step=0, seed=1)
    assert s.tiles_per_step == 16
    first = s.next_origins()
    assert len(first) == 16
    assert sorted(first) == sorted(lr_tile_origins(512, 512, 128))
    second = s.next_origins()
    assert len(second) == 16
    assert sorted(second) == sorted(first)


def test_sampler_mini_batch_cycles():
    s = LrTileSampler.from_shapes(512, 512, 64, tiles_per_step=4, seed=2)
    seen: set[tuple[int, int]] = set()
    for _ in range(16):
        batch = s.next_origins()
        assert len(batch) == 4
        seen.update(batch)
    assert len(seen) == 64


def test_zero_tiles_per_step_means_cover_not_one():
    assert raw_tiles_per_step(SimpleNamespace(lr_tiles_per_step=0)) == 0
    dataset = SimpleNamespace(lr_height=512, lr_width=512)
    args = SimpleNamespace(lr_tile=128, lr_tiles_per_step=0, seed=0, lr_tile_mix="within")
    sampler = build_lr_tile_sampler(dataset, args)
    assert sampler is not None
    assert sampler.tiles_per_step == 16
    assert len(sampler.next_origins()) == 16


def test_within_mix_builds_spatial_not_cross():
    dataset = SimpleNamespace(lr_height=512, lr_width=512, num_samples=8)
    args = SimpleNamespace(lr_tile=128, lr_tiles_per_step=2, seed=0, lr_tile_mix="within")
    assert build_lr_tile_sampler(dataset, args) is not None
    assert build_cross_frame_tile_sampler(dataset, args) is None


def test_cross_epoch_builds_cross_not_within():
    dataset = SimpleNamespace(lr_height=512, lr_width=512, num_samples=8)
    args = SimpleNamespace(
        lr_tile=128, lr_tiles_per_step=2, seed=0, lr_tile_mix="cross_epoch"
    )
    assert build_lr_tile_sampler(dataset, args) is None
    cross = build_cross_frame_tile_sampler(dataset, args)
    assert cross is not None
    assert cross.mode == "epoch"
    assert cross.tiles_per_step == 2
    assert cross.num_frames == 8


def test_cross_epoch_covers_all_spatial_then_reshuffles():
    s = CrossFrameTileSampler.from_shapes(
        512, 512, 128, tiles_per_step=4, num_frames=5, mode="epoch", seed=3
    )
    seen_spatial: set[tuple[int, int]] = set()
    frames: set[int] = set()
    for _ in range(4):  # 16 origins / 4 = one epoch
        batch = s.next_pairs()
        assert len(batch) == 4
        for origin, fid in batch:
            seen_spatial.add(origin)
            frames.add(fid)
            assert 0 <= fid < 5
    assert len(seen_spatial) == 16
    assert frames  # at least some frames drawn


def test_cross_iid_independent_draws():
    s = CrossFrameTileSampler.from_shapes(
        512, 512, 128, tiles_per_step=2, num_frames=16, mode="iid", seed=4
    )
    batch = s.next_pairs()
    assert len(batch) == 2
    for origin, fid in batch:
        assert origin in lr_tile_origins(512, 512, 128)
        assert 0 <= fid < 16


def test_stack_cross_frame_uses_per_pair_frame_ids():
    class _FakeDS:
        def __init__(self):
            self._coords = torch.zeros(2048, 2048, 2)
            self._coords[0:512, 0:512, 0] = 1.0
            self._coords[0:512, 512:1024, 0] = 3.0
            self._lrs = [
                torch.full((512, 512, 3), 10.0),
                torch.full((512, 512, 3), 20.0),
            ]

        def get_hr_coordinates(self):
            return self._coords

        def get_lr_sample_hwc(self, idx: int):
            return self._lrs[idx]

    ds = _FakeDS()
    masks = [
        torch.ones(1, 512, 512, 1, dtype=torch.bool),
        torch.ones(1, 512, 512, 1, dtype=torch.bool),
    ]
    pairs = [((0, 0), 0), ((0, 128), 1)]
    c, t, m, sid, dx, dy = stack_cross_frame_tiles(
        ds, pairs, 128, device=torch.device("cpu"), train_masks=masks
    )
    assert c.shape == (2, 512, 512, 2)
    assert t.shape == (2, 128, 128, 3)
    assert float(t[0, 0, 0, 0]) == 10.0
    assert float(t[1, 0, 0, 0]) == 20.0
    assert sid.tolist() == [0, 1]
    assert m is not None and m.shape == (2, 128, 128, 1)
    assert dx.shape == (2,)


def test_cross_same_tile_one_origin_many_frames():
    s = CrossFrameTileSampler.from_shapes(
        512, 512, 128, tiles_per_step=4, num_frames=16, mode="same_tile", seed=5
    )
    batch = s.next_pairs()
    assert len(batch) == 4
    origins = {o for o, _ in batch}
    frames = [f for _, f in batch]
    assert len(origins) == 1
    assert len(set(frames)) == 4
    assert all(0 <= f < 16 for f in frames)


def test_cross_same_tile_build():
    dataset = SimpleNamespace(lr_height=512, lr_width=512, num_samples=16)
    args = SimpleNamespace(
        lr_tile=128, lr_tiles_per_step=2, seed=0, lr_tile_mix="cross_same_tile"
    )
    cross = build_cross_frame_tile_sampler(dataset, args)
    assert cross is not None
    assert cross.mode == "same_tile"
    assert build_lr_tile_sampler(dataset, args) is None


def test_resolve_lr_tile_mix():
    assert resolve_lr_tile_mix(SimpleNamespace(lr_tile_mix="cross_iid")) == "cross_iid"
    assert (
        resolve_lr_tile_mix(SimpleNamespace(lr_tile_mix="cross_same_tile"))
        == "cross_same_tile"
    )
