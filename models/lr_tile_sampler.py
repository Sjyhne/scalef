"""Random LR spatial tiles for memory-bounded training.

Each LR tile of size T maps to an HR window of size T * df. Training still uses
the existing HR→PSF→LR path; only the spatial extent of each forward is cropped.

Mixing modes (``--lr_tile_mix``):
  within      — k tiles from the current DataLoader frame (legacy)
  cross_epoch — shuffle spatial index + random frame IDs, walk in mini-batches
  cross_iid   — each step draws k independent (tile, frame) pairs
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


def lr_tile_origins(lr_h: int, lr_w: int, tile: int) -> list[tuple[int, int]]:
    """Non-overlapping complete-tile origins (row, col) in LR pixels."""
    tile = int(tile)
    if tile <= 0:
        return [(0, 0)]
    h, w = int(lr_h), int(lr_w)
    if tile > h or tile > w:
        return [(0, 0)]
    return [(r, c) for r in range(0, h - tile + 1, tile) for c in range(0, w - tile + 1, tile)]


def hr_window_for_lr_tile(
    row: int, col: int, tile: int, df: int
) -> tuple[int, int, int, int]:
    """Return (hr_row, hr_col, hr_h, hr_w) for an LR tile origin."""
    df = max(1, int(df))
    return int(row) * df, int(col) * df, int(tile) * df, int(tile) * df


def infer_df(hr_h: int, lr_h: int) -> int:
    if lr_h <= 0 or hr_h < lr_h or hr_h % lr_h != 0:
        raise ValueError(f"HR height {hr_h} must be an integer multiple of LR height {lr_h}.")
    return hr_h // lr_h


def crop_lr_hr_tensors(
    coords: torch.Tensor,
    lr_target: torch.Tensor,
    row: int,
    col: int,
    tile: int,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Crop ``coords`` [B,HR,W,2] and ``lr_target`` [B,LR,W,C] to one LR tile."""
    df = infer_df(int(coords.shape[1]), int(lr_target.shape[1]))
    hr_r, hr_c, hr_h, hr_w = hr_window_for_lr_tile(row, col, tile, df)
    coords_c = coords[:, hr_r : hr_r + hr_h, hr_c : hr_c + hr_w].contiguous()
    lr_c = lr_target[:, row : row + tile, col : col + tile].contiguous()
    mask_c = None
    if mask is not None:
        mask_c = mask[:, row : row + tile, col : col + tile].contiguous()
    return coords_c, lr_c, mask_c


def _repeat_batch(t: torch.Tensor | None, n: int) -> torch.Tensor | None:
    if t is None or n <= 1:
        return t
    if t.ndim == 0:
        t = t.reshape(1)
    return t.repeat(n, *([1] * (t.ndim - 1)))


def stack_lr_hr_tiles(
    coords: torch.Tensor,
    lr_target: torch.Tensor,
    origins: list[tuple[int, int]],
    tile: int,
    mask: torch.Tensor | None = None,
    sample_id: torch.Tensor | None = None,
    gt_dx: torch.Tensor | None = None,
    gt_dy: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Crop each origin and concatenate on the batch dim (one fused forward)."""
    n = len(origins)
    if n <= 1:
        origin = origins[0]
        coords_c, lr_c, mask_c = crop_lr_hr_tensors(
            coords, lr_target, origin[0], origin[1], tile, mask=mask
        )
        return coords_c, lr_c, mask_c, sample_id, gt_dx, gt_dy
    cs, lrs, ms = [], [], []
    for row, col in origins:
        c, t, m = crop_lr_hr_tensors(coords, lr_target, row, col, tile, mask=mask)
        cs.append(c)
        lrs.append(t)
        if m is not None:
            ms.append(m)
    return (
        torch.cat(cs, dim=0),
        torch.cat(lrs, dim=0),
        torch.cat(ms, dim=0) if ms else None,
        _repeat_batch(sample_id, n),
        _repeat_batch(gt_dx, n),
        _repeat_batch(gt_dy, n),
    )


@dataclass
class LrTileSampler:
    origins: list[tuple[int, int]]
    tiles_per_step: int
    generator: torch.Generator
    _cursor: int = 0
    _order: list[int] | None = None

    @classmethod
    def from_shapes(
        cls,
        lr_h: int,
        lr_w: int,
        tile: int,
        tiles_per_step: int,
        seed: int = 0,
    ) -> "LrTileSampler":
        origins = lr_tile_origins(lr_h, lr_w, tile)
        gen = torch.Generator()
        gen.manual_seed(int(seed))
        n = len(origins)
        per = int(tiles_per_step)
        if per <= 0:
            per = n
        per = max(1, min(per, n))
        sampler = cls(origins=origins, tiles_per_step=per, generator=gen)
        sampler._reshuffle()
        return sampler

    def _reshuffle(self) -> None:
        n = len(self.origins)
        perm = torch.randperm(n, generator=self.generator).tolist()
        self._order = perm
        self._cursor = 0

    def next_origins(self) -> list[tuple[int, int]]:
        if not self.origins:
            return [(0, 0)]
        if self._order is None:
            self._reshuffle()
        assert self._order is not None
        take = self.tiles_per_step
        if self._cursor + take > len(self._order):
            self._reshuffle()
        chunk = self._order[self._cursor : self._cursor + take]
        self._cursor += take
        return [self.origins[i] for i in chunk]


def raw_tiles_per_step(args) -> int:
    """CLI tiles-per-step. 0 means cover all complete tiles (do not coerce 0 → 1)."""
    raw = getattr(args, "lr_tiles_per_step", 1)
    if raw is None:
        return 1
    return int(raw)


def resolve_lr_tile_mix(args) -> str:
    mix = str(getattr(args, "lr_tile_mix", "within") or "within").lower().strip()
    allowed = ("within", "cross_epoch", "cross_iid", "cross_same_tile")
    if mix not in allowed:
        raise ValueError(
            f"Unknown lr_tile_mix={mix!r}; use {', '.join(allowed)}."
        )
    return mix


def _dataset_lr_hw(dataset, args) -> tuple[int, int]:
    lr_h = int(getattr(dataset, "lr_height", 0) or getattr(args, "lr_height", 0) or 0)
    lr_w = int(getattr(dataset, "lr_width", 0) or getattr(args, "lr_width", 0) or 0)
    if lr_h <= 0 or lr_w <= 0:
        sample = dataset[0]
        lr_h, lr_w = int(sample["lr_target"].shape[-3]), int(sample["lr_target"].shape[-2])
    return lr_h, lr_w


def build_lr_tile_sampler(dataset, args) -> LrTileSampler | None:
    """Within-frame spatial sampler, or None when tiling off / cross-frame mix."""
    tile = int(getattr(args, "lr_tile", 0) or 0)
    if tile <= 0 or resolve_lr_tile_mix(args) != "within":
        return None
    lr_h, lr_w = _dataset_lr_hw(dataset, args)
    return LrTileSampler.from_shapes(
        lr_h,
        lr_w,
        tile,
        tiles_per_step=raw_tiles_per_step(args),
        seed=int(getattr(args, "seed", 0) or 0),
    )


@dataclass
class CrossFrameTileSampler:
    """Cross-frame fused batches of ``(origin, frame_id)`` pairs.

    Modes:
      epoch     — shuffle spatial index + random frame IDs, walk in mini-batches
      iid       — each step draws k independent (tile, frame) pairs
      same_tile — each step: one spatial tile × k frames (multi-view on one patch)
    """

    origins: list[tuple[int, int]]
    num_frames: int
    tiles_per_step: int
    mode: str  # "epoch" | "iid" | "same_tile"
    generator: torch.Generator
    _cursor: int = 0
    _spatial_order: list[int] | None = None
    _frame_ids: list[int] | None = None

    @classmethod
    def from_shapes(
        cls,
        lr_h: int,
        lr_w: int,
        tile: int,
        tiles_per_step: int,
        num_frames: int,
        mode: str = "epoch",
        seed: int = 0,
    ) -> "CrossFrameTileSampler":
        origins = lr_tile_origins(lr_h, lr_w, tile)
        gen = torch.Generator()
        gen.manual_seed(int(seed))
        n = len(origins)
        n_frames = max(1, int(num_frames))
        mode = str(mode).lower().strip()
        if mode not in ("epoch", "iid", "same_tile"):
            raise ValueError(f"Unknown cross-frame mode {mode!r}")
        per = int(tiles_per_step)
        if mode == "same_tile":
            # tiles_per_step = number of frames fused on one spatial tile
            if per <= 0:
                per = n_frames
            per = max(1, per)
        else:
            if per <= 0:
                per = n
            per = max(1, min(per, max(1, n)))
        sampler = cls(
            origins=origins,
            num_frames=n_frames,
            tiles_per_step=per,
            mode=mode,
            generator=gen,
        )
        if mode in ("epoch", "same_tile"):
            sampler._reshuffle_epoch()
        return sampler

    def _reshuffle_epoch(self) -> None:
        n = len(self.origins)
        self._spatial_order = torch.randperm(n, generator=self.generator).tolist()
        if self.mode == "same_tile":
            self._frame_ids = None
        else:
            self._frame_ids = torch.randint(
                0, self.num_frames, (n,), generator=self.generator
            ).tolist()
        self._cursor = 0

    def _sample_frame_ids(self, take: int) -> list[int]:
        if take <= self.num_frames:
            return torch.randperm(self.num_frames, generator=self.generator)[
                :take
            ].tolist()
        return torch.randint(
            0, self.num_frames, (take,), generator=self.generator
        ).tolist()

    def next_pairs(self) -> list[tuple[tuple[int, int], int]]:
        """Return ``tiles_per_step`` pairs of ``((row, col), frame_id)``."""
        take = self.tiles_per_step
        if not self.origins:
            return [((0, 0), 0)] * take
        if self.mode == "same_tile":
            if self._spatial_order is None or self._cursor >= len(self._spatial_order):
                self._reshuffle_epoch()
            assert self._spatial_order is not None
            origin = self.origins[self._spatial_order[self._cursor]]
            self._cursor += 1
            return [(origin, int(f)) for f in self._sample_frame_ids(take)]
        if self.mode == "iid":
            oi = torch.randint(
                0, len(self.origins), (take,), generator=self.generator
            ).tolist()
            fi = torch.randint(
                0, self.num_frames, (take,), generator=self.generator
            ).tolist()
            return [(self.origins[i], int(f)) for i, f in zip(oi, fi)]
        if self._spatial_order is None or self._frame_ids is None:
            self._reshuffle_epoch()
        assert self._spatial_order is not None and self._frame_ids is not None
        if self._cursor + take > len(self._spatial_order):
            self._reshuffle_epoch()
        chunk_s = self._spatial_order[self._cursor : self._cursor + take]
        chunk_f = self._frame_ids[self._cursor : self._cursor + take]
        self._cursor += take
        return [(self.origins[i], int(f)) for i, f in zip(chunk_s, chunk_f)]


def build_cross_frame_tile_sampler(dataset, args) -> CrossFrameTileSampler | None:
    tile = int(getattr(args, "lr_tile", 0) or 0)
    mix = resolve_lr_tile_mix(args)
    if tile <= 0 or not mix.startswith("cross_"):
        return None
    lr_h, lr_w = _dataset_lr_hw(dataset, args)
    num_frames = getattr(dataset, "num_samples", None)
    if num_frames is None:
        num_frames = len(dataset)
    num_frames = int(num_frames)
    mode = {
        "cross_epoch": "epoch",
        "cross_iid": "iid",
        "cross_same_tile": "same_tile",
    }[mix]
    return CrossFrameTileSampler.from_shapes(
        lr_h,
        lr_w,
        tile,
        tiles_per_step=raw_tiles_per_step(args),
        num_frames=num_frames,
        mode=mode,
        seed=int(getattr(args, "seed", 0) or 0),
    )


def _dataset_train_coords(dataset) -> torch.Tensor:
    """HR query coords used in training (same source as ``dataset[i]['input']``)."""
    if hasattr(dataset, "input_coords"):
        sf = float(getattr(dataset, "scale_factor", 1.0) or 1.0)
        cache = dataset.input_coords
        if sf in cache:
            coords = cache[sf]
        else:
            coords = next(iter(cache.values()))
    else:
        coords = dataset.get_hr_coordinates()
    if coords.dim() == 3:
        coords = coords.unsqueeze(0)
    return coords


def stack_cross_frame_tiles(
    dataset,
    pairs: Sequence[tuple[tuple[int, int], int]],
    tile: int,
    device: torch.device,
    train_masks: Sequence[torch.Tensor] | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Crop ``(origin, frame)`` pairs and concatenate on batch (one fused forward)."""
    coords_full = _dataset_train_coords(dataset).to(device=device, non_blocking=True)

    cs: list[torch.Tensor] = []
    lrs: list[torch.Tensor] = []
    ms: list[torch.Tensor] = []
    sids: list[torch.Tensor] = []
    for (row, col), fid in pairs:
        lr = dataset.get_lr_sample_hwc(int(fid))
        if lr.dim() == 3:
            lr = lr.unsqueeze(0)
        lr = lr.to(device=device, non_blocking=True)
        mask = None
        if train_masks is not None:
            mask = train_masks[int(fid)].to(device=device, non_blocking=True)
        c, t, m = crop_lr_hr_tensors(coords_full, lr, row, col, tile, mask=mask)
        cs.append(c)
        lrs.append(t)
        if m is not None:
            ms.append(m)
        sids.append(
            torch.tensor([int(fid)], device=device, dtype=torch.long)
        )
    n = len(pairs)
    return (
        torch.cat(cs, dim=0),
        torch.cat(lrs, dim=0),
        torch.cat(ms, dim=0) if ms else None,
        torch.cat(sids, dim=0),
        torch.zeros(n, device=device),
        torch.zeros(n, device=device),
    )
