"""Tiled HR rendering for scenes too large for a single forward."""

from __future__ import annotations

from typing import Iterator

import torch


def iter_hr_tile_windows(
    height: int, width: int, tile: int
) -> Iterator[tuple[int, int, int, int]]:
    """Yield ``(row0, col0, row1, col1)`` non-overlapping windows covering ``H×W``."""
    tile = max(1, int(tile))
    h, w = int(height), int(width)
    for r0 in range(0, h, tile):
        for c0 in range(0, w, tile):
            yield r0, c0, min(r0 + tile, h), min(c0 + tile, w)


def resolve_hr_render_tile(height: int, width: int, requested: int = 0) -> int:
    """Tile side for HR decode. ``0`` = auto (full if ≤2048², else 2048)."""
    req = int(requested or 0)
    if req > 0:
        return req
    h, w = int(height), int(width)
    if h * w <= 2048 * 2048:
        return max(h, w)
    return 2048


@torch.no_grad()
def render_hr_rgb_tiled(
    model: torch.nn.Module,
    hr_coords: torch.Tensor,
    sample_id: torch.Tensor,
    *,
    device: torch.device,
    tile: int = 2048,
    eval_autocast_dtype: torch.dtype | None = None,
    **fwd_kwargs,
) -> torch.Tensor:
    """Decode full HR RGB by tiles.

    Parameters
    ----------
    hr_coords
        ``[H,W,2]`` or ``[1,H,W,2]`` coordinate grid (any device).
    sample_id
        Frame index tensor (broadcast to each tile batch).

    Returns
    -------
    torch.Tensor
        ``[1, H, W, 3]`` float32 on ``device`` (model output space, pre-denorm).
    """
    coords = hr_coords
    if coords.dim() == 4:
        coords = coords[0]
    if coords.dim() != 3 or coords.shape[-1] != 2:
        raise ValueError(f"hr_coords must be HWC×2, got {tuple(coords.shape)}")
    h, w = int(coords.shape[0]), int(coords.shape[1])
    tile = resolve_hr_render_tile(h, w, tile)
    out = torch.empty(1, h, w, 3, device=device, dtype=torch.float32)
    sid = sample_id.to(device=device).reshape(-1)
    if sid.numel() == 0:
        sid = torch.zeros(1, device=device, dtype=torch.long)
    elif sid.numel() > 1:
        sid = sid[:1]

    was_training = model.training
    model.eval()
    try:
        for r0, c0, r1, c1 in iter_hr_tile_windows(h, w, tile):
            tile_coords = coords[r0:r1, c0:c1].to(device=device, non_blocking=True)
            if tile_coords.dim() == 3:
                tile_coords = tile_coords.unsqueeze(0)

            def _fwd():
                return model(
                    tile_coords,
                    sid,
                    scale_factor=1,
                    training=False,
                    **fwd_kwargs,
                )

            if eval_autocast_dtype is not None and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=eval_autocast_dtype):
                    pred, _ = _fwd()
            else:
                pred, _ = _fwd()
            if isinstance(pred, (tuple, list)):
                pred = pred[0]
            pred = pred[..., :3].float()
            out[:, r0:r1, c0:c1] = pred
            del pred, tile_coords
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        if was_training:
            model.train()
    return out
