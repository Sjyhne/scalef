import math

import torch
import torch.nn as nn

_HASH_INTERPOLATION_CHOICES = ("smoothstep", "linear")


def _smoothstep01(t: torch.Tensor) -> torch.Tensor:
    """NGP-style smooth Hermite blend on fractional coordinates in [0, 1]."""
    t = t.clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _hash_interp_weight(t: torch.Tensor, interpolation: str) -> torch.Tensor:
    mode = str(interpolation).lower().strip()
    if mode == "smoothstep":
        return _smoothstep01(t)
    if mode == "linear":
        return t
    raise ValueError(
        f"Unknown hashgrid interpolation {interpolation!r}. "
        f"Use one of: {', '.join(_HASH_INTERPOLATION_CHOICES)}."
    )


def _grid_coord(axis: torch.Tensor, resolution, interpolation: str) -> torch.Tensor:
    """Map a normalized axis in [0, 1] to continuous grid coordinates.

    ``resolution`` may be an int or a broadcastable tensor of per-level resolutions.
    """
    if torch.is_tensor(resolution):
        scale = (resolution - 1).to(axis.dtype)
    else:
        scale = float(resolution - 1)
    coord = axis * scale
    if str(interpolation).lower().strip() == "smoothstep":
        # NGP Appendix A: stagger each level by half a voxel (1/(2N) in [0, 1])
        # so smoothstep zero-derivatives do not align across levels. Matches tcnn's
        # `fma(scale, input, 0.5f)` when scale is the per-level voxel count.
        coord = coord + 0.5
    return coord


def hash_level_resolutions_rect(
    base_h: int, max_h: int, base_w: int, max_w: int, n_levels: int
) -> list[tuple[int, int]]:
    """Per-axis geometric level resolutions for a rectangular grid."""
    rh = hash_level_resolutions(base_h, max_h, n_levels)
    rw = hash_level_resolutions(base_w, max_w, n_levels)
    return list(zip(rh, rw))


def hash_level_resolutions(base_resolution: int, max_resolution: int, n_levels: int) -> list[int]:
    """NGP geometric level resolutions from base to max (last level exact)."""
    b, m, n = int(base_resolution), int(max_resolution), int(n_levels)
    if n == 1:
        return [b]
    s = math.exp(math.log(m / b) / (n - 1))
    res = [int(math.floor(b * (s**i))) for i in range(n)]
    res[-1] = m
    return res


class HashGridProjection(nn.Module):
    """Instant-NGP style multi-level hash-grid encoding for 2D coordinates.

    With ``interpolation='smoothstep'``, applies the Appendix A recommendation:
    smoothstep weights plus a half-voxel per-level coordinate offset so zero
    derivatives do not align across levels. ``linear`` uses multilinear weights
    with no offset (the paper's default for main results).
    """

    def __init__(
        self,
        input_dim=2,
        n_levels=16,
        n_features_per_level=2,
        log2_hashmap_size=19,
        base_resolution=16,
        base_resolution_h=0,
        base_resolution_w=0,
        max_resolution=2048,
        max_resolution_h=0,
        max_resolution_w=0,
        interpolation="smoothstep",
        device=None,
    ):
        super().__init__()
        if input_dim != 2:
            raise ValueError("HashGridProjection currently supports input_dim=2 only.")

        self.interpolation = str(interpolation).lower().strip()
        if self.interpolation not in _HASH_INTERPOLATION_CHOICES:
            raise ValueError(
                f"interpolation must be one of {_HASH_INTERPOLATION_CHOICES}, got {interpolation!r}"
            )
        self.input_dim = input_dim
        self.n_levels = int(n_levels)
        self.n_features_per_level = int(n_features_per_level)
        self.log2_hashmap_size = int(log2_hashmap_size)
        self.hashmap_size = 1 << self.log2_hashmap_size
        self.output_dim = self.n_levels * self.n_features_per_level
        self.projection_output_dim = self.output_dim

        if self.n_levels <= 0:
            raise ValueError("n_levels must be positive.")
        if self.n_features_per_level <= 0:
            raise ValueError("n_features_per_level must be positive.")

        # Support rectangular grids via per-axis max resolutions.
        # If only the isotropic max_resolution is given, both axes use it.
        max_h = int(max_resolution_h) if int(max_resolution_h) > 0 else int(max_resolution)
        max_w = int(max_resolution_w) if int(max_resolution_w) > 0 else int(max_resolution)
        # Per-axis base wins; then the isotropic base; else a quarter of max (4x span).
        base_h = int(base_resolution_h) or int(base_resolution) or max(8, max_h // 4)
        base_w = int(base_resolution_w) or int(base_resolution) or max(8, max_w // 4)
        if base_h >= max_h:
            base_h = max(8, max_h // 4)
        if base_w >= max_w:
            base_w = max(8, max_w // 4)
        self.rectangular = (max_h != max_w)
        self.base_resolution = base_h  # kept for compat / display
        self.max_resolution = max(max_h, max_w)

        if max_h <= 0 or max_w <= 0 or base_h <= 0 or base_w <= 0:
            raise ValueError("base_resolution and max_resolution must be positive.")

        if self.rectangular:
            rect_res = hash_level_resolutions_rect(base_h, max_h, base_w, max_w, self.n_levels)
            self.resolutions_h = torch.tensor([r[0] for r in rect_res], dtype=torch.long)
            self.resolutions_w = torch.tensor([r[1] for r in rect_res], dtype=torch.long)
            # resolutions kept as the geometric mean for display
            self.resolutions = torch.tensor(
                [int(math.sqrt(rh * rw)) for rh, rw in rect_res], dtype=torch.long
            )
        else:
            res = hash_level_resolutions(base_h, max_h, self.n_levels)
            self.resolutions = torch.tensor(res, dtype=torch.long)
            self.resolutions_h = self.resolutions
            self.resolutions_w = self.resolutions

        # Hash tables: one embedding table per level (hashmap_size x n_features)
        self.tables = nn.Parameter(
            torch.empty(self.n_levels, self.hashmap_size, self.n_features_per_level)
        )
        nn.init.uniform_(self.tables, a=-1e-4, b=1e-4)

        if device is not None:
            self.to(device)

    @staticmethod
    def _hash(ix: torch.Tensor, iy: torch.Tensor, size: int) -> torch.Tensor:
        # 2D spatial hash (cheap, works fine for this use case)
        # Use uint32 primes; wrap-around behavior via uint32 ops.
        ix = ix.to(torch.int64)
        iy = iy.to(torch.int64)
        h = (ix * 1_540_863 + iy * 1_125_899) % size
        return h

    def forward(self, x, fs=None, progress=None):
        del fs, progress
        orig_shape = x.shape[:-1]
        x = x.reshape(-1, 2).contiguous()
        x = x.clamp(0.0, 1.0 - 1e-6)
        n_pts = x.shape[0]
        device = x.device

        # All levels are evaluated in one batched pass: looping in Python costs
        # hundreds of tiny kernel launches per step and dominates runtime.
        rh = self.resolutions_h.to(device).view(-1, 1)
        rw = self.resolutions_w.to(device).view(-1, 1)

        # Dataset coordinate grid is (x, y) == (width axis, height axis), so the
        # W-resolution applies to x[:,0] and the H-resolution to x[:,1].
        gx = _grid_coord(x[:, 0].unsqueeze(0), rw, self.interpolation)
        gy = _grid_coord(x[:, 1].unsqueeze(0), rh, self.interpolation)
        x0 = torch.floor(gx).to(torch.int64)
        y0 = torch.floor(gy).to(torch.int64)
        x1 = torch.minimum(x0 + 1, rw - 1)
        y1 = torch.minimum(y0 + 1, rh - 1)

        wx = _hash_interp_weight(gx - x0.to(gx.dtype), self.interpolation).unsqueeze(-1)
        wy = _hash_interp_weight(gy - y0.to(gy.dtype), self.interpolation).unsqueeze(-1)

        # Offset each level into its own slice of the flattened table so all four
        # corners can be gathered with a single index_select per corner.
        level_offset = (
            torch.arange(self.n_levels, device=device, dtype=torch.int64) * self.hashmap_size
        ).view(-1, 1)
        flat = self.tables.reshape(self.n_levels * self.hashmap_size, self.n_features_per_level)

        def _gather(ix, iy):
            idx = (self._hash(ix, iy, self.hashmap_size) + level_offset).reshape(-1)
            return flat[idx].view(self.n_levels, n_pts, self.n_features_per_level)

        f00 = _gather(x0, y0)
        f10 = _gather(x1, y0)
        f01 = _gather(x0, y1)
        f11 = _gather(x1, y1)

        f0 = f00 * (1 - wx) + f10 * wx
        f1 = f01 * (1 - wx) + f11 * wx
        f = f0 * (1 - wy) + f1 * wy

        # [L, N, F] -> [N, L*F] keeps the per-level concatenation order.
        y = f.permute(1, 0, 2).reshape(n_pts, self.output_dim)
        return y.reshape(*orig_shape, self.output_dim)

