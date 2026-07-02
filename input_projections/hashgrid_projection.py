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


def _grid_coord(axis: torch.Tensor, resolution: int, interpolation: str) -> torch.Tensor:
    """Map a normalized axis in [0, 1] to continuous grid coordinates."""
    scale = float(resolution - 1)
    coord = axis * scale
    if str(interpolation).lower().strip() == "smoothstep":
        # NGP Appendix A: stagger each level by half a voxel (1/(2N) in [0, 1])
        # so smoothstep zero-derivatives do not align across levels. Matches tcnn's
        # `fma(scale, input, 0.5f)` when scale is the per-level voxel count.
        coord = coord + 0.5
    return coord


def hash_level_resolutions(base_resolution: int, max_resolution: int, n_levels: int) -> list[int]:
    """NGP geometric level resolutions from base to max (last level exact)."""
    b, m, n = int(base_resolution), int(max_resolution), int(n_levels)
    if n == 1:
        return [b]
    s = math.exp(math.log(m / b) / (n - 1))
    res = [int(math.floor(b * (s**i))) for i in range(n)]
    res[-1] = m
    return res


def compute_level_footprint_weights(resolutions, level_sigma: float) -> torch.Tensor | None:
    """Zip-NeRF style anti-aliasing level weights: w_l = erf(1 / (sqrt(8) * sigma * N_l)).

    ``level_sigma`` is the supervision footprint std in normalized coordinates.
    Levels whose cell size is much finer than the footprint get weights → 0,
    suppressing frequencies the LR supervision cannot constrain. Returns None
    when ``level_sigma <= 0`` (disabled).
    """
    sigma = float(level_sigma)
    if sigma <= 0.0:
        return None
    n = torch.as_tensor(list(resolutions), dtype=torch.float32)
    return torch.erf(1.0 / (math.sqrt(8.0) * sigma * n))


class HashGridProjection(nn.Module):
    """Instant-NGP style multi-level hash-grid encoding for 2D coordinates.

    With ``interpolation='smoothstep'``, applies the Appendix A recommendation:
    smoothstep weights plus a half-voxel per-level coordinate offset so zero
    derivatives do not align across levels. ``linear`` uses multilinear weights
    with no offset (the paper's default for main results).

    ``level_sigma > 0`` enables Zip-NeRF style footprint downweighting of fine
    levels: features of level with resolution N_l are scaled by
    ``erf(1 / (sqrt(8) * level_sigma * N_l))``.
    """

    def __init__(
        self,
        input_dim=2,
        n_levels=16,
        n_features_per_level=2,
        log2_hashmap_size=19,
        base_resolution=16,
        max_resolution=2048,
        interpolation="smoothstep",
        level_sigma=0.0,
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
        self.base_resolution = int(base_resolution)
        self.max_resolution = int(max_resolution)
        self.output_dim = self.n_levels * self.n_features_per_level
        self.projection_output_dim = self.output_dim

        if self.n_levels <= 0:
            raise ValueError("n_levels must be positive.")
        if self.n_features_per_level <= 0:
            raise ValueError("n_features_per_level must be positive.")
        if self.base_resolution <= 0 or self.max_resolution <= 0:
            raise ValueError("base_resolution and max_resolution must be positive.")

        # Per-level grid resolutions
        self.resolutions = torch.tensor(
            hash_level_resolutions(self.base_resolution, self.max_resolution, self.n_levels),
            dtype=torch.long,
        )

        self.level_sigma = float(level_sigma)
        weights = compute_level_footprint_weights(self.resolutions.tolist(), self.level_sigma)
        if weights is not None:
            self.register_buffer("level_weights", weights)
        else:
            self.level_weights = None

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

        # For each level, bilinear interpolate features from hashed grid vertices.
        outs = []
        device = x.device
        for li in range(self.n_levels):
            r = int(self.resolutions[li].item())
            gx = _grid_coord(x[:, 0], r, self.interpolation)
            gy = _grid_coord(x[:, 1], r, self.interpolation)
            x0 = torch.floor(gx).to(torch.int64)
            y0 = torch.floor(gy).to(torch.int64)
            x1 = torch.clamp(x0 + 1, max=r - 1)
            y1 = torch.clamp(y0 + 1, max=r - 1)

            wx = _hash_interp_weight(gx - x0.to(gx.dtype), self.interpolation).unsqueeze(-1)
            wy = _hash_interp_weight(gy - y0.to(gy.dtype), self.interpolation).unsqueeze(-1)

            # hash four corners
            h00 = self._hash(x0, y0, self.hashmap_size)
            h10 = self._hash(x1, y0, self.hashmap_size)
            h01 = self._hash(x0, y1, self.hashmap_size)
            h11 = self._hash(x1, y1, self.hashmap_size)

            t = self.tables[li].to(device)
            f00 = t[h00]
            f10 = t[h10]
            f01 = t[h01]
            f11 = t[h11]

            f0 = f00 * (1 - wx) + f10 * wx
            f1 = f01 * (1 - wx) + f11 * wx
            f = f0 * (1 - wy) + f1 * wy
            if self.level_weights is not None:
                f = f * self.level_weights[li].to(f.dtype)
            outs.append(f)

        y = torch.cat(outs, dim=-1)
        return y.reshape(*orig_shape, self.output_dim)

