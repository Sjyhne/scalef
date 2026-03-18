import math

import torch
import torch.nn as nn


class HashGridProjection(nn.Module):
    """Instant-NGP style multi-level hash-grid encoding for 2D coordinates."""

    def __init__(
        self,
        input_dim=2,
        n_levels=16,
        n_features_per_level=2,
        log2_hashmap_size=19,
        base_resolution=16,
        max_resolution=2048,
        device=None,
    ):
        super().__init__()
        if input_dim != 2:
            raise ValueError("HashGridProjection currently supports input_dim=2 only.")

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
        b = self.base_resolution
        m = self.max_resolution
        if self.n_levels == 1:
            self.resolutions = torch.tensor([b], dtype=torch.long)
        else:
            # Geometric progression from base -> max
            s = math.exp(math.log(m / b) / (self.n_levels - 1))
            self.resolutions = torch.tensor(
                [int(math.floor(b * (s**i))) for i in range(self.n_levels)], dtype=torch.long
            )
            self.resolutions[-1] = m

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
            # coordinate in grid space
            gx = x[:, 0] * (r - 1)
            gy = x[:, 1] * (r - 1)
            x0 = torch.floor(gx).to(torch.int64)
            y0 = torch.floor(gy).to(torch.int64)
            x1 = torch.clamp(x0 + 1, max=r - 1)
            y1 = torch.clamp(y0 + 1, max=r - 1)

            wx = (gx - x0.to(gx.dtype)).unsqueeze(-1)
            wy = (gy - y0.to(gy.dtype)).unsqueeze(-1)

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
            outs.append(f)

        y = torch.cat(outs, dim=-1)
        return y.reshape(*orig_shape, self.output_dim)

