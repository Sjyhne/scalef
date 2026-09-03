"""
Hash grid encoding via tiny-cuda-nn (tcnn). Drop-in for HashGridProjection when tcnn is installed.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

try:
    import tinycudann as tcnn
except ImportError:
    tcnn = None

from input_projections.hashgrid_projection import hash_level_resolutions


def _per_level_scale(base_resolution: int, max_resolution: int, n_levels: int) -> float:
    if n_levels <= 1:
        return 1.0
    return math.exp(math.log(max_resolution / base_resolution) / (n_levels - 1))


class HashGridTcnn(nn.Module):
    """Hash grid encoding using tiny-cuda-nn. Same interface as HashGridProjection (forward(x, fs=None, progress=None)).

    tcnn applies the half-voxel per-level offset internally (``scale * x + 0.5``)
    for both linear and smoothstep modes.
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
        output_dtype="fp32",
        device=None,
    ):
        super().__init__()
        if tcnn is None:
            raise ImportError(
                "tinycudann is required for HashGridTcnn. Install with: "
                "pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"
            )
        if input_dim != 2:
            raise ValueError("HashGridTcnn currently supports input_dim=2 only.")

        self.input_dim = 2
        self.n_levels = int(n_levels)
        self.n_features_per_level = int(n_features_per_level)
        self.output_dim = self.n_levels * self.n_features_per_level
        self.projection_output_dim = self.output_dim

        self.output_dtype = str(output_dtype).lower()
        if self.output_dtype not in {"fp16", "fp32"}:
            raise ValueError("HashGridTcnn output_dtype must be 'fp16' or 'fp32'.")

        interp = str(interpolation).lower().strip()
        if interp == "smoothstep":
            tcnn_interp = "Smoothstep"
        elif interp == "linear":
            tcnn_interp = "Linear"
        else:
            raise ValueError(f"interpolation must be 'smoothstep' or 'linear', got {interpolation!r}")

        # tcnn's grid is isotropic over the unit square, but the resolution
        # actually realised along an axis is (level resolution x the extent of
        # coordinates fed along that axis). Building the ladder from the long
        # axis and squeezing the short axis into [0, short/long] therefore
        # reproduces a rectangular grid with square cells, with no tcnn change.
        # Squeezing (rather than stretching the long axis past 1) matters: the
        # coarse levels are stored densely and indexed directly, so inputs above
        # 1 would run off the end of the level.
        max_h = int(max_resolution_h) or int(max_resolution)
        max_w = int(max_resolution_w) or int(max_resolution)
        long_max = max(max_h, max_w)
        base_h = int(base_resolution_h) or int(base_resolution) or max(8, max_h // 4)
        base_w = int(base_resolution_w) or int(base_resolution) or max(8, max_w // 4)
        long_base = base_w if max_w >= max_h else base_h
        if long_base >= long_max:
            long_base = max(8, long_max // 4)

        # x is the width axis and y the height axis, matching HashGridProjection.
        self.register_buffer(
            "coord_scale",
            torch.tensor([max_w / long_max, max_h / long_max], dtype=torch.float32),
            persistent=False,
        )
        self.rectangular = max_h != max_w

        per_level_scale = _per_level_scale(long_base, long_max, self.n_levels)

        encoding_config = {
            "otype": "HashGrid",
            "n_levels": self.n_levels,
            "n_features_per_level": self.n_features_per_level,
            "log2_hashmap_size": int(log2_hashmap_size),
            "base_resolution": int(long_base),
            "per_level_scale": per_level_scale,
            "interpolation": tcnn_interp,
        }
        try:
            self._encoding = tcnn.Encoding(2, encoding_config)
        except Exception:
            encoding_config["otype"] = "Grid"
            encoding_config["type"] = "Hash"
            if tcnn_interp != "Linear":
                try:
                    self._encoding = tcnn.Encoding(2, encoding_config)
                except Exception:
                    encoding_config.pop("interpolation", None)
                    self._encoding = tcnn.Encoding(2, encoding_config)
            else:
                encoding_config.pop("interpolation", None)
                self._encoding = tcnn.Encoding(2, encoding_config)

        long_res = hash_level_resolutions(long_base, long_max, self.n_levels)
        # Reported resolutions use the geometric-mean cell size, folding in the
        # short-axis squeeze the same way HashGridProjection does for rect grids.
        aspect_geo = math.sqrt((max_h / long_max) * (max_w / long_max))
        self.resolutions = [max(1, int(r * aspect_geo)) for r in long_res]
        self.resolutions_w = [max(1, int(r * max_w / long_max)) for r in long_res]
        self.resolutions_h = [max(1, int(r * max_h / long_max)) for r in long_res]

        if device is not None:
            self.to(device)

    def forward(self, x, fs=None, progress=None):
        del fs, progress
        orig_shape = x.shape[:-1]
        x = x.reshape(-1, 2)
        x = x.clamp(0.0, 1.0 - 1e-6)
        if self.rectangular:
            x = x * self.coord_scale.to(x.dtype)
        y = self._encoding(x.contiguous())
        if self.output_dtype == "fp32" and y.dtype != torch.float32:
            y = y.float()
        return y.reshape(*orig_shape, self.output_dim)

