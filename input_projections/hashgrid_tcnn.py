
"""
Hash grid encoding via tiny-cuda-nn (tcnn). Standard multiresolution hash encoding;
forward matches other projections: ``forward(x, fs=None, progress=None)`` (extra args ignored).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

try:
    import tinycudann as tcnn
except ImportError:
    tcnn = None


def _per_level_scale(base_resolution: int, max_resolution: int, n_levels: int) -> float:
    if n_levels <= 1:
        return 1.0
    return math.exp(math.log(max_resolution / base_resolution) / (n_levels - 1))


class HashGridTcnn(nn.Module):
    """Multiresolution hash grid via tiny-cuda-nn ``Encoding`` (HashGrid / Grid+Hash).

    The field is defined on the base-frame unit square. Coordinates passed to tcnn are clamped
    to ``[0, 1)``; training masks LR pixels whose affine-warped HR footprint leaves the unit square.

    ``encoding_preset``:
    - ``hashgrid`` (default): HashGrid with ``Linear`` interpolation and ``per_level_scale``
      derived from ``base_resolution`` / ``max_resolution``.
    - ``smoothstep_grid``: ``Grid`` with configurable ``type`` (Hash/Dense/Tiled) and
      ``Smoothstep`` interpolation, using the same hash hyperparameters and geometric
      finest-level rule.
    """

    def __init__(
        self,
        input_dim=2,
        n_levels=16,
        n_features_per_level=2,
        log2_hashmap_size=21,
        base_resolution=16,
        max_resolution=2048,
        output_dtype="fp32",
        device=None,
        encoding_preset: str = "hashgrid",
        grid_type: str = "Hash",
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

        preset = str(encoding_preset or "hashgrid").lower().strip()
        self.encoding_preset = preset
        self.grid_type = str(grid_type or "Hash").strip().capitalize()
        if self.grid_type not in {"Hash", "Dense", "Tiled"}:
            raise ValueError("grid_type must be one of: Hash, Dense, Tiled")

        if device is not None and getattr(device, "type", "") == "cuda":
            torch.cuda.set_device(device.index)

        if preset == "smoothstep_grid":
            # Grid + configurable backing storage + Smoothstep; use CLI hash hyperparameters.
            br = int(base_resolution)
            nl = self.n_levels
            mr = int(max_resolution)
            if mr <= br:
                mr = br * 32  # sane floor if CLI is mis-set
            per_level_scale_smooth = _per_level_scale(br, mr, nl)
            self.output_dim = self.n_levels * self.n_features_per_level
            self.projection_output_dim = self.output_dim
            encoding_config = {
                "otype": "Grid",
                "type": self.grid_type,
                "n_levels": nl,
                "n_features_per_level": self.n_features_per_level,
                "base_resolution": br,
                "per_level_scale": per_level_scale_smooth,
                "interpolation": "Smoothstep",
            }
            if self.grid_type == "Hash":
                encoding_config["log2_hashmap_size"] = int(log2_hashmap_size)
            self._encoding = tcnn.Encoding(2, encoding_config)
        else:
            per_level_scale = _per_level_scale(int(base_resolution), int(max_resolution), self.n_levels)
            encoding_config = {
                "otype": "HashGrid",
                "n_levels": self.n_levels,
                "n_features_per_level": self.n_features_per_level,
                "log2_hashmap_size": int(log2_hashmap_size),
                "base_resolution": int(base_resolution),
                "per_level_scale": per_level_scale,
                "interpolation": "Linear",
            }
            try:
                self._encoding = tcnn.Encoding(2, encoding_config)
            except Exception:
                encoding_config["otype"] = "Grid"
                encoding_config["type"] = "Hash"
                encoding_config.pop("interpolation", None)
                self._encoding = tcnn.Encoding(2, encoding_config)

        if device is not None:
            self.to(device)

    def forward(self, x, fs=None, progress=None):
        del fs, progress
        orig_shape = x.shape[:-1]
        x = x.reshape(-1, 2).contiguous()
        t = x.clamp(0.0, 1.0 - 1e-6)
        y = self._encoding(t)
        if self.output_dtype == "fp32" and y.dtype != torch.float32:
            y = y.float()
        return y.reshape(*orig_shape, self.output_dim)
