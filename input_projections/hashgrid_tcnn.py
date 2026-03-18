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


def _per_level_scale(base_resolution: int, max_resolution: int, n_levels: int) -> float:
    if n_levels <= 1:
        return 1.0
    return math.exp(math.log(max_resolution / base_resolution) / (n_levels - 1))


class HashGridTcnn(nn.Module):
    """Hash grid encoding using tiny-cuda-nn. Same interface as HashGridProjection (forward(x, fs=None, progress=None))."""

    def __init__(
        self,
        input_dim=2,
        n_levels=16,
        n_features_per_level=2,
        log2_hashmap_size=19,
        base_resolution=16,
        max_resolution=2048,
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
        x = x.clamp(0.0, 1.0 - 1e-6)
        y = self._encoding(x)
        if self.output_dtype == "fp32" and y.dtype != torch.float32:
            y = y.float()
        return y.reshape(*orig_shape, self.output_dim)

