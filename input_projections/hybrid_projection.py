import inspect

import torch
import torch.nn as nn


class HybridProjection(nn.Module):
    """Concatenate multiple coordinate encodings evaluated on the same coordinates."""

    def __init__(self, *projections: nn.Module):
        super().__init__()
        if not projections:
            raise ValueError("HybridProjection requires at least one sub-projection.")

        self.projections = nn.ModuleList(projections)
        dims = []
        for proj in self.projections:
            dim = getattr(proj, "projection_output_dim", getattr(proj, "output_dim", None))
            if dim is None:
                raise ValueError(
                    f"Projection {proj.__class__.__name__} must define projection_output_dim or output_dim."
                )
            dims.append(int(dim))
        self.projection_output_dim = sum(dims)
        self.output_dim = self.projection_output_dim

    def _forward_one(self, proj: nn.Module, x, fs=None, progress=None):
        params = inspect.signature(proj.forward).parameters
        kwargs = {}
        if "fs" in params:
            kwargs["fs"] = fs
        if "progress" in params:
            kwargs["progress"] = progress
        return proj(x, **kwargs)

    def forward(self, x, fs=None, progress=None):
        outs = [self._forward_one(proj, x, fs=fs, progress=progress) for proj in self.projections]
        return torch.cat(outs, dim=-1)
