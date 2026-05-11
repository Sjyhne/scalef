"""Per-frame per-band radiometric gain (and optional offset) on RGB predictions (plan §7)."""

from __future__ import annotations

import torch
import torch.nn as nn


class FrameRadiometric(nn.Module):
    """``gain = 1 + 0.1*tanh(raw_gain)``, ``offset = 0.05*tanh(raw_offset)`` per frame × band."""

    def __init__(self, num_samples: int, num_bands: int = 3, mode: str = "gain_offset"):
        super().__init__()
        mode = str(mode).lower().strip()
        if mode not in {"gain", "gain_offset"}:
            raise ValueError("mode must be 'gain' or 'gain_offset'")
        self.mode = mode
        self.raw_gain = nn.Parameter(torch.zeros(int(num_samples), int(num_bands)))
        self.raw_offset = (
            nn.Parameter(torch.zeros(int(num_samples), int(num_bands))) if mode == "gain_offset" else None
        )

    def regularization(self) -> torch.Tensor:
        g = torch.tanh(self.raw_gain)
        loss = (g**2).mean()
        if self.raw_offset is not None:
            loss = loss + (torch.tanh(self.raw_offset) ** 2).mean()
        return loss

    def forward(self, x: torch.Tensor, sample_idx: torch.Tensor) -> torch.Tensor:
        """Apply per-batch-item frame index from ``sample_idx`` (shape ``[B]``)."""
        if sample_idx is None:
            return x
        sid = sample_idx.long().view(-1).clamp(0, self.raw_gain.shape[0] - 1)
        gain = 1.0 + 0.1 * torch.tanh(self.raw_gain)
        g = gain[sid].unsqueeze(1).unsqueeze(1)
        y = x * g
        if self.raw_offset is not None:
            off = 0.05 * torch.tanh(self.raw_offset[sid].unsqueeze(1).unsqueeze(1))
            y = y + off
        return y
