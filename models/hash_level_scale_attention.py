"""Per-level softmax or sigmoid gating over HashGrid features (gated concatenation)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class HashLevelScaleAttention(nn.Module):
    def __init__(
        self,
        n_levels: int,
        features_per_level: int,
        hidden_dim: int = 64,
        temperature: float = 1.0,
        mode: str = "softmax",
        use_layernorm: bool = True,
        use_coord_in_gate: bool = False,
        lowfreq_coord_pe: bool = False,
    ):
        super().__init__()
        mode = str(mode).lower().strip()
        if mode not in {"softmax", "sigmoid"}:
            raise ValueError("mode must be 'softmax' or 'sigmoid'")
        self.n_levels = int(n_levels)
        self.features_per_level = int(features_per_level)
        self.temperature = float(temperature)
        self.mode = mode
        self.use_coord_in_gate = bool(use_coord_in_gate)
        self.lowfreq_coord_pe = bool(lowfreq_coord_pe)
        self.coord_feat_dim = 6 if self.lowfreq_coord_pe else 2

        d_flat = self.n_levels * self.features_per_level
        self.norm = nn.LayerNorm(d_flat) if use_layernorm else nn.Identity()
        gate_in = d_flat + (self.coord_feat_dim if self.use_coord_in_gate else 0)
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_in, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), self.n_levels),
        )
        nn.init.zeros_(self.gate_mlp[-1].weight)
        nn.init.zeros_(self.gate_mlp[-1].bias)

    @staticmethod
    def _coord_features(q: torch.Tensor, lowfreq: bool) -> torch.Tensor:
        if not lowfreq:
            return q
        x = q[..., 0:1]
        y = q[..., 1:2]
        return torch.cat(
            [x, y, torch.sin(math.pi * x), torch.cos(math.pi * x), torch.sin(math.pi * y), torch.cos(math.pi * y)],
            dim=-1,
        )

    def forward(self, h: torch.Tensor, q_flat: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
        if h.ndim != 3:
            raise ValueError(f"Expected h [B,L,F], got {tuple(h.shape)}")
        b, l, f = h.shape
        if l != self.n_levels or f != self.features_per_level:
            raise ValueError(f"Bad shape {tuple(h.shape)}")
        if self.use_coord_in_gate:
            if q_flat is None or q_flat.shape != (b, 2):
                raise ValueError("use_coord_in_gate requires q_flat [B,2]")
            q_feat = self._coord_features(q_flat, self.lowfreq_coord_pe)
        h_flat = h.reshape(b, l * f)
        parts = [self.norm(h_flat)]
        if self.use_coord_in_gate:
            parts.append(q_feat)
        logits = self.gate_mlp(torch.cat(parts, dim=-1))
        if self.mode == "softmax":
            p = torch.softmax(logits / max(self.temperature, 1e-6), dim=-1)
            gates = p * float(self.n_levels)
        else:
            gates = 2.0 * torch.sigmoid(logits)
            p = gates / gates.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        z = (h * gates.unsqueeze(-1)).reshape(b, l * f)
        with torch.no_grad():
            ent = (-(p * (p + 1e-8).log()).sum(dim=-1)).mean().item()
            if l < 3:
                coarse = float(p.sum(dim=-1).mean().item())
                mid = 0.0
                fine = 0.0
            else:
                thirds = max(1, l // 3)
                coarse = float(p[:, :thirds].sum(dim=-1).mean().item())
                mid = float(p[:, thirds : 2 * thirds].sum(dim=-1).mean().item()) if 2 * thirds <= l else 0.0
                fine = float(p[:, 2 * thirds :].sum(dim=-1).mean().item()) if 2 * thirds < l else 0.0
        stats = {"mean_entropy": ent, "coarse_mass_mean": coarse, "mid_mass_mean": mid, "fine_mass_mean": fine}
        return z, stats
