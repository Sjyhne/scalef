"""Coordinate-conditioned attention over Fourier frequency bands.

Mirrors ``HashLevelAttentionDecoderLite`` but groups Fourier features into K
band tokens instead of HashGrid level tokens. ``FourierProjection`` lays out
its output as ``[sin_b0, ..., sin_b{N-1}, cos_b0, ..., cos_b{N-1}]`` where
``N = output_dim // 2``. We reshape that to ``[N_pix, 2, N]`` and split the
``N`` bands into ``K`` contiguous groups, then concatenate sin and cos for
each group so each token sees its own frequency band's sin/cos pair(s).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _pick_num_bands(n_bands: int, requested: int) -> int:
    """Return the largest divisor of ``n_bands`` that is <= ``requested``.

    Falls back through the suggested sequence (8 -> 4 -> 2 -> 1) when the
    explicit request does not divide ``n_bands`` cleanly. Always returns
    a value >= 1 that divides ``n_bands``.
    """
    if n_bands <= 0:
        raise ValueError(f"n_bands must be positive, got {n_bands}")
    candidates = [int(requested)] + [k for k in (8, 4, 2, 1) if k != int(requested)]
    for k in candidates:
        if k >= 1 and n_bands % k == 0:
            return k
    return 1


class FourierBandAttentionDecoderLite(nn.Module):
    """Lightweight coordinate-conditioned attention over Fourier bands.

    Expected call signature::

        rgb, aux = decoder(coords, fourier_features)

    coords:            [N, coord_dim]
    fourier_features:  [N, D]   with D = 2 * n_bands  (sin half, cos half)
    rgb:               [N, out_dim]
    aux: diagnostics with detached attention stats
    """

    requires_coords = True

    def __init__(
        self,
        input_dim: int,
        num_bands: int = 8,
        token_dim: int = 32,
        hidden_dim: int = 64,
        out_dim: int = 3,
        coord_dim: int = 2,
    ):
        super().__init__()
        if input_dim <= 0 or input_dim % 2 != 0:
            raise ValueError(
                f"FourierBandAttention expects even input_dim = 2*n_bands, got {input_dim}"
            )

        self.input_dim = int(input_dim)
        self.coord_dim = int(coord_dim)
        self.token_dim = int(token_dim)

        n_bands = self.input_dim // 2
        self.n_bands = int(n_bands)
        self.num_bands = _pick_num_bands(self.n_bands, num_bands)
        self.bands_per_group = self.n_bands // self.num_bands
        # Each token concatenates sin+cos for ``bands_per_group`` adjacent freqs.
        self.feat_per_token = 2 * self.bands_per_group

        self.band_proj = nn.Linear(self.feat_per_token, self.token_dim)
        self.band_embed = nn.Parameter(torch.zeros(1, self.num_bands, self.token_dim))

        self.coord_query = nn.Sequential(
            nn.Linear(self.coord_dim, self.token_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.token_dim, self.token_dim),
        )

        self.rgb_mlp = nn.Sequential(
            nn.Linear(self.token_dim + self.coord_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, coords: torch.Tensor, fourier_features: torch.Tensor):
        if coords.ndim != 2 or coords.shape[-1] != self.coord_dim:
            raise ValueError(f"coords must be [N,{self.coord_dim}], got {tuple(coords.shape)}")
        if fourier_features.ndim != 2 or fourier_features.shape[-1] != self.input_dim:
            raise ValueError(
                f"fourier_features must be [N,{self.input_dim}], got {tuple(fourier_features.shape)}"
            )

        n = fourier_features.shape[0]
        # [N, 2, n_bands]: split sin/cos halves and stack on a new axis.
        sin_part, cos_part = fourier_features.split(self.n_bands, dim=-1)
        sc = torch.stack((sin_part, cos_part), dim=1)  # [N, 2, n_bands]
        # Group adjacent bands -> [N, 2, num_bands, bands_per_group]
        sc = sc.view(n, 2, self.num_bands, self.bands_per_group)
        # Token features per band group = [sin_group, cos_group] concatenated.
        tokens_raw = sc.permute(0, 2, 1, 3).reshape(n, self.num_bands, self.feat_per_token)

        tokens = self.band_proj(tokens_raw) + self.band_embed
        query = self.coord_query(coords).unsqueeze(1)

        logits = (tokens * query).sum(dim=-1) / math.sqrt(self.token_dim)
        attn = torch.softmax(logits, dim=-1)
        fused = (attn.unsqueeze(-1) * tokens).sum(dim=1)

        rgb = self.rgb_mlp(torch.cat([fused, coords], dim=-1))

        with torch.no_grad():
            ap = attn.clamp_min(1e-8)
            entropy = -(ap * ap.log()).sum(dim=-1)
            fine_k = min(3, self.num_bands)
            aux = {
                "attn": attn.detach(),
                "attn_mean": attn.detach().mean(dim=0),
                "attn_entropy_mean": entropy.detach().mean(),
                "attn_fine_mass_last3": attn[:, -fine_k:].detach().sum(dim=-1).mean(),
            }

        return rgb, aux
