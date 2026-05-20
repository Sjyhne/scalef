import math

import torch
import torch.nn as nn


class HashLevelAttentionDecoderLite(nn.Module):
    """Lightweight coordinate-conditioned attention over HashGrid levels.

    Expected call signature:
        rgb, aux = decoder(coords, hash_features)

    coords:        [N, 2]
    hash_features: [N, L * F]
    rgb:           [N, out_dim]
  aux: diagnostics with detached attention stats
    """

    requires_coords = True

    def __init__(
        self,
        n_levels: int,
        n_features_per_level: int,
        token_dim: int = 32,
        hidden_dim: int = 64,
        out_dim: int = 3,
        coord_dim: int = 2,
    ):
        super().__init__()
        self.n_levels = int(n_levels)
        self.n_features_per_level = int(n_features_per_level)
        self.token_dim = int(token_dim)
        self.coord_dim = int(coord_dim)

        self.level_proj = nn.Linear(self.n_features_per_level, self.token_dim)
        self.level_embed = nn.Parameter(torch.zeros(1, self.n_levels, self.token_dim))

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

    def forward(self, coords: torch.Tensor, hash_features: torch.Tensor):
        if coords.ndim != 2 or coords.shape[-1] != self.coord_dim:
            raise ValueError(f"coords must be [N,{self.coord_dim}], got {tuple(coords.shape)}")
        if hash_features.ndim != 2:
            raise ValueError(f"hash_features must be [N,L*F], got {tuple(hash_features.shape)}")

        expected = self.n_levels * self.n_features_per_level
        if hash_features.shape[-1] != expected:
            raise ValueError(
                f"hash_features last dim must be {expected}, got {hash_features.shape[-1]}"
            )

        n = hash_features.shape[0]
        h = hash_features.reshape(n, self.n_levels, self.n_features_per_level)

        tokens = self.level_proj(h) + self.level_embed
        query = self.coord_query(coords).unsqueeze(1)

        logits = (tokens * query).sum(dim=-1) / math.sqrt(self.token_dim)
        attn = torch.softmax(logits, dim=-1)
        fused = (attn.unsqueeze(-1) * tokens).sum(dim=1)

        rgb = self.rgb_mlp(torch.cat([fused, coords], dim=-1))

        with torch.no_grad():
            entropy = -(attn.clamp_min(1e-8) * attn.clamp_min(1e-8).log()).sum(dim=-1)
            fine_k = min(3, self.n_levels)
            aux = {
                "attn": attn.detach(),
                "attn_mean": attn.detach().mean(dim=0),
                "attn_entropy_mean": entropy.detach().mean(),
                "attn_fine_mass_last3": attn[:, -fine_k:].detach().sum(dim=-1).mean(),
            }

        return rgb, aux
