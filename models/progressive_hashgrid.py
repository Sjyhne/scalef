"""Hard schedule for progressively activating HashGrid levels (TTO-only SuperF plan)."""

from __future__ import annotations

import torch


def active_level_count(iteration: int, n_levels: int) -> int:
    """Piecewise-constant active coarse levels; see ``tto_only_superf_improvement_plan.md`` §6.3."""
    if iteration < 300:
        active = 4
    elif iteration < 700:
        active = 6
    elif iteration < 1200:
        active = 9
    else:
        active = int(n_levels)
    return min(int(active), int(n_levels))


def level_mask_vector(iteration: int, n_levels: int, n_features: int, *, device: torch.device) -> torch.Tensor:
    """Return ``[L * F]`` mask (values 0 or 1) for level-major flattened hash features."""
    k = active_level_count(iteration, n_levels)
    m = torch.zeros(int(n_levels), device=device, dtype=torch.float32)
    m[:k] = 1.0
    return m.repeat_interleave(int(n_features))
