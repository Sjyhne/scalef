from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RouterOutput:
    raw_support: torch.Tensor         # [N, H, W]
    normalized_support: torch.Tensor  # [N, H, W]
    coverage: torch.Tensor            # [H, W]
    warped_coords: torch.Tensor       # [N, H, W, 2]
    valid: torch.Tensor               # [N, H, W]


def apply_affine_to_grid(grid_xy: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """Apply 2x3 affine to grid [H, W, 2] -> [H, W, 2]."""
    h, w, _ = grid_xy.shape
    ones = torch.ones(h, w, 1, device=grid_xy.device, dtype=grid_xy.dtype)
    homo = torch.cat([grid_xy, ones], dim=-1)          # [H, W, 3]
    warped = torch.einsum("ij,hwj->hwi", A, homo)      # [H, W, 2]
    return warped


class ContributionRouter(nn.Module):
    """
    Alignment-aware support routing in canonical space.

    Minimal implementation uses geometric support only:
      support_i(x) = 1[warped_i(x) inside frame_i_domain]
    """

    def __init__(self, frame_shapes, normalized_coords: bool = False):
        super().__init__()
        self.frame_shapes = frame_shapes  # list[(H_lr, W_lr)]
        self.normalized_coords = bool(normalized_coords)

    @staticmethod
    def bilinear_sample_single(
        map_1chw: torch.Tensor,
        coords_xy: torch.Tensor,
        hw,
    ) -> torch.Tensor:
        # map_1chw: [1, 1, Hm, Wm], coords_xy: [H, W, 2] in pixel space
        hm, wm = hw
        x = coords_xy[..., 0]
        y = coords_xy[..., 1]
        gx = 2.0 * (x / max(wm - 1, 1)) - 1.0
        gy = 2.0 * (y / max(hm - 1, 1)) - 1.0
        grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)  # [1, H, W, 2]
        out = F.grid_sample(
            map_1chw,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        return out[0, 0]  # [H, W]

    def forward(
        self,
        canonical_grid: torch.Tensor,    # [H, W, 2]
        affines: torch.Tensor,           # [N, 2, 3] canonical -> frame coords
        quality_maps: Optional[torch.Tensor] = None,    # [N, 1, Hq, Wq]
        confidence_maps: Optional[torch.Tensor] = None, # [N, 1, Hc, Wc]
    ) -> RouterOutput:
        n = affines.shape[0]
        raw_support = []
        warped_coords = []
        valid_list = []

        for i in range(n):
            coords_i = apply_affine_to_grid(canonical_grid, affines[i])   # [H, W, 2]
            warped_coords.append(coords_i)

            hlr, wlr = self.frame_shapes[i]
            x = coords_i[..., 0]
            y = coords_i[..., 1]

            if self.normalized_coords:
                valid = (x >= 0.0) & (x <= 1.0) & (y >= 0.0) & (y <= 1.0)
            else:
                valid = (
                    (x >= 0.0) & (x <= wlr - 1) &
                    (y >= 0.0) & (y <= hlr - 1)
                )
            valid_list.append(valid)

            support = valid.float()
            if quality_maps is not None:
                q = self.bilinear_sample_single(
                    quality_maps[i : i + 1], coords_i, quality_maps.shape[-2:]
                )
                support = support * q.clamp_min(0.0)
            if confidence_maps is not None:
                c = self.bilinear_sample_single(
                    confidence_maps[i : i + 1], coords_i, confidence_maps.shape[-2:]
                )
                support = support * c.clamp_min(0.0)
            raw_support.append(support)

        raw_support = torch.stack(raw_support, dim=0)      # [N, H, W]
        warped_coords = torch.stack(warped_coords, dim=0)  # [N, H, W, 2]
        valid = torch.stack(valid_list, dim=0)             # [N, H, W]

        coverage = raw_support.sum(dim=0)                  # [H, W]
        normalized = raw_support / coverage.clamp_min(1e-6)

        return RouterOutput(
            raw_support=raw_support,
            normalized_support=normalized,
            coverage=coverage,
            warped_coords=warped_coords,
            valid=valid,
        )


def boxcar_downsample(x: torch.Tensor, scale: int) -> torch.Tensor:
    return F.avg_pool2d(x, kernel_size=scale, stride=scale)


def masked_boxcar_projection(pred_hr: torch.Tensor, mask_hr: torch.Tensor, scale: int):
    """
    pred_hr: [B, C, H_hr, W_hr]
    mask_hr: [B, 1, H_hr, W_hr]
    """
    num = boxcar_downsample(pred_hr * mask_hr, scale)
    den = boxcar_downsample(mask_hr, scale)
    pred_lr = num / den.clamp_min(1e-6)
    valid_lr = den > (1.0 - 1e-6)
    return pred_lr, valid_lr, den


def blended_probs(coverage: torch.Tensor, alpha: float = 0.8) -> torch.Tensor:
    p_cov = coverage.reshape(-1)
    p_cov = p_cov / p_cov.sum().clamp_min(1e-6)
    p_uni = torch.full_like(p_cov, 1.0 / p_cov.numel())
    return alpha * p_cov + (1.0 - alpha) * p_uni


def sample_coords_from_coverage(
    canonical_grid: torch.Tensor, coverage: torch.Tensor, n_samples: int, alpha: float = 0.8
):
    probs = blended_probs(coverage, alpha=alpha)
    idx = torch.multinomial(probs, num_samples=n_samples, replacement=True)
    flat = canonical_grid.reshape(-1, 2)
    return flat[idx], idx
