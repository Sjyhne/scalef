import torch
import torch.nn as nn

from models.progressive_hashgrid import level_mask_vector
from models.utils import get_learnable_affines


class ChannelAffine1x1(nn.Module):
    """Per-channel affine transform with 2D parameters for strict Muon compatibility."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, 1))
        self.bias = nn.Parameter(torch.zeros(1, 1))

    def forward(self, x):
        return x * self.weight + self.bias


class INRBase(nn.Module):
    """
    Base INR: affine transform, input projection, decoder, color shift.
    Optional: hash-level scale gates, progressive HashGrid masking, per-frame radiometry.
    """

    def __init__(
        self,
        input_projection,
        decoder,
        num_samples,
        coordinate_dim=2,
        hash_scale_attention=None,
        hash_n_levels: int | None = None,
        hash_n_features_per_level: int | None = None,
        radiometric=None,
        hash_gate_warmup_iters: int = 0,
    ):
        super().__init__()

        self.input_projection = input_projection
        self.decoder = decoder
        self.coordinate_dim = coordinate_dim

        self.num_samples = num_samples

        self.use_gnll = False

        self.hash_scale_attention = hash_scale_attention
        self.hash_n_levels = int(hash_n_levels) if hash_n_levels is not None else None
        self.hash_n_features_per_level = (
            int(hash_n_features_per_level) if hash_n_features_per_level is not None else None
        )
        self.last_hash_scale_stats: dict | None = None
        self.radiometric = radiometric
        self.hash_gate_warmup_iters = int(hash_gate_warmup_iters)

        self.affine_params = get_learnable_affines(num_samples=num_samples, freeze_first=True)

        self.color_transforms = nn.ModuleList(
            [nn.ModuleList([ChannelAffine1x1() for _ in range(3)]) for _ in range(num_samples)]
        )

    def get_direct_affine(self, sample_id):
        B = sample_id.shape[0]
        params = [self.affine_params[idx.item()] for idx in sample_id]
        affine = torch.cat(params, dim=0)  # [B, 6]
        A = affine.view(B, 2, 3)
        assert A.shape == (B, 2, 3), f"A.shape: {A.shape}"
        return A

    def apply_affine(self, coords, A):
        B, H, W, C = coords.shape
        coords = coords.reshape(B, -1, C)
        homogenous_coords = torch.cat(
            [coords, torch.ones(B, coords.shape[1], 1, device=coords.device)], dim=2
        )
        transformed_coords = torch.matmul(homogenous_coords, A.mT)
        return transformed_coords.reshape(B, H, W, C)

    def apply_color_transform(self, x, sample_idx):
        if sample_idx is None:
            return x
        result = x.clone()
        for i, idx in enumerate(sample_idx):
            idx_i = int(idx.item()) if torch.is_tensor(idx) else int(idx)
            if idx_i != 0:
                for channel in range(3):
                    transformed = self.color_transforms[idx_i][channel](
                        x[i, :, :, channel].unsqueeze(-1)
                    )
                    result[i, :, :, channel] = transformed.squeeze(-1)
        return result

    def clamp_reference_frame(self):
        """
        Keep reference frame (index 0) fixed as canonical anchor.
        Safe to call after optimizer step.
        """
        with torch.no_grad():
            self.affine_params[0].copy_(
                torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]], device=self.affine_params[0].device)
            )

    def _encode_and_decode(self, coords, sample_idx, progress=None):
        """Core path: affine -> projection -> decoder. Returns (output, shifts, warped_coords)."""
        B, H, W, C = coords.shape
        A = self.get_direct_affine(sample_idx)
        dx_list = A[:, 0, 2]
        dy_list = A[:, 1, 2]
        warped_coords = self.apply_affine(coords, A)
        warped_q = warped_coords

        if self.input_projection is not None:
            projected = self.input_projection(warped_q, progress=progress)
        else:
            projected = warped_q

        if (
            progress
            and progress.get("progressive")
            and self.hash_n_levels is not None
            and self.hash_n_features_per_level is not None
        ):
            lv = level_mask_vector(
                int(progress.get("iteration", 0)),
                self.hash_n_levels,
                self.hash_n_features_per_level,
                device=projected.device,
            ).to(dtype=projected.dtype)
            projected = projected * lv.view(1, 1, 1, -1)

        self.last_hash_scale_stats = None
        n_pix = B * H * W
        q_flat = warped_q.reshape(n_pix, C)
        raw_flat = projected.reshape(n_pix, -1)

        iter_i = int(progress.get("iteration", 10**9)) if progress else 10**9
        warm = int(progress.get("hash_gate_warmup_iters", self.hash_gate_warmup_iters)) if progress else self.hash_gate_warmup_iters
        gate_bypass = self.hash_scale_attention is not None and warm > 0 and iter_i < warm

        if self.hash_scale_attention is not None and not gate_bypass:
            if self.hash_n_levels is None or self.hash_n_features_per_level is None:
                raise ValueError("hash_scale_attention needs hash_n_levels and hash_n_features_per_level")
            nl, nf = self.hash_n_levels, self.hash_n_features_per_level
            h_lv = projected.view(B, H, W, nl, nf).reshape(n_pix, nl, nf)
            use_q = self.hash_scale_attention.use_coord_in_gate
            z_flat, stats = self.hash_scale_attention(h_lv, q_flat if use_q else None)
            self.last_hash_scale_stats = stats
            output_flat = self.decoder(z_flat)
        else:
            output_flat = self.decoder(raw_flat)
        output = output_flat.reshape(B, H, W, -1)

        shifts = [dx_list, dy_list]
        return output, shifts, warped_coords

    def forward(
        self,
        coords,
        sample_idx=None,
        scale_factor=None,
        training=True,
        lr_frames=None,
        progress=None,
    ):
        output, shifts, warped_coords = self._encode_and_decode(coords, sample_idx, progress=progress)
        output = output[:, :, :, :3]
        output = self.apply_color_transform(output, sample_idx)
        if self.radiometric is not None:
            output = self.radiometric(output, sample_idx)

        if scale_factor is not None:
            if torch.is_tensor(scale_factor):
                if scale_factor.unique().numel() == 1 and float(scale_factor.unique().item()) == 1.0:
                    pass
                else:
                    raise ValueError(
                        "INRBase.forward(scale_factor!=1) is no longer supported. "
                        "Render at HR and downsample outside the model."
                    )
            else:
                if float(scale_factor) != 1.0:
                    raise ValueError(
                        "INRBase.forward(scale_factor!=1) is no longer supported. "
                        "Render at HR and downsample outside the model."
                    )

        return output, shifts


def get_inr(
    input_projection,
    decoder,
    num_samples,
    use_gnll=False,
    coordinate_dim=2,
    hash_scale_attention=None,
    hash_n_levels: int | None = None,
    hash_n_features_per_level: int | None = None,
    radiometric=None,
    hash_gate_warmup_iters: int = 0,
    **kwargs,
):
    """Build the simplified MSE INR. Extra kwargs are ignored for compatibility."""
    _ = use_gnll
    _ = kwargs
    return INRBase(
        input_projection,
        decoder,
        num_samples,
        coordinate_dim=coordinate_dim,
        hash_scale_attention=hash_scale_attention,
        hash_n_levels=hash_n_levels,
        hash_n_features_per_level=hash_n_features_per_level,
        radiometric=radiometric,
        hash_gate_warmup_iters=hash_gate_warmup_iters,
    )


# Backward compatibility alias
INR = INRBase
