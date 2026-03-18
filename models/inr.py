import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils import get_learnable_transforms


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
    No GNLL. Core logic lives here so changes apply to INRGNLL as well.
    """

    def __init__(
        self,
        input_projection,
        decoder,
        num_samples,
        coordinate_dim=2,
    ):
        super().__init__()

        self.input_projection = input_projection
        self.decoder = decoder
        self.coordinate_dim = coordinate_dim

        self.log_fs = nn.Parameter(torch.full((1, 1), math.log(5.0)))
        self.log_fs.requires_grad = True

        self.num_samples = num_samples
        self.time_vectors = torch.FloatTensor(np.linspace(0, 1, self.num_samples))

        self.use_gnll = False

        self.shift_vectors = get_learnable_transforms(
            num_samples=num_samples,
            coordinate_dim=coordinate_dim,
            zeros=True,
            freeze_first=False,
        )
        self.rotation_angle = get_learnable_transforms(
            num_samples=num_samples,
            coordinate_dim=1,
            zeros=True,
            freeze_first=False,
        )

        self.color_transforms = nn.ModuleList(
            [nn.ModuleList([ChannelAffine1x1() for _ in range(3)]) for _ in range(num_samples)]
        )

    def get_affine_transform(self, sample_id):
        if self.time_vectors.device != sample_id.device:
            self.time_vectors = self.time_vectors.to(sample_id.device)
        return self.get_direct_affine(sample_id)

    def get_direct_affine(self, sample_id):
        B = sample_id.shape[0]
        shifts = []
        angles = []
        for i, idx in enumerate(sample_id):
            shifts.append(self.shift_vectors[idx.item()])
            angles.append(self.rotation_angle[idx.item()])
        shift = torch.stack(shifts).squeeze(1)
        angle = torch.stack(angles).squeeze(1)
        a1 = torch.stack([torch.cos(angle[:, 0]), -torch.sin(angle[:, 0]), shift[:, 0]], dim=1)
        a2 = torch.stack([torch.sin(angle[:, 0]), torch.cos(angle[:, 0]), shift[:, 1]], dim=1)
        A = torch.stack([a1, a2], dim=1)
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
        result = x.clone()
        for i, idx in enumerate(sample_idx):
            if idx != 0:
                for channel in range(3):
                    transformed = self.color_transforms[idx][channel](
                        x[i, :, :, channel].unsqueeze(-1)
                    )
                    result[i, :, :, channel] = transformed.squeeze(-1)
        return result

    def _encode_and_decode(self, coords, sample_idx, training):
        """Core path: affine -> projection -> decoder. Returns (output, shifts)."""
        B, H, W, C = coords.shape
        A = self.get_affine_transform(sample_idx)
        dx_list = A[:, 0, 2]
        dy_list = A[:, 1, 2]
        coords = self.apply_affine(coords, A)

        if self.input_projection is not None:
            coords = self.input_projection(coords)

        coords_flat = coords.reshape(B * H * W, -1)
        output_flat = self.decoder(coords_flat)
        output = output_flat.reshape(B, H, W, -1)

        shifts = [dx_list, dy_list]
        return output, shifts

    def forward(self, coords, sample_idx=None, scale_factor=None, training=True, lr_frames=None):
        output, shifts = self._encode_and_decode(coords, sample_idx, training)
        output = output[:, :, :, :3]
        output = self.apply_color_transform(output, sample_idx)

        if training and scale_factor is not None:
            if scale_factor.unique().shape[0] == 1:
                scale_factor = scale_factor.unique().item()
            else:
                raise ValueError("Not implemented: multiple scale factors in the same batch")
            output = F.interpolate(
                output.permute(0, 3, 1, 2),
                scale_factor=scale_factor,
                mode="area",
            ).permute(0, 2, 3, 1)

        return output, shifts


class INRGNLL(INRBase):
    """
    INR with Gaussian NLL: variance from decoder output (rgb + logvars per frame).
    Reuses _encode_and_decode from INRBase so core logic changes apply here too.
    """

    def __init__(
        self,
        input_projection,
        decoder,
        num_samples,
        coordinate_dim=2,
    ):
        super().__init__(
            input_projection=input_projection,
            decoder=decoder,
            num_samples=num_samples,
            coordinate_dim=coordinate_dim,
        )
        self.use_gnll = True

    def forward(self, coords, sample_idx=None, scale_factor=None, training=True, lr_frames=None):
        output, shifts = self._encode_and_decode(coords, sample_idx, training)

        logvars = None
        if output.shape[-1] >= 3 + self.num_samples * 3:
            rgb = output[:, :, :, :3]
            logvars_list = []
            for i in range(self.num_samples):
                logvars_list.append(output[:, :, :, 3 + i * 3 : 6 + i * 3])
            logvars = torch.stack(logvars_list, dim=0)
            output = rgb
        else:
            output = output[:, :, :, :3]
        output = self.apply_color_transform(output, sample_idx)

        if training and scale_factor is not None:
            if scale_factor.unique().shape[0] == 1:
                scale_factor = scale_factor.unique().item()
            else:
                raise ValueError("Not implemented: multiple scale factors in the same batch")
            output = F.interpolate(
                output.permute(0, 3, 1, 2),
                scale_factor=scale_factor,
                mode="area",
            ).permute(0, 2, 3, 1)

        if not (self.use_gnll and lr_frames is not None):
            return output, shifts

        B, H, W, _ = output.shape
        if logvars is not None and logvars.shape[1] > 0:
            selected_logvars = []
            for b in range(logvars.shape[1]):
                frame_idx = (sample_idx[b] % self.num_samples).item()
                frame_idx = min(max(0, frame_idx), self.num_samples - 1)
                selected_logvars.append(logvars[frame_idx, b])
            variances = torch.stack(selected_logvars, dim=0)
            variances = F.interpolate(
                variances.permute(0, 3, 1, 2), scale_factor=scale_factor
            ).permute(0, 2, 3, 1)
            variances = torch.exp(variances)
        else:
            variances = torch.ones(B, H, W, 3, device=output.device) * 0.1

        return output, shifts, variances


def get_inr(
    input_projection,
    decoder,
    num_samples,
    use_gnll=False,
    coordinate_dim=2,
    **kwargs,
):
    """Build INRBase or INRGNLL. Extra kwargs are ignored."""
    if use_gnll:
        return INRGNLL(
            input_projection,
            decoder,
            num_samples,
            coordinate_dim=coordinate_dim,
        )
    return INRBase(
        input_projection,
        decoder,
        num_samples,
        coordinate_dim=coordinate_dim,
    )


# Backward compatibility: INR refers to base; use get_inr(..., use_gnll=True) or INRGNLL for GNLL.
INR = INRBase
