import torch
import torch.nn as nn

from models.lr_alignment import align_prediction_hwc_to_target, default_lr_align_args
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
    SuperF-style INR: per-frame affine alignment, input projection (Fourier or HashGrid),
    MLP decoder, optional per-frame color shift.
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

        self.num_samples = num_samples

        self.use_gnll = False

        self.affine_params = get_learnable_affines(num_samples=num_samples, freeze_first=True)

        self.color_transforms = nn.ModuleList(
            [nn.ModuleList([ChannelAffine1x1() for _ in range(3)]) for _ in range(num_samples)]
        )
        self.last_decoder_aux = {}

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
        """Affine -> projection -> MLP. Returns (output, shifts, warped_coords)."""
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

        n_pix = B * H * W
        raw_flat = projected.reshape(n_pix, -1)

        if getattr(self.decoder, "requires_coords", False):
            q_flat = warped_q.reshape(n_pix, -1)
            output_flat, aux = self.decoder(q_flat, raw_flat)
            self.last_decoder_aux = aux
        else:
            output_flat = self.decoder(raw_flat)
            self.last_decoder_aux = {}

        output = output_flat.reshape(B, H, W, -1)

        shifts = [dx_list, dy_list]
        return output, shifts, warped_coords

    def forward(
        self,
        coords,
        sample_idx=None,
        lr_frames=None,
        progress=None,
        lr_align_args=None,
        **kwargs,
    ):
        del kwargs  # callers may pass scale_factor, training (e.g. GNLL path in optimize.py)
        output, shifts, _ = self._encode_and_decode(coords, sample_idx, progress=progress)
        output = output[:, :, :, :3]
        output = self.apply_color_transform(output, sample_idx)

        if lr_frames is not None:
            aargs = lr_align_args if lr_align_args is not None else default_lr_align_args()
            output = align_prediction_hwc_to_target(
                output, lr_frames, args=aargs, device=output.device
            )

        return output, shifts


def get_inr(
    input_projection,
    decoder,
    num_samples,
    use_gnll=False,
    coordinate_dim=2,
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
    )


# Backward compatibility alias
INR = INRBase
