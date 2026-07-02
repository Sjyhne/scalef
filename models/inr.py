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
        self.lr_degradation = "area"

        self.use_gnll = False

        self.affine_params = get_learnable_affines(num_samples=num_samples, freeze_first=True)

        self.color_transforms = nn.ModuleList(
            [nn.ModuleList([ChannelAffine1x1() for _ in range(3)]) for _ in range(num_samples)]
        )

    def get_direct_affine(self, sample_id):
        B = sample_id.shape[0]
        params = [self.affine_params[int(idx.item())] for idx in sample_id.reshape(-1)]
        affine = torch.cat(params, dim=0)
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
        idx = sample_idx.reshape(-1).long()
        if (idx == 0).all():
            return x
        out = x
        for i in range(idx.shape[0]):
            sid = int(idx[i].item())
            if sid == 0:
                continue
            if out is x:
                out = x.clone()
            for channel in range(3):
                out[i, :, :, channel] = self.color_transforms[sid][channel](
                    x[i, :, :, channel].unsqueeze(-1)
                ).squeeze(-1)
        return out

    def clamp_reference_frame(self):
        """Keep reference frame (index 0) fixed as canonical anchor."""
        with torch.no_grad():
            self.affine_params[0].copy_(
                torch.tensor(
                    [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]],
                    device=self.affine_params[0].device,
                )
            )

    def _encode_and_decode(self, coords, sample_idx, progress=None):
        """Affine -> projection -> MLP. Returns (output, shifts)."""
        B, H, W, C = coords.shape
        A = self.get_direct_affine(sample_idx)
        dx_list = A[:, 0, 2]
        dy_list = A[:, 1, 2]
        coords = self.apply_affine(coords, A)

        if self.input_projection is not None:
            try:
                coords = self.input_projection(coords, progress=progress)
            except TypeError:
                coords = self.input_projection(coords)

        coords_flat = coords.reshape(B * H * W, -1)
        output_flat = self.decoder(coords_flat)
        output = output_flat.reshape(B, H, W, -1)

        shifts = [dx_list, dy_list]
        return output, shifts

    def forward(
        self,
        coords,
        sample_idx=None,
        lr_frames=None,
        progress=None,
        lr_align_args=None,
        **kwargs,
    ):
        del kwargs
        output, shifts = self._encode_and_decode(coords, sample_idx, progress=progress)
        output = output[:, :, :, :3]
        output = self.apply_color_transform(output, sample_idx)

        if lr_frames is not None:
            aargs = lr_align_args if lr_align_args is not None else default_lr_align_args(
                lr_degradation=getattr(self, "lr_degradation", "area")
            )
            output = align_prediction_hwc_to_target(
                output, lr_frames, args=aargs, device=output.device
            )

        return output, shifts


class INRGNLL(INRBase):
    """
    INR with Gaussian NLL: variance from decoder output (rgb + logvars per frame).
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

    def forward(
        self,
        coords,
        sample_idx=None,
        lr_frames=None,
        progress=None,
        lr_align_args=None,
        **kwargs,
    ):
        del kwargs
        output, shifts = self._encode_and_decode(coords, sample_idx, progress=progress)

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

        if not (self.use_gnll and lr_frames is not None):
            return output, shifts

        aargs = lr_align_args if lr_align_args is not None else default_lr_align_args(
            lr_degradation=getattr(self, "lr_degradation", "area")
        )
        output = align_prediction_hwc_to_target(
            output, lr_frames, args=aargs, device=output.device
        )

        B = output.shape[0]
        if logvars is not None and logvars.shape[1] > 0:
            selected_logvars = []
            for b in range(B):
                frame_idx = int(sample_idx.reshape(-1)[b].item())
                frame_idx = min(max(0, frame_idx), self.num_samples - 1)
                selected_logvars.append(logvars[frame_idx, b])
            variances = torch.stack(selected_logvars, dim=0)
            variances = align_prediction_hwc_to_target(
                variances, lr_frames, args=aargs, device=output.device
            )
            variances = torch.exp(variances)
        else:
            variances = torch.ones(B, *output.shape[1:], device=output.device) * 0.1

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
    del kwargs
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


INR = INRBase
