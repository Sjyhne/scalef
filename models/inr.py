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

    # Largest query-row count verified safe in one fused-MLP call (width 256 →
    # rows*width = 2**31). Above this the encode/decode pass is split.
    MAX_FORWARD_ROWS = 1 << 23

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
        self.max_forward_rows = self.MAX_FORWARD_ROWS

        self.num_samples = num_samples
        self.lr_degradation = "s2_psf"

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
        """Apply per-frame 2x3 affines to a ``[B,H,W,2]`` coordinate grid.

        Written as broadcast multiply-adds rather than a batched matmul: the
        matmul form makes the gradient w.r.t. ``A`` a GEMM whose reduction dim is
        H*W, which cuBLAS runs as a degenerate 3x2-output kernel (~124 ms/step at
        B=2, H=W=2048). It only shows up for frames whose affine is learnable,
        so frame 0 (frozen) never hit it.
        """
        x = coords[..., 0]
        y = coords[..., 1]
        a = A[:, 0, 0].view(-1, 1, 1)
        b = A[:, 0, 1].view(-1, 1, 1)
        c = A[:, 0, 2].view(-1, 1, 1)
        d = A[:, 1, 0].view(-1, 1, 1)
        e = A[:, 1, 1].view(-1, 1, 1)
        f = A[:, 1, 2].view(-1, 1, 1)
        return torch.stack((a * x + b * y + c, d * x + e * y + f), dim=-1)

    def apply_affine_matmul(self, coords, A):
        """Reference implementation kept for equivalence tests / profiling."""
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
        # Frame 0 is the canonical anchor: identity, and it must receive no
        # gradient, so those rows use constants instead of its parameters.
        weights = []
        biases = []
        for sid in idx.tolist():
            if sid == 0:
                weights.append(torch.ones(3, device=x.device))
                biases.append(torch.zeros(3, device=x.device))
                continue
            channels = self.color_transforms[int(sid)]
            weights.append(torch.cat([channels[c].weight.reshape(1) for c in range(3)]))
            biases.append(torch.cat([channels[c].bias.reshape(1) for c in range(3)]))
        weight = torch.stack(weights).view(-1, 1, 1, 3).to(x.dtype)
        bias = torch.stack(biases).view(-1, 1, 1, 3).to(x.dtype)
        return x * weight + bias

    def apply_color_transform_loop(self, x, sample_idx):
        """Reference implementation kept for equivalence tests / profiling."""
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

    def _project(self, coords, progress=None):
        if self.input_projection is None:
            return coords
        try:
            return self.input_projection(coords, progress=progress)
        except TypeError:
            return self.input_projection(coords)

    def _encode_and_decode(self, coords, sample_idx, progress=None, coord_query_scale=None):
        """Affine -> projection -> MLP. Returns (output, shifts)."""
        B, H, W, C = coords.shape
        A = self.get_direct_affine(sample_idx)
        dx_list = A[:, 0, 2]
        dy_list = A[:, 1, 2]
        coords = self.apply_affine(coords, A)
        if coord_query_scale is not None and abs(float(coord_query_scale) - 1.0) >= 1e-6:
            s = float(coord_query_scale)
            coords = 0.5 + (coords - 0.5) * s

        rows = B * H * W
        max_rows = int(getattr(self, "max_forward_rows", 0) or 0)
        if max_rows <= 0 or rows <= max_rows:
            coords = self._project(coords, progress=progress)
            output_flat = self.decoder(coords.reshape(rows, -1))
        else:
            # tinycudann's fused MLP indexes rows*width with 32 bits, so a single
            # call must stay under 2**32 elements or it faults with an illegal
            # memory access. Split the query rows and concatenate.
            coords_flat = coords.reshape(rows, C)
            parts = [
                self.decoder(self._project(chunk, progress=progress))
                for chunk in coords_flat.split(max_rows, dim=0)
            ]
            output_flat = torch.cat(parts, dim=0)
        output = output_flat.reshape(B, H, W, -1)

        shifts = [dx_list, dy_list]
        return output, shifts

    def forward(
        self,
        coords,
        sample_idx=None,
        lr_frames=None,
        progress=None,
        coord_query_scale=None,
        lr_align_args=None,
        **kwargs,
    ):
        del kwargs
        output, shifts = self._encode_and_decode(
            coords, sample_idx, progress=progress, coord_query_scale=coord_query_scale
        )
        output = output[:, :, :, :3]
        output = self.apply_color_transform(output, sample_idx)

        if lr_frames is not None:
            aargs = lr_align_args if lr_align_args is not None else default_lr_align_args(
                lr_degradation=getattr(self, "lr_degradation", "s2_psf")
            )
            output = align_prediction_hwc_to_target(
                output, lr_frames, args=aargs, device=output.device
            )

        return output, shifts


class INRGNLL(INRBase):
    """
    INR with heteroscedastic uncertainty: decoder outputs rgb + log-scale per frame.
    Used for Gaussian NLL (L2) or Laplace NLL (L1).
    """

    def __init__(
        self,
        input_projection,
        decoder,
        num_samples,
        coordinate_dim=2,
        hetero_loss_type: str = "gaussian",
        hetero_scale: str = "pixel",
        hetero_region_size: int = 4,
    ):
        super().__init__(
            input_projection=input_projection,
            decoder=decoder,
            num_samples=num_samples,
            coordinate_dim=coordinate_dim,
        )
        self.use_gnll = True
        self.hetero_loss_type = str(hetero_loss_type).lower()
        self.hetero_scale = str(hetero_scale).lower()
        self.region_size = max(1, int(hetero_region_size))
        self.log_scales = None
        if self.hetero_scale == "frame":
            self.log_scales = nn.Parameter(torch.zeros(num_samples))

    def _region_grid_size(self, height: int, width: int) -> tuple[int, int]:
        rh = rw = self.region_size
        return max(1, (height + rh - 1) // rh), max(1, (width + rw - 1) // rw)

    def _ensure_region_log_scales(self, height: int, width: int, device, dtype):
        n_rh, n_rw = self._region_grid_size(height, width)
        shape = (self.num_samples, n_rh, n_rw)
        if self.log_scales is None:
            self.log_scales = nn.Parameter(torch.zeros(shape, device=device, dtype=dtype))
            return
        if tuple(self.log_scales.shape) != shape:
            raise ValueError(
                f"Region hetero scales shape {tuple(self.log_scales.shape)} != expected {shape}."
            )

    def _region_pixel_scales(self, sample_idx: torch.Tensor, height: int, width: int, dtype):
        idx = sample_idx.reshape(-1).long().clamp(0, self.num_samples - 1)
        device = idx.device
        self._ensure_region_log_scales(height, width, device, dtype)
        frame_scales = torch.exp(self.log_scales[idx])
        row_idx = torch.arange(height, device=device) // self.region_size
        col_idx = torch.arange(width, device=device) // self.region_size
        return frame_scales[:, row_idx][:, :, col_idx]

    def forward(
        self,
        coords,
        sample_idx=None,
        lr_frames=None,
        progress=None,
        lr_align_args=None,
        **kwargs,
    ):
        coord_query_scale = kwargs.pop("coord_query_scale", None)
        del kwargs
        raw, shifts = self._encode_and_decode(
            coords, sample_idx, progress=progress, coord_query_scale=coord_query_scale
        )

        logvars = None
        if self.hetero_scale == "pixel" and raw.shape[-1] >= 3 + self.num_samples * 3:
            logvars_list = []
            for i in range(self.num_samples):
                logvars_list.append(raw[:, :, :, 3 + i * 3 : 6 + i * 3])
            logvars = torch.stack(logvars_list, dim=0)
        output = raw[:, :, :, :3]
        output = self.apply_color_transform(output, sample_idx)

        if not (self.use_gnll and lr_frames is not None):
            return output, shifts

        aargs = lr_align_args if lr_align_args is not None else default_lr_align_args(
            lr_degradation=getattr(self, "lr_degradation", "s2_psf")
        )
        output = align_prediction_hwc_to_target(
            output, lr_frames, args=aargs, device=output.device
        )

        B, H, W, C = output.shape
        if self.hetero_scale == "frame":
            idx = sample_idx.reshape(-1).long().clamp(0, self.num_samples - 1)
            scales = torch.exp(self.log_scales[idx]).view(B, 1, 1, 1)
            variances = scales.expand(B, H, W, C)
            return output, shifts, variances

        if self.hetero_scale == "region":
            pixel_scales = self._region_pixel_scales(sample_idx, H, W, output.dtype)
            variances = pixel_scales.unsqueeze(-1).expand(B, H, W, C)
            return output, shifts, variances

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
            variances = torch.ones(B, H, W, C, device=output.device) * 0.1

        return output, shifts, variances


def get_inr(
    input_projection,
    decoder,
    num_samples,
    use_gnll=False,
    use_laplace_nll=False,
    hetero_scale: str = "pixel",
    hetero_region_size: int = 4,
    coordinate_dim=2,
    **kwargs,
):
    """Build INRBase or INRGNLL. Extra kwargs are ignored."""
    del kwargs
    if use_gnll and use_laplace_nll:
        raise ValueError("use_gnll and use_laplace_nll are mutually exclusive.")
    if use_gnll or use_laplace_nll:
        hetero_loss_type = "laplace" if use_laplace_nll else "gaussian"
        return INRGNLL(
            input_projection,
            decoder,
            num_samples,
            coordinate_dim=coordinate_dim,
            hetero_loss_type=hetero_loss_type,
            hetero_scale=str(hetero_scale).lower(),
            hetero_region_size=int(hetero_region_size),
        )
    return INRBase(
        input_projection,
        decoder,
        num_samples,
        coordinate_dim=coordinate_dim,
    )


INR = INRBase
