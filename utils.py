import torch
import torch.nn.functional as F
import cv2
import numpy as np


def bilinear_resize_torch(image, size, antialiasing=True):
    return F.interpolate(image, size=size, mode="bilinear", align_corners=False, antialias=antialiasing)


def align_spatial(input: torch.Tensor, reference: torch.Tensor, mode: str = "ECC") -> torch.Tensor:
    """Align input to reference via ECC (affine). Both (3, H, W) RGB."""
    if mode != "ECC":
        raise NotImplementedError(f"Mode '{mode}' is not implemented.")

    input_np = (input.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    reference_np = (reference.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    input_gray = cv2.cvtColor(input_np, cv2.COLOR_RGB2GRAY)
    reference_gray = cv2.cvtColor(reference_np, cv2.COLOR_RGB2GRAY)
    warp_mode = cv2.MOTION_AFFINE
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-10)
    _, warp_matrix = cv2.findTransformECC(reference_gray, input_gray, warp_matrix, warp_mode, criteria)
    aligned_img_np = cv2.warpAffine(
        input_np,
        warp_matrix,
        (reference_np.shape[1], reference_np.shape[0]),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
    )
    aligned_img_tensor = torch.tensor(aligned_img_np, dtype=torch.float32) / 255.0
    return aligned_img_tensor.permute(2, 0, 1)


def align_spectral(input: torch.Tensor, reference: torch.Tensor, mode: str = "shift_scale") -> torch.Tensor:
    """Match input per-channel mean/std to reference. Both (3, H, W)."""
    if mode != "shift_scale":
        raise NotImplementedError(f"Mode '{mode}' is not implemented.")

    input_mean, input_std = input.mean(dim=(1, 2), keepdim=True), input.std(dim=(1, 2), keepdim=True)
    reference_mean, reference_std = reference.mean(dim=(1, 2), keepdim=True), reference.std(dim=(1, 2), keepdim=True)
    adjusted_input = (input - input_mean) / (input_std + 1e-6) * reference_std + reference_mean
    return torch.clamp(adjusted_input, 0, 1)


def align_output_to_target(
    input: torch.Tensor, reference: torch.Tensor, spectral: bool = True, spatial: bool = True
) -> torch.Tensor:
    """Spectral (color) then spatial (ECC) alignment to reference. (3, H, W)."""
    input = input.squeeze(0)
    reference = reference.squeeze(0)
    aligned = align_spectral(input, reference, mode="shift_scale") if spectral else input
    if spatial:
        aligned = align_spatial(aligned, reference, mode="ECC")
    return aligned.unsqueeze(0).cuda()


def get_valid_mask(input: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Mask where both input and reference have all channels > 0. (1, 3, H, W) -> (1, 1, H, W)."""
    input = input.squeeze(0)
    reference = reference.squeeze(0)
    input_valid = (input > 0).all(dim=0)
    reference_valid = (reference > 0).all(dim=0)
    return (input_valid & reference_valid)[None, None, ...]
