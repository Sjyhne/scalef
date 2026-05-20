import ctypes
import os
import site
from pathlib import Path


def _preload_bundled_nvjitlink() -> None:
    """Load wheel libnvJitLink before any other lib pins an older SONAME.

    If LD_LIBRARY_PATH lists /usr/local/cuda/lib64 first, the loader may bind
    libnvJitLink.so.12 there before nvidia-cusparse resolves its RUNPATH; that
    copy can lack __nvJitLinkComplete_12_4 and breaks `import torch`.
    RTLD_GLOBAL preload matches the fix in PyTorch 2.6+ slim wheels.
    """
    if os.name != "posix":
        return
    bases = list(site.getsitepackages())
    user = site.getusersitepackages()
    if user:
        bases.append(user)
    for base in bases:
        so = Path(base) / "nvidia" / "nvjitlink" / "lib" / "libnvJitLink.so.12"
        if not so.is_file():
            continue
        try:
            ctypes.CDLL(str(so), mode=ctypes.RTLD_GLOBAL)
        except OSError:
            return
        return


_preload_bundled_nvjitlink()

import argparse
import json
import random
import shlex
import sys
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchmetrics.functional.image import peak_signal_noise_ratio
from tqdm import tqdm

from data import get_dataset
from input_projections.utils import get_input_projection, normalize_input_projection_name
from losses import BasicLosses
from models.inr import get_inr
from models.utils import get_decoder
from optimizers import build_optimizer
from utils import (
    apply_eval_center_crop,
)


WORLDSTRAT_DATASETS = ("worldstrat_test", "worldstrat_sweet", "worldstrat_bitter")


def resolve_satburst_data_root(args) -> str:
    """Parent directory with one subfolder per scene; each scene contains ``scale_<df>_...``."""
    r = getattr(args, "satburst_data_root", None)
    if r is not None and str(r).strip():
        return str(r).strip().rstrip("/")
    if getattr(args, "dataset", "") == "satburst_real":
        return "data_real"
    return "data"


def resolve_worldstrat_data_root(args) -> str:
    """Parent directory with one subfolder per area (each has ``hr/`` and ``lr/``)."""
    r = getattr(args, "worldstrat_data_root", None)
    if r is not None and str(r).strip():
        return str(r).strip().rstrip("/")
    ds = getattr(args, "dataset", "")
    if ds == "worldstrat_sweet":
        return "worldstrat_datasets/worldstrat_sweet"
    if ds == "worldstrat_bitter":
        return "worldstrat_datasets/worldstrat_bitter"
    return "worldstrat_test_data"


def discover_worldstrat_sample_ids(data_root: str | Path) -> list[str]:
    """Area names under a WorldStrat root (each subdir must contain ``hr/`` and ``lr/``)."""
    root = Path(data_root)
    if not root.is_dir():
        return []
    out: list[str] = []
    for p in sorted(root.iterdir(), key=lambda x: x.name):
        if not p.is_dir() or p.name.startswith("."):
            continue
        if (p / "hr").is_dir() and (p / "lr").is_dir():
            out.append(p.name)
    return out


def satburst_scene_dir(args) -> str:
    root = resolve_satburst_data_root(args)
    return f"{root}/{args.sample_id}/scale_{args.df}_shift_{args.lr_shift:.1f}px_aug_{args.aug}"


def _lr_max_side_from_worldstrat_scene(args) -> int | None:
    """LR crop side used by ``WorldStratTestDataset`` (center crop, capped at 64)."""
    if getattr(args, "dataset", "") not in WORLDSTRAT_DATASETS:
        return None
    lr_dir = Path(resolve_worldstrat_data_root(args)) / str(args.sample_id) / "lr"
    if not lr_dir.is_dir():
        return None
    paths = sorted(lr_dir.glob("*.png"))
    if not paths:
        return None
    img = cv2.imread(str(paths[0]))
    if img is None:
        return None
    h, w = img.shape[:2]
    return min(min(h, w), 64)


def _lr_max_side_from_satburst_scene(args) -> int | None:
    """Largest LR side (max H, W) from ``transform_log.json`` for the current scene."""
    if getattr(args, "dataset", "") not in ("satburst_synth", "satburst_real"):
        return None
    tl_path = Path(satburst_scene_dir(args)) / "transform_log.json"
    if not tl_path.is_file():
        return None
    with tl_path.open("r") as f:
        log = json.load(f)
    sides: list[int] = []
    for v in log.values():
        sh = v.get("shape") if isinstance(v, dict) else None
        if isinstance(sh, (list, tuple)) and len(sh) >= 2:
            sides.append(max(int(sh[0]), int(sh[1])))
    return max(sides) if sides else None


def resolve_hash_grid_resolutions(args, *, canon: str | None = None) -> tuple[int, int]:
    """Coarsest and finest HashGrid level resolutions.

    For satburst scenes, when CLI values are 0 (default):
    - ``hash_max_resolution`` = max LR side from ``transform_log.json``
    - ``hash_base_resolution`` = max(8, LR side // 4) so the multires span scales with patch size

    Pass explicit positive values to override either knob independently.
    """
    if canon is None:
        canon, _ = normalize_input_projection_name(args.input_projection, args.fourier_scale)
    if canon != "hashgrid_tcnn":
        base = int(getattr(args, "hash_base_resolution", 0) or 0) or 8
        mx = int(getattr(args, "hash_max_resolution", 0) or 0) or 48
        return base, mx

    explicit_max = int(getattr(args, "hash_max_resolution", 0) or 0)
    explicit_base = int(getattr(args, "hash_base_resolution", 0) or 0)

    lr_side = _lr_max_side_from_satburst_scene(args)
    if lr_side is None:
        lr_side = _lr_max_side_from_worldstrat_scene(args)
    if explicit_max > 0:
        hash_max = explicit_max
    elif lr_side is not None and lr_side > 0:
        hash_max = int(lr_side)
    else:
        hash_max = 48

    if explicit_base > 0:
        hash_base = explicit_base
    elif lr_side is not None and lr_side > 0:
        hash_base = max(8, int(lr_side) // 4)
    else:
        hash_base = 8

    if hash_base >= hash_max:
        hash_base = max(8, hash_max // 4)
    return hash_base, hash_max


def resolve_hash_max_resolution(args, *, canon: str | None = None) -> int:
    """Finest HashGrid level (see ``resolve_hash_grid_resolutions``)."""
    return resolve_hash_grid_resolutions(args, canon=canon)[1]


import time

import lpips
from torchmetrics.functional.image import structural_similarity_index_measure as ssim


def _hwc_to_bchw(x: torch.Tensor) -> torch.Tensor:
    """Convert [B,H,W,C] or [H,W,C] to [B,C,H,W] on the same device (no host copy)."""
    if x.dim() == 3:
        x = x.unsqueeze(0)
    return x.permute(0, 3, 1, 2).contiguous()


def _bchw_to_hwc_np(t: torch.Tensor) -> np.ndarray:
    return np.clip(t.squeeze(0).permute(1, 2, 0).detach().cpu().numpy(), 0.0, 1.0)


def _psnr_bchw(pred_bchw: torch.Tensor, gt_bchw: torch.Tensor) -> float:
    return peak_signal_noise_ratio(
        pred_bchw.detach().cpu(), gt_bchw.detach().cpu(), data_range=1.0
    ).item()


def _save_comparison_quad(
    sample_dir: Path,
    filename: str,
    lr_hwc: np.ndarray,
    bilinear_hwc: np.ndarray,
    pred_hwc: np.ndarray,
    gt_hwc: np.ndarray,
    *,
    model_psnr: float,
    bilinear_psnr: float,
    banner: str = "",
) -> None:
    """2×2 panel: LR | bilinear, model | GT."""
    lr_hwc = np.clip(lr_hwc, 0.0, 1.0)
    bilinear_hwc = np.clip(bilinear_hwc, 0.0, 1.0)
    pred_hwc = np.clip(pred_hwc, 0.0, 1.0)
    gt_hwc = np.clip(gt_hwc, 0.0, 1.0)
    if lr_hwc.ndim == 3 and lr_hwc.shape[0] == 3:
        lr_hwc = np.transpose(lr_hwc, (1, 2, 0))

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    if banner:
        fig.suptitle(banner, fontsize=13, fontweight="bold", y=1.02)

    axes[0, 0].imshow(lr_hwc)
    axes[0, 0].set_title("Original LR Image", fontsize=14, fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(bilinear_hwc)
    axes[0, 1].set_title(
        f"Bilinear Upsampling\nPSNR: {bilinear_psnr:.2f} dB", fontsize=14, fontweight="bold"
    )
    axes[0, 1].axis("off")

    axes[1, 0].imshow(pred_hwc)
    axes[1, 0].set_title(
        f"Model Output\nPSNR: {model_psnr:.2f} dB", fontsize=14, fontweight="bold"
    )
    axes[1, 0].axis("off")

    axes[1, 1].imshow(gt_hwc)
    axes[1, 1].set_title("Ground Truth HR", fontsize=14, fontweight="bold")
    axes[1, 1].axis("off")

    plt.tight_layout(pad=2.0)
    sample_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(sample_dir / filename, bbox_inches="tight", pad_inches=0.1, dpi=300)
    plt.close()


def _needs_full_comparison(pred_full_bchw: torch.Tensor, pred_crop_bchw: torch.Tensor) -> bool:
    return tuple(pred_full_bchw.shape[-2:]) != tuple(pred_crop_bchw.shape[-2:])


def _save_full_comparison_if_needed(
    sample_dir: Path,
    *,
    pred_full_bchw: torch.Tensor,
    gt_full_bchw: torch.Tensor,
    bilinear_full_bchw: torch.Tensor,
    lr_full_hwc: np.ndarray,
    lr_bilinear_full_hwc: np.ndarray,
    pred_crop_bchw: torch.Tensor,
    lr_h_full: int,
    lr_w_full: int,
) -> None:
    if not _needs_full_comparison(pred_full_bchw, pred_crop_bchw):
        return
    full_model_psnr = _psnr_bchw(pred_full_bchw, gt_full_bchw)
    full_bilinear_psnr = _psnr_bchw(bilinear_full_bchw, gt_full_bchw)
    full_h, full_w = int(gt_full_bchw.shape[-2]), int(gt_full_bchw.shape[-1])
    _save_comparison_quad(
        sample_dir,
        "comparison_full.png",
        lr_full_hwc,
        lr_bilinear_full_hwc,
        _bchw_to_hwc_np(pred_full_bchw),
        _bchw_to_hwc_np(gt_full_bchw),
        model_psnr=full_model_psnr,
        bilinear_psnr=full_bilinear_psnr,
        banner=f"Full patch (LR {lr_h_full}×{lr_w_full} → HR {full_h}×{full_w})",
    )


def get_eval_autocast_dtype(eval_mixed_precision, device):
    """Return autocast dtype for evaluation, or None for float32. Only applies on CUDA."""
    if device.type != "cuda" or not eval_mixed_precision or eval_mixed_precision == "none":
        return None
    if eval_mixed_precision == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if eval_mixed_precision == "bfloat16":
        return torch.bfloat16
    if eval_mixed_precision == "fp16":
        return torch.float16
    return None


def _projection_encoded_dim(input_projection, args):
    if input_projection is None:
        return 2
    return int(
        getattr(
            input_projection,
            "projection_output_dim",
            getattr(input_projection, "output_dim", args.projection_dim),
        )
    )


def build_input_projection_decoder_bundle(args, device, *, output_dim=3):
    if args.model not in {"mlp", "mlp_tcnn", "nir", "hash_attn", "fourier_band_attn"}:
        raise ValueError(
            f"Unknown --model {args.model!r}. "
            "Use mlp, mlp_tcnn, nir, hash_attn, or fourier_band_attn."
        )

    canon, fs = normalize_input_projection_name(args.input_projection, args.fourier_scale)
    args.input_projection = canon
    args.fourier_scale = fs

    if args.model == "hash_attn" and canon != "hashgrid_tcnn":
        raise ValueError("--model hash_attn requires --input_projection hashgrid/hashgrid_tcnn")
    if args.model == "fourier_band_attn" and canon != "fourier":
        raise ValueError("--model fourier_band_attn requires --input_projection fourier(_N)")

    hash_base_res, hash_max_res = resolve_hash_grid_resolutions(args, canon=canon)
    args.hash_base_resolution = hash_base_res
    args.hash_max_resolution = hash_max_res
    if canon == "hashgrid_tcnn" and (
        int(getattr(args, "_hash_max_resolution_cli", 0) or 0) <= 0
        or int(getattr(args, "_hash_base_resolution_cli", 0) or 0) <= 0
    ):
        print(
            f"HashGrid resolutions for {args.sample_id!r}: "
            f"base={hash_base_res}, max={hash_max_res} (from LR patch size when CLI is 0)",
            flush=True,
        )

    input_projection = get_input_projection(
        canon,
        2,
        args.projection_dim,
        device,
        args.fourier_scale,
        hash_n_levels=args.hash_n_levels,
        hash_n_features_per_level=args.hash_n_features_per_level,
        hash_log2_hashmap_size=args.hash_log2_hashmap_size,
        hash_base_resolution=hash_base_res,
        hash_max_resolution=hash_max_res,
        hash_encoding_output_dtype=args.hash_encoding_dtype,
        hash_encoding_preset=args.hash_encoding_preset,
        hash_grid_type=args.hash_grid_type,
    )

    d_enc = _projection_encoded_dim(input_projection, args)

    decoder_in = 2 if canon == "none" else d_enc
    decoder = get_decoder(
        args.model,
        args.network_depth,
        decoder_in,
        args.network_hidden_dim,
        output_dim=output_dim,
        tcnn_mlp_dtype=args.tcnn_mlp_dtype,
        mlp_init=args.mlp_init,
        device=device,
        hash_n_levels=args.hash_n_levels,
        hash_n_features_per_level=args.hash_n_features_per_level,
        hash_attn_token_dim=args.hash_attn_token_dim,
        attn_token_dim=int(getattr(args, "attn_token_dim", 32)),
        fourier_num_bands=int(getattr(args, "fourier_num_bands", 8)),
    )
    return input_projection, decoder


def build_inr_model(input_projection, decoder, args, device):
    return get_inr(
        input_projection,
        decoder,
        args.num_samples,
        use_gnll=args.use_gnll,
        use_base_frame=not args.no_base_frame,
        use_direct_param_T=not args.no_direct_param_T,
        use_color_shift=args.use_color_shift,
        use_separate_ud=args.use_separate_ud,
    ).to(device)


def train_one_iteration(
    model,
    optimizer,
    train_sample,
    device,
    args,
    iteration,
):
    model.train()

    use_gnll_loss = model.use_gnll
    if use_gnll_loss:
        recon_criterion = nn.GaussianNLLLoss()
    else:
        recon_criterion = BasicLosses.mse_loss

    input = train_sample["input"].to(device)
    lr_target = train_sample["lr_target"].to(device)
    sample_id = train_sample["sample_id"].to(device)
    if "shifts" in train_sample and "dx_percent" in train_sample["shifts"]:
        gt_dx = train_sample["shifts"]["dx_percent"].to(device)
        gt_dy = train_sample["shifts"]["dy_percent"].to(device)
    else:
        gt_dx = torch.zeros(lr_target.shape[0], device=device)
        gt_dy = torch.zeros(lr_target.shape[0], device=device)

    optimizer.zero_grad()

    progress = {"iteration": int(iteration)}

    hvf = int(getattr(args, "heldout_val_frames", 0) or 0)
    sid0 = int(sample_id.reshape(-1)[0].item())
    heldout = hvf > 0 and sid0 >= (int(args.num_samples) - hvf)

    # INR renders HR; LR supervision applies degradation inside forward when lr_frames is set.
    if use_gnll_loss:
        output, pred_shifts, pred_variance = model(
            input,
            sample_id,
            scale_factor=1,
            lr_frames=lr_target,
            progress=progress,
            lr_align_args=args,
        )
        recon_loss = recon_criterion(output, lr_target, pred_variance)
    else:
        output, pred_shifts = model(
            input, sample_id, lr_frames=lr_target, progress=progress, lr_align_args=args
        )
        if heldout:
            recon_loss = (output * 0.0).mean()
        else:
            recon_loss = recon_criterion(output, lr_target)

    pred_dx, pred_dy = pred_shifts
    lr_h, lr_w = lr_target.shape[1:3]
    pred_dx_percent = pred_dx / lr_w
    pred_dy_percent = pred_dy / lr_h
    trans_loss = torch.mean(
        torch.sqrt((pred_dx_percent - gt_dx) ** 2 + (pred_dy_percent - gt_dy) ** 2)
    )

    total_loss = recon_loss
    total_loss.backward()
    optimizer.step()

    return {
        "recon_loss": recon_loss.item(),
        "trans_loss": trans_loss.item(),
        "total_loss": total_loss.item(),
    }


def test_one_epoch(
    model,
    test_loader,
    device,
    eval_autocast_dtype=None,
    *,
    eval_crop_lr_size: int = 0,
    df: int = 4,
    crop_anchor: str = "topleft",
):
    model.eval()

    with torch.no_grad():
        hr_coords = test_loader.get_hr_coordinates().unsqueeze(0).to(device)
        hr_image = test_loader.get_original_hr().unsqueeze(0).to(device)
        sample_id = torch.tensor([0]).to(device)

        if eval_autocast_dtype is not None:
            with torch.autocast(device_type="cuda", dtype=eval_autocast_dtype):
                if model.use_gnll:
                    output, _ = model(hr_coords, sample_id, scale_factor=1, training=False)
                else:
                    output, _ = model(hr_coords, sample_id, training=False)
        else:
            if model.use_gnll:
                output, _ = model(hr_coords, sample_id, scale_factor=1, training=False)
            else:
                output, _ = model(hr_coords, sample_id, training=False)

        # Unstandardize the output
        output = output * test_loader.get_lr_std(0).to(device) + test_loader.get_lr_mean(0).to(
            device
        )

        if int(eval_crop_lr_size) > 0:
            crop_info = apply_eval_center_crop(
                pred_bchw=output.permute(0, 3, 1, 2),
                bilinear_bchw=output.permute(0, 3, 1, 2),
                gt_bchw=hr_image.permute(0, 3, 1, 2),
                lr_hwc=None,
                crop_lr_size=int(eval_crop_lr_size),
                df=int(df),
                crop_anchor=str(crop_anchor),
            )
            output = crop_info["pred_bchw"].permute(0, 2, 3, 1)
            hr_image = crop_info["gt_bchw"].permute(0, 2, 3, 1)

        loss = F.mse_loss(output, hr_image)

        # Calculate PSNR
        psnr = -10 * torch.log10(loss)

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return loss.item(), psnr.item()


def optimize_and_evaluate_sample(model, train_data, device, sample_idx, args, output_dir):
    print(f"\n{'='*60}")
    print(f"Optimizing sample {sample_idx + 1}")
    print(f"{'='*60}")

    # Record start time for timing metrics
    start_time = time.time()

    # Setup optimizer for this sample
    optimizer = build_optimizer(model.parameters(), args)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.iters, eta_min=1e-6)

    # Training loop for this sample
    iteration = 0
    progress_bar = tqdm(total=args.iters, desc=f"Training Sample {sample_idx + 1}")

    # Lists to store training metrics
    psnr_list = []
    recon_loss_list = []
    trans_loss_list = []
    total_loss_list = []
    iteration_list = []

    # Track timing for different phases
    training_start_time = time.time()

    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)

    while iteration < args.iters:
        for train_sample in train_dataloader:
            if iteration >= args.iters:
                break

            train_losses = train_one_iteration(
                model,
                optimizer,
                train_sample,
                device,
                args,
                iteration,
            )
            scheduler.step()
            iteration += 1

            progress_bar.update(1)
            postfix_dict = {
                "recon": f"{train_losses['recon_loss']:.4f}",
                "trans": f"{train_losses['trans_loss']:.4f}",
            }
            progress_bar.set_postfix(postfix_dict)

            # Periodic evaluation
            if iteration % 100 == 0:
                eval_autocast_dtype = get_eval_autocast_dtype(args.eval_mixed_precision, device)
                test_loss, test_psnr = test_one_epoch(
                    model,
                    train_data,
                    device,
                    eval_autocast_dtype,
                    eval_crop_lr_size=int(getattr(args, "eval_crop_lr_size", 0) or 0),
                    df=int(args.df),
                    crop_anchor=str(getattr(args, "eval_crop_anchor", "topleft")),
                )
                print(
                    f"\nIter {iteration}: Train Loss: {train_losses['total_loss']:.6f}, "
                    f"Test Loss: {test_loss:.6f}, Test PSNR: {test_psnr:.2f} dB"
                )

                # Store training metrics
                iteration_list.append(iteration)
                psnr_list.append(test_psnr)
                recon_loss_list.append(train_losses["recon_loss"])
                trans_loss_list.append(train_losses["trans_loss"])
                total_loss_list.append(train_losses["total_loss"])

    progress_bar.close()

    # Record training end time
    training_end_time = time.time()
    training_time = training_end_time - training_start_time

    # Final evaluation with alignment and color matching
    evaluation_start_time = time.time()
    model.eval()
    eval_autocast_dtype = get_eval_autocast_dtype(args.eval_mixed_precision, device)
    with torch.no_grad():
        hr_coords = train_data.get_hr_coordinates().unsqueeze(0).to(device)
        hr_image = train_data.get_original_hr().unsqueeze(0).to(device)
        sample_id = torch.tensor([0]).to(device)

        if eval_autocast_dtype is not None:
            with torch.autocast(device_type="cuda", dtype=eval_autocast_dtype):
                if model.use_gnll:
                    output, _ = model(hr_coords, sample_id, scale_factor=1, training=False)
                else:
                    output, _ = model(hr_coords, sample_id, scale_factor=1, training=False)
        else:
            if model.use_gnll:
                output, _ = model(hr_coords, sample_id, scale_factor=1, training=False)
            else:
                output, _ = model(hr_coords, sample_id, scale_factor=1, training=False)

        # Unstandardize the output
        output = output * train_data.get_lr_std(0).to(device) + train_data.get_lr_mean(0).to(device)

        final_test_loss = F.mse_loss(output, hr_image).item()
        final_psnr = -10 * torch.log10(torch.tensor(final_test_loss)).item()

        pred_tensor = _hwc_to_bchw(output)
        gt_tensor = _hwc_to_bchw(hr_image)

        # Get LR for bilinear comparison – always work in HWC
        if hasattr(train_data, "get_lr_sample_hwc"):
            lr_standardized_hwc = (
                train_data.get_lr_sample_hwc(0).cpu().numpy()
            )  # H, W, 3 (standardized)
            lr_needs_unstandardize = True
        else:
            lr_any = train_data.get_lr_sample(0).cpu().numpy()  # might be CHW or HWC or multi-frame
            if lr_any.ndim == 3 and lr_any.shape[0] == 3:  # CHW -> HWC
                lr_standardized_hwc = np.transpose(lr_any, (1, 2, 0))
            elif lr_any.ndim == 3 and lr_any.shape[2] > 3:  # H, W, (3*T)
                H, W, C = lr_any.shape
                if C % 3 == 0:
                    T = C // 3
                    lr_standardized_hwc = lr_any.reshape(H, W, T, 3)[:, :, 0, :]
                else:
                    lr_standardized_hwc = lr_any[:, :, :3]
            else:
                lr_standardized_hwc = lr_any  # assume HWC
            # SRData.get_lr_sample returns unstandardized already → do NOT unstandardize again
            lr_needs_unstandardize = False

        # Unstandardize only if the LR we fetched is standardized (e.g., WorldStratTestDataset)
        if lr_needs_unstandardize:
            lr_std = train_data.get_lr_std(0).cpu().numpy()
            lr_mean = train_data.get_lr_mean(0).cpu().numpy()
            if lr_std.ndim == 1:
                lr_std = lr_std.reshape(1, 1, -1)
            if lr_mean.ndim == 1:
                lr_mean = lr_mean.reshape(1, 1, -1)
            lr_original = lr_standardized_hwc * lr_std + lr_mean  # H, W, 3
        else:
            lr_original = lr_standardized_hwc

        lr_h, lr_w = lr_original.shape[:2]
        hr_h, hr_w = hr_image.shape[1], hr_image.shape[2]

        # Resize LR (still HWC) then convert to BCHW for metrics
        lr_bilinear = cv2.resize(lr_original, (hr_w, hr_h), interpolation=cv2.INTER_LINEAR)
        bilinear_tensor = torch.from_numpy(lr_bilinear).unsqueeze(0).permute(0, 3, 1, 2).to(device)

        # Align outputs for fair comparison
        # Alignment disabled to avoid OOM errors - can be re-enabled if needed
        print("Skipping alignment (disabled to avoid memory issues)")
        pred_aligned = pred_tensor
        bilinear_aligned = bilinear_tensor

        # Spatial alignment disabled (causes OOM on small GPUs).

        lr_bilinear_hwc = np.clip(lr_bilinear, 0.0, 1.0)
        pred_full_bchw = pred_aligned
        gt_full_bchw = gt_tensor
        bilinear_full_bchw = bilinear_aligned
        lr_full_hwc = np.clip(lr_original, 0.0, 1.0)

        # Optional center-crop of SR/GT/bilinear/LR before metrics+viz so different LR sizes
        # (e.g. 64/128/256/512) are evaluated on the same physical area on the ground.
        eval_crop_info = apply_eval_center_crop(
            pred_bchw=pred_aligned,
            bilinear_bchw=bilinear_aligned,
            gt_bchw=gt_tensor,
            lr_hwc=lr_original,
            lr_bilinear_hwc=lr_bilinear_hwc,
            crop_lr_size=int(getattr(args, "eval_crop_lr_size", 0) or 0),
            df=int(args.df),
            crop_anchor=str(getattr(args, "eval_crop_anchor", "topleft")),
        )
        pred_aligned = eval_crop_info["pred_bchw"]
        bilinear_aligned = eval_crop_info["bilinear_bchw"]
        gt_tensor = eval_crop_info["gt_bchw"]
        lr_original = eval_crop_info["lr_hwc"]
        if eval_crop_info.get("lr_bilinear_hwc") is not None:
            lr_bilinear_hwc = eval_crop_info["lr_bilinear_hwc"]
        if eval_crop_info["cropped"]:
            print(
                f"Eval crop ({eval_crop_info.get('crop_anchor', 'topleft')}): LR -> "
                f"{eval_crop_info['lr_crop']}px, HR/SR/GT -> {eval_crop_info['hr_crop']}px "
                f"(--eval_crop_lr_size={int(args.eval_crop_lr_size)} * df={int(args.df)})."
            )
            final_test_loss = F.mse_loss(pred_aligned, gt_tensor).item()
            final_psnr = -10 * torch.log10(torch.tensor(final_test_loss)).item()

        # Calculate comprehensive metrics using aligned tensors
        pred_cpu = pred_aligned.detach().cpu()
        gt_cpu = gt_tensor.detach().cpu()
        bilinear_cpu = bilinear_aligned.detach().cpu()

        model_psnr = peak_signal_noise_ratio(pred_cpu, gt_cpu, data_range=1.0).item()
        bilinear_psnr = peak_signal_noise_ratio(bilinear_cpu, gt_cpu, data_range=1.0).item()

        model_ssim = ssim(pred_cpu, gt_cpu, data_range=1.0).item()
        bilinear_ssim = ssim(bilinear_cpu, gt_cpu, data_range=1.0).item()

        lpips_fn = lpips.LPIPS(net="vgg").to(device)
        model_lpips = lpips_fn((pred_aligned * 2 - 1), (gt_tensor * 2 - 1)).item()
        bilinear_lpips = lpips_fn((bilinear_aligned * 2 - 1), (gt_tensor * 2 - 1)).item()

        # Calculate additional metrics
        # MSE (Mean Squared Error)
        model_mse = F.mse_loss(pred_aligned, gt_tensor).item()
        bilinear_mse = F.mse_loss(bilinear_aligned, gt_tensor).item()

        # MAE (Mean Absolute Error)
        model_mae = F.l1_loss(pred_aligned, gt_tensor).item()
        bilinear_mae = F.l1_loss(bilinear_aligned, gt_tensor).item()

        # Convert aligned tensors back to numpy for visualization (driven by cropped tensors so
        # comparison.png et al. match the metrics).
        pred_aligned_np = pred_aligned.squeeze(0).permute(1, 2, 0).cpu().numpy()
        bilinear_aligned_np = bilinear_aligned.squeeze(0).permute(1, 2, 0).cpu().numpy()
        gt_np = gt_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

        # Ensure images are in valid range
        pred_aligned_np = np.clip(pred_aligned_np, 0, 1)
        bilinear_aligned_np = np.clip(bilinear_aligned_np, 0, 1)
        gt_np = np.clip(gt_np, 0, 1)
        lr_original = np.clip(lr_original, 0, 1)

        # Convert lr_original from CHW to HWC for visualization
        if lr_original.ndim == 3 and lr_original.shape[0] == 3:
            lr_original = np.transpose(lr_original, (1, 2, 0))  # Convert from CHW to HWC

        # Save individual sample visualization
        sample_dir = output_dir / f"sample_{sample_idx:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        crop_banner = ""
        if eval_crop_info["cropped"]:
            lr_c, hr_c = eval_crop_info["lr_crop"], eval_crop_info["hr_crop"]
            crop_banner = (
                f"Eval crop ({eval_crop_info.get('crop_anchor', 'topleft')}): "
                f"LR {lr_c}×{lr_c} → HR {hr_c}×{hr_c}"
            )

        _save_comparison_quad(
            sample_dir,
            "comparison.png",
            lr_original,
            bilinear_aligned_np,
            pred_aligned_np,
            gt_np,
            model_psnr=model_psnr,
            bilinear_psnr=bilinear_psnr,
            banner=crop_banner,
        )

        _save_full_comparison_if_needed(
            sample_dir,
            pred_full_bchw=pred_full_bchw,
            gt_full_bchw=gt_full_bchw,
            bilinear_full_bchw=bilinear_full_bchw,
            lr_full_hwc=lr_full_hwc,
            lr_bilinear_full_hwc=_bchw_to_hwc_np(bilinear_full_bchw),
            pred_crop_bchw=pred_aligned,
            lr_h_full=lr_h,
            lr_w_full=lr_w,
        )

        # Save individual images
        plt.figure(figsize=(8, 8))
        plt.imshow(pred_aligned_np)
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(
            sample_dir / "prediction_aligned.png", bbox_inches="tight", pad_inches=0, dpi=300
        )
        plt.close()

        plt.figure(figsize=(8, 8))
        plt.imshow(gt_np)
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(sample_dir / "ground_truth.png", bbox_inches="tight", pad_inches=0, dpi=300)
        plt.close()

        # Plot training curves if we have data
        if len(psnr_list) > 0:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

            ax1.plot(iteration_list, psnr_list, color="blue", linewidth=2, label="PSNR (Test)")
            ax1.set_xlabel("Iteration", fontsize=12)
            ax1.set_ylabel("PSNR (dB)", fontsize=12)
            ax1.set_title(
                f"Sample {sample_idx + 1} - Training PSNR Evolution", fontsize=14, fontweight="bold"
            )
            ax1.grid(True, alpha=0.3)
            ax1.legend()

            ax2.plot(
                iteration_list,
                recon_loss_list,
                color="red",
                linewidth=2,
                label="Reconstruction Loss",
            )
            ax2.plot(
                iteration_list,
                trans_loss_list,
                color="green",
                linewidth=2,
                label="Transformation Loss",
            )
            ax2.plot(
                iteration_list, total_loss_list, color="purple", linewidth=2, label="Total Loss"
            )
            ax2.set_xlabel("Iteration", fontsize=12)
            ax2.set_ylabel("Loss", fontsize=12)
            ax2.set_title(
                f"Sample {sample_idx + 1} - Training Loss Evolution", fontsize=14, fontweight="bold"
            )
            ax2.grid(True, alpha=0.3)
            ax2.legend()

            plt.tight_layout()
            plt.savefig(
                sample_dir / "training_metrics.png", bbox_inches="tight", pad_inches=0.1, dpi=300
            )
            plt.close()

    # Generate variance visualizations if using GNLL (unless disabled)
    if model.use_gnll and not args.no_variance_viz:
        print(f"\nGenerating variance visualizations for sample {sample_idx + 1}...")
        visualize_lr_variance(model, train_data, device, sample_dir, sample_idx)

    # Record evaluation end time
    evaluation_end_time = time.time()
    evaluation_time = evaluation_end_time - evaluation_start_time
    total_time = evaluation_end_time - start_time

    # Return comprehensive results for this sample
    return {
        "sample_idx": sample_idx,
        "sample_info": {
            "dataset": args.dataset,
            "sample_id": getattr(args, "sample_id", f"sample_{sample_idx}"),
            "num_lr_frames": len(train_data),
            "iterations": args.iters,
            "model_type": args.model,
            "input_projection": args.input_projection,
            "network_depth": args.network_depth,
            "network_hidden_dim": args.network_hidden_dim,
        },
        "image_metrics": {
            "model_psnr": model_psnr,
            "bilinear_psnr": bilinear_psnr,
            "psnr_improvement": model_psnr - bilinear_psnr,
            "model_ssim": model_ssim,
            "bilinear_ssim": bilinear_ssim,
            "ssim_improvement": model_ssim - bilinear_ssim,
            "model_lpips": model_lpips,
            "bilinear_lpips": bilinear_lpips,
            "lpips_improvement": bilinear_lpips - model_lpips,
            "model_mse": model_mse,
            "bilinear_mse": bilinear_mse,
            "mse_improvement": bilinear_mse - model_mse,
            "model_mae": model_mae,
            "bilinear_mae": bilinear_mae,
            "mae_improvement": bilinear_mae - model_mae,
        },
        "training_metrics": {
            "final_test_loss": final_test_loss,
            "final_test_psnr": final_psnr,
            "iterations": iteration_list,
            "psnr": psnr_list,
            "recon_loss": recon_loss_list,
            "trans_loss": trans_loss_list,
            "total_loss": total_loss_list,
            "convergence_iteration": len(psnr_list),  # Number of evaluation points
            "final_recon_loss": recon_loss_list[-1] if recon_loss_list else None,
            "final_trans_loss": trans_loss_list[-1] if trans_loss_list else None,
            "final_total_loss": total_loss_list[-1] if total_loss_list else None,
        },
        "timing_metrics": {
            "training_time_seconds": training_time,
            "training_time_minutes": training_time / 60.0,
            "evaluation_time_seconds": evaluation_time,
            "evaluation_time_minutes": evaluation_time / 60.0,
            "total_time_seconds": total_time,
            "total_time_minutes": total_time / 60.0,
            "time_per_iteration_seconds": training_time / args.iters if args.iters > 0 else 0,
        },
        "image_dimensions": {
            "hr_height": hr_image.shape[1],
            "hr_width": hr_image.shape[2],
            "lr_height": lr_original.shape[0],
            "lr_width": lr_original.shape[1],
            "scale_factor": hr_image.shape[1] / lr_original.shape[0],
        },
    }


def visualize_lr_variance(model, train_data, device, output_dir, sample_id):
    """
    Visualize variance maps for each LR sample when using GNLL.

    Args:
        model: Trained model with GNLL enabled
        train_data: Training dataset
        device: Device to run on
        output_dir: Directory to save visualizations
        sample_id: Sample ID being processed
    """
    use_gnll_loss = model.use_gnll
    if not use_gnll_loss:
        print("Warning: visualize_lr_variance called but model does not use GNLL")
        return

    model.eval()
    with torch.no_grad():
        # Get HR coordinates for inference
        hr_coords = train_data.get_hr_coordinates().unsqueeze(0).to(device)
        hr_image = train_data.get_original_hr().unsqueeze(0).to(device)

        # Create output directory for variance visualizations
        variance_dir = output_dir / "variance_visualizations"
        variance_dir.mkdir(exist_ok=True)

        # Get number of LR samples based on dataset type
        if hasattr(train_data, "num_samples"):
            num_samples = train_data.num_samples
        elif hasattr(train_data, "lr_paths"):
            num_samples = len(train_data.lr_paths)
        else:
            print(
                "Warning: Cannot determine number of LR samples. Skipping variance visualization."
            )
            return

        print(f"Creating variance visualizations for {num_samples} LR samples...")

        # Collect data for 2x8 grid visualization
        lr_samples_for_grid = []
        variance_maps_for_grid = []
        global_vmin = None
        global_vmax = None

        # Process each LR sample individually
        for i in range(num_samples):
            sample_id_tensor = torch.tensor([i]).to(device)

            # Get the model output with variance for this specific sample
            # Pass an HR-sized frame so GNLL variance head can run at test-time
            output, _, variance = model(
                hr_coords, sample_id_tensor, scale_factor=1, training=False, lr_frames=hr_image
            )

            # Ensure variance is a tensor
            if isinstance(variance, list):
                try:
                    variance = torch.stack(variance, dim=0)
                except Exception:
                    variance = None
            if variance is None:
                variance = torch.full_like(output, 1e-6)

            # Best-effort unstandardization and variance scaling with dataset stats
            std_i = None
            mean_i = None

            try:
                std_i = train_data.get_lr_std(i)
                mean_i = train_data.get_lr_mean(i)
            except (TypeError, IndexError):
                # Some datasets (e.g., worldstrat_test) may not index per-sample; fall back to 0
                try:
                    std_i = train_data.get_lr_std(0)
                    mean_i = train_data.get_lr_mean(0)
                except (TypeError, IndexError, AttributeError):
                    pass
            except AttributeError:
                pass

            lr_np = None
            if std_i is not None and mean_i is not None:
                # Convert to numpy first, then to tensor to avoid indexing issues
                if hasattr(std_i, "cpu"):
                    std_i = std_i.cpu().numpy()
                if hasattr(mean_i, "cpu"):
                    mean_i = mean_i.cpu().numpy()

                # Convert to tensor
                std_i = torch.tensor(std_i, device=device, dtype=torch.float32)
                mean_i = torch.tensor(mean_i, device=device, dtype=torch.float32)

                # Ensure shapes broadcast: [1,1,C]
                if std_i.ndim == 1:
                    std_i = std_i.view(1, 1, -1)
                    mean_i = mean_i.view(1, 1, -1)
                output = output * std_i + mean_i
                # Variance scales by std^2
                # variance = variance * (std_i ** 2)

                # Try to fetch and unstandardize the LR sample for display
                try:
                    if hasattr(train_data, "get_lr_sample_hwc"):
                        lr_sample = train_data.get_lr_sample_hwc(i)
                        if hasattr(lr_sample, "cpu"):
                            lr_np = lr_sample.cpu().numpy()
                        else:
                            lr_np = np.array(lr_sample)
                    elif hasattr(train_data, "get_lr_sample"):
                        lr_sample = train_data.get_lr_sample(i)
                        if hasattr(lr_sample, "cpu"):
                            lr_np = lr_sample.permute(1, 2, 0).cpu().numpy()
                        else:
                            lr_np = np.array(lr_sample).transpose(1, 2, 0)
                    # Unstandardize LR
                    if lr_np is not None:
                        std_np = std_i.squeeze(0).squeeze(0).detach().cpu().numpy()
                        mean_np = mean_i.squeeze(0).squeeze(0).detach().cpu().numpy()
                        lr_np = lr_np * std_np + mean_np
                except Exception:
                    lr_np = None

            # Convert to numpy
            output_np = output.squeeze().cpu().numpy()
            variance_np = variance.squeeze().cpu().numpy()
            hr_np = hr_image.squeeze().cpu().numpy()

            # Clip values to valid range
            output_np = np.clip(output_np, 0, 1)
            hr_np = np.clip(hr_np, 0, 1)
            # Variance should be non-negative
            if variance_np.min() < 0:
                variance_np = np.maximum(variance_np, 0)

            # Convert variance to standard deviation (std = sqrt(variance))
            std_np = np.sqrt(variance_np)

            # Create visualization - 2x2 grid (removed absolute error and high variance regions)
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))

            # Row 1: Original images
            axes[0, 0].imshow(hr_np)
            axes[0, 0].set_title(f"Ground Truth HR", fontsize=12, fontweight="bold")
            axes[0, 0].axis("off")

            axes[0, 1].imshow(output_np)
            axes[0, 1].set_title(f"Model Output (Sample {i})", fontsize=12, fontweight="bold")
            axes[0, 1].axis("off")

            # Row 2: Standard deviation analysis
            # Raw std map - upsample to match output size if needed
            std_display = std_np.copy()
            if std_np.shape[:2] != output_np.shape[:2]:
                # Std is at different resolution, upsample to match output
                if std_np.ndim == 3:
                    # Resize each channel
                    std_display = np.zeros(
                        (output_np.shape[0], output_np.shape[1], std_np.shape[2])
                    )
                    for c in range(std_np.shape[2]):
                        std_display[:, :, c] = cv2.resize(
                            std_np[:, :, c],
                            (output_np.shape[1], output_np.shape[0]),
                            interpolation=cv2.INTER_LINEAR,
                        )
                else:
                    std_display = cv2.resize(
                        std_np,
                        (output_np.shape[1], output_np.shape[0]),
                        interpolation=cv2.INTER_LINEAR,
                    )

            # Build a 2D std map (H x W) for display
            if std_display.ndim == 3:
                std_map = std_display.mean(axis=-1)
            else:
                std_map = std_display

            # Calculate color scale centered around 1 (neutral)
            # Find maximum deviation from 1
            max_deviation = max(abs(std_map.max() - 1), abs(std_map.min() - 1))

            # Set symmetric range around 1, but ensure vmin >= 0 (std is sqrt(variance) which is always >= 0)
            vmin = max(0, 1 - max_deviation)
            vmax = 1 + max_deviation

            # Ensure we have a reasonable range (at least some small deviation)
            if max_deviation < 1e-6:
                # If all values are very close to 1, use a small symmetric range
                vmin = max(0, 0.99)  # Ensure >= 0
                vmax = 1.01

            # Track global std range for consistent color scale across all samples
            if global_vmin is None:
                global_vmin = vmin
                global_vmax = vmax
            else:
                # Update global range to include this sample's range
                global_max_deviation = max(
                    abs(global_vmax - 1), abs(global_vmin - 1), max_deviation
                )
                global_vmin = 1 - global_max_deviation
                global_vmax = 1 + global_max_deviation

            # Store std map and LR sample for grid visualization
            variance_maps_for_grid.append(std_map.copy())
            if lr_np is not None:
                # Prepare LR sample for grid (resize to HR size)
                if lr_np.ndim == 2:
                    lr_np_grid = np.repeat(lr_np[..., None], 3, axis=-1)
                elif lr_np.shape[-1] == 1:
                    lr_np_grid = np.repeat(lr_np, 3, axis=-1)
                else:
                    lr_np_grid = lr_np.copy()
                lr_vis_grid = cv2.resize(
                    lr_np_grid,
                    (output_np.shape[1], output_np.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
                # Brighten LR image for better visibility (scale + shift)
                lr_vis_grid = lr_vis_grid * 1.2 + 0.15
                lr_vis_grid = np.clip(lr_vis_grid, 0.0, 1.0)
                lr_samples_for_grid.append(lr_vis_grid)
            else:
                lr_samples_for_grid.append(None)

            # Display std map with Blues colormap
            im_var = axes[1, 0].imshow(std_map, cmap="Blues", vmin=vmin, vmax=vmax)
            axes[1, 0].set_title(
                f"Standard Deviation Map (Sample {i})", fontsize=12, fontweight="bold"
            )
            axes[1, 0].axis("off")
            cbar = plt.colorbar(im_var, ax=axes[1, 0], fraction=0.046, pad=0.04)
            cbar.set_label("Standard Deviation", rotation=270, labelpad=15)

            # Show the LR sample alongside
            if lr_np is not None:
                # Resize LR to HR size for visualization
                if lr_np.ndim == 2:
                    lr_np = np.repeat(lr_np[..., None], 3, axis=-1)
                elif lr_np.shape[-1] == 1:
                    lr_np = np.repeat(lr_np, 3, axis=-1)
                lr_vis = cv2.resize(
                    lr_np, (output_np.shape[1], output_np.shape[0]), interpolation=cv2.INTER_LINEAR
                )
                # Brighten LR image for better visibility (scale + shift)
                lr_vis = lr_vis * 1.2 + 0.15
                lr_vis = np.clip(lr_vis, 0.0, 1.0)
                axes[1, 1].imshow(lr_vis)
                axes[1, 1].set_title(f"LR Sample (Sample {i})", fontsize=12, fontweight="bold")
                axes[1, 1].axis("off")
            else:
                # Fallback: show std stats if LR not available
                axes[1, 1].text(
                    0.1,
                    0.8,
                    f"Standard Deviation Statistics:",
                    fontsize=12,
                    fontweight="bold",
                    transform=axes[1, 1].transAxes,
                )
                axes[1, 1].text(
                    0.1,
                    0.7,
                    f"Mean: {np.mean(std_np):.6f}",
                    fontsize=10,
                    transform=axes[1, 1].transAxes,
                )
                axes[1, 1].text(
                    0.1,
                    0.6,
                    f"Std: {np.std(std_np):.6f}",
                    fontsize=10,
                    transform=axes[1, 1].transAxes,
                )
                axes[1, 1].text(
                    0.1,
                    0.5,
                    f"Min: {np.min(std_np):.6f}",
                    fontsize=10,
                    transform=axes[1, 1].transAxes,
                )
                axes[1, 1].text(
                    0.1,
                    0.4,
                    f"Max: {np.max(std_np):.6f}",
                    fontsize=10,
                    transform=axes[1, 1].transAxes,
                )
                axes[1, 1].text(
                    0.1,
                    0.3,
                    f"75th percentile: {np.percentile(std_np, 75):.6f}",
                    fontsize=10,
                    transform=axes[1, 1].transAxes,
                )
                axes[1, 1].text(
                    0.1,
                    0.2,
                    f"95th percentile: {np.percentile(std_np, 95):.6f}",
                    fontsize=10,
                    transform=axes[1, 1].transAxes,
                )
                axes[1, 1].set_xlim(0, 1)
                axes[1, 1].set_ylim(0, 1)
                axes[1, 1].axis("off")

            plt.tight_layout(pad=2.0)

            # Save individual std visualization
            variance_path = variance_dir / f"sample_{i:03d}_variance_analysis.png"
            plt.savefig(variance_path, bbox_inches="tight", pad_inches=0.1, dpi=300)
            plt.close()

            # Save individual std map as an image (for later 2x8 grid visualization)
            fig_var_only = plt.figure(figsize=(8, 8))
            ax_var_only = fig_var_only.add_subplot(111)
            im_var_only = ax_var_only.imshow(std_map, cmap="Blues", vmin=vmin, vmax=vmax)
            ax_var_only.axis("off")
            cbar_var_only = plt.colorbar(im_var_only, ax=ax_var_only, fraction=0.046, pad=0.04)
            cbar_var_only.set_label("Standard Deviation", rotation=270, labelpad=15)
            plt.tight_layout(pad=0)
            variance_map_path = variance_dir / f"sample_{i:03d}_variance_map.png"
            plt.savefig(variance_map_path, bbox_inches="tight", pad_inches=0, dpi=300)
            plt.close(fig_var_only)

            # Save individual LR sample as an image (for later 2x8 grid visualization)
            if lr_np is not None:
                fig_lr_only = plt.figure(figsize=(8, 8))
                ax_lr_only = fig_lr_only.add_subplot(111)
                # Resize LR to HR size for visualization if needed
                if lr_np.ndim == 2:
                    lr_np_vis = np.repeat(lr_np[..., None], 3, axis=-1)
                elif lr_np.shape[-1] == 1:
                    lr_np_vis = np.repeat(lr_np, 3, axis=-1)
                else:
                    lr_np_vis = lr_np.copy()
                lr_vis_resized = cv2.resize(
                    lr_np_vis,
                    (output_np.shape[1], output_np.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
                # Brighten LR image for better visibility (scale + shift)
                lr_vis_resized = lr_vis_resized * 1.2 + 0.15
                lr_vis_resized = np.clip(lr_vis_resized, 0.0, 1.0)
                ax_lr_only.imshow(lr_vis_resized)
                ax_lr_only.axis("off")
                plt.tight_layout(pad=0)
                lr_sample_path = variance_dir / f"sample_{i:03d}_lr_sample.png"
                plt.savefig(lr_sample_path, bbox_inches="tight", pad_inches=0, dpi=300)
                plt.close(fig_lr_only)

            # Save individual std map as numpy array (also save variance for reference)
            np.save(variance_dir / f"sample_{i:03d}_std.npy", std_np)
            np.save(variance_dir / f"sample_{i:03d}_variance.npy", variance_np)
            np.save(variance_dir / f"sample_{i:03d}_output.npy", output_np)

        # Create a summary visualization showing all variance maps side by side
        create_variance_summary(train_data, variance_dir, device)

        # Create 2x8 grid: top row = LR samples, bottom row = std maps
        if len(lr_samples_for_grid) >= 8 and len(variance_maps_for_grid) >= 8:
            create_lr_variance_grid(
                lr_samples_for_grid[:8],
                variance_maps_for_grid[:8],
                global_vmin,
                global_vmax,
                variance_dir,
            )

        print(f"Standard deviation visualizations saved to {variance_dir}")


def create_lr_variance_grid(lr_samples, variance_maps, vmin, vmax, variance_dir):
    """
    Create a 2x8 grid visualization: top row = LR samples, bottom row = std maps.

    Args:
        lr_samples: List of LR sample images (numpy arrays) or None
        variance_maps: List of std maps (numpy arrays) - note: variable name kept for compatibility
        vmin: Minimum value for std color scale
        vmax: Maximum value for std color scale
        variance_dir: Directory to save the grid
    """
    from matplotlib.patches import Rectangle

    if len(variance_maps) < 8:
        print(f"Warning: Only {len(variance_maps)} samples available, need 8 for grid")
        return

    # Ensure vmin is at least 0 (std is sqrt(variance) which is always >= 0)
    vmin = max(0, vmin)

    fig, axes = plt.subplots(
        2, 8, figsize=(24, 8)
    )  # Increased height from 6 to 8 for less compact y direction

    # Top row: LR samples
    for i in range(8):
        ax = axes[0, i]
        if lr_samples[i] is not None:
            ax.imshow(lr_samples[i])
        else:
            ax.text(0.5, 0.5, f"LR {i}", ha="center", va="center", transform=ax.transAxes)
        # Get image bounds for border
        if lr_samples[i] is not None:
            h, w = lr_samples[i].shape[:2]
            rect = Rectangle(
                (-0.5, -0.5), w, h, fill=False, edgecolor="gray", linewidth=0.5, clip_on=False
            )
            ax.add_patch(rect)
        ax.axis("off")

    # Bottom row: Standard deviation maps
    for i in range(8):
        ax = axes[1, i]
        im = ax.imshow(variance_maps[i], cmap="Blues", vmin=vmin, vmax=vmax)
        # Get image bounds for border
        h, w = variance_maps[i].shape[:2]
        rect = Rectangle(
            (-0.5, -0.5), w, h, fill=False, edgecolor="gray", linewidth=0.5, clip_on=False
        )
        ax.add_patch(rect)
        ax.axis("off")

    # Adjust layout to leave room for colorbar on the right
    # Use tight_layout first to get proper spacing, then adjust for colorbar
    plt.tight_layout(pad=1.0)

    # Get the position of the bottom-right subplot to align colorbar
    # The bottom row is axes[1, 7] (last std map)
    bottom_right_ax = axes[1, 7]
    bbox = bottom_right_ax.get_position()

    # Position colorbar to the right of the last std map
    # [left, bottom, width, height] in figure coordinates
    cbar_width = 0.015
    cbar_left = bbox.x1 + 0.02  # Small gap after the last subplot
    cbar_bottom = bbox.y0  # Align with bottom of bottom row
    cbar_height = bbox.height  # Match height of bottom row subplots

    cbar_ax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Standard Deviation (1 = neutral)", rotation=270, labelpad=20)
    grid_path = variance_dir / "lr_variance_grid_2x8.png"
    plt.savefig(grid_path, bbox_inches="tight", pad_inches=0.1, dpi=300)
    plt.close()
    print(f"Created 2x8 grid visualization: {grid_path}")


def create_variance_summary(train_data, variance_dir, device):
    """
    Create a summary visualization showing all variance maps in a grid.
    """
    # This would require loading all the saved variance maps and creating a grid
    # For now, we'll create a simple summary
    summary_path = variance_dir / "variance_summary.txt"

    # Get number of LR samples based on dataset type
    if hasattr(train_data, "num_samples"):
        num_samples = train_data.num_samples
    elif hasattr(train_data, "lr_paths"):
        num_samples = len(train_data.lr_paths)
    else:
        num_samples = "Unknown"

    with open(summary_path, "w") as f:
        f.write("Standard Deviation Analysis Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Number of LR samples: {num_samples}\n")
        f.write(f"Each sample has been processed individually to show model uncertainty.\n")
        f.write(f"High standard deviation regions indicate where the model is less confident.\n")
        f.write(f"Standard deviation is computed as sqrt(variance) for easier interpretation.\n")
        f.write(f"Check individual sample_XXX_variance_analysis.png files for detailed analysis.\n")

    print(f"Variance summary saved to {summary_path}")


def create_summary_visualization(all_results, output_dir):
    """Create summary visualization showing metrics across all samples."""
    if not all_results:
        return

    # Extract metrics from nested structure
    sample_indices = [r["sample_idx"] for r in all_results]
    model_psnr = [r["image_metrics"]["model_psnr"] for r in all_results]
    bilinear_psnr = [r["image_metrics"]["bilinear_psnr"] for r in all_results]
    psnr_improvement = [r["image_metrics"]["psnr_improvement"] for r in all_results]
    model_ssim = [r["image_metrics"]["model_ssim"] for r in all_results]
    bilinear_ssim = [r["image_metrics"]["bilinear_ssim"] for r in all_results]
    ssim_improvement = [r["image_metrics"]["ssim_improvement"] for r in all_results]
    model_lpips = [r["image_metrics"]["model_lpips"] for r in all_results]
    bilinear_lpips = [r["image_metrics"]["bilinear_lpips"] for r in all_results]
    lpips_improvement = [r["image_metrics"]["lpips_improvement"] for r in all_results]
    trans_loss_values = [
        (
            r["training_metrics"]["final_trans_loss"]
            if r["training_metrics"]["final_trans_loss"] is not None
            else 0.0
        )
        for r in all_results
    ]

    # Create summary plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # PSNR comparison
    axes[0, 0].bar(sample_indices, model_psnr, alpha=0.7, label="Model", color="blue")
    axes[0, 0].bar(sample_indices, bilinear_psnr, alpha=0.7, label="Bilinear", color="orange")
    axes[0, 0].set_xlabel("Sample Index")
    axes[0, 0].set_ylabel("PSNR (dB)")
    axes[0, 0].set_title("PSNR Comparison Across Samples")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # PSNR improvement
    colors = ["green" if x > 0 else "red" for x in psnr_improvement]
    axes[0, 1].bar(sample_indices, psnr_improvement, color=colors, alpha=0.7)
    axes[0, 1].axhline(y=0, color="black", linestyle="-", alpha=0.5)
    axes[0, 1].set_xlabel("Sample Index")
    axes[0, 1].set_ylabel("PSNR Improvement (dB)")
    axes[0, 1].set_title("PSNR Improvement (Model - Bilinear)")
    axes[0, 1].grid(True, alpha=0.3)

    # Transformation Loss
    axes[0, 2].bar(sample_indices, trans_loss_values, alpha=0.7, color="teal")
    axes[0, 2].set_xlabel("Sample Index")
    axes[0, 2].set_ylabel("Transformation Loss")
    axes[0, 2].set_title("Final Transformation Loss Across Samples")
    axes[0, 2].grid(True, alpha=0.3)

    # SSIM comparison
    axes[1, 0].bar(sample_indices, model_ssim, alpha=0.7, label="Model", color="purple")
    axes[1, 0].bar(sample_indices, bilinear_ssim, alpha=0.7, label="Bilinear", color="orange")
    axes[1, 0].set_xlabel("Sample Index")
    axes[1, 0].set_ylabel("SSIM")
    axes[1, 0].set_title("SSIM Comparison Across Samples")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # LPIPS comparison
    axes[1, 1].bar(sample_indices, model_lpips, alpha=0.7, label="Model", color="brown")
    axes[1, 1].bar(sample_indices, bilinear_lpips, alpha=0.7, label="Bilinear", color="orange")
    axes[1, 1].set_xlabel("Sample Index")
    axes[1, 1].set_ylabel("LPIPS")
    axes[1, 1].set_title("LPIPS Comparison Across Samples")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Overall improvement metrics
    axes[1, 2].bar(sample_indices, psnr_improvement, alpha=0.7, color="green")
    axes[1, 2].axhline(y=0, color="black", linestyle="-", alpha=0.5)
    axes[1, 2].set_xlabel("Sample Index")
    axes[1, 2].set_ylabel("Improvement (dB)")
    axes[1, 2].set_title("PSNR Improvement per Sample")
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "summary_metrics.png", bbox_inches="tight", pad_inches=0.1, dpi=300)
    plt.close()

    # Create box plots for aggregated metrics
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # PSNR box plot
    axes[0].boxplot([model_psnr, bilinear_psnr], labels=["Model", "Bilinear"])
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].set_title("PSNR Distribution Comparison")
    axes[0].grid(True, alpha=0.3)

    # SSIM box plot
    axes[1].boxplot([model_ssim, bilinear_ssim], labels=["Model", "Bilinear"])
    axes[1].set_ylabel("SSIM")
    axes[1].set_title("SSIM Distribution Comparison")
    axes[1].grid(True, alpha=0.3)

    # LPIPS box plot
    axes[2].boxplot([model_lpips, bilinear_lpips], labels=["Model", "Bilinear"])
    axes[2].set_ylabel("LPIPS")
    axes[2].set_title("LPIPS Distribution Comparison")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        output_dir / "metrics_distribution.png", bbox_inches="tight", pad_inches=0.1, dpi=300
    )
    plt.close()

    # Calculate and save aggregated statistics
    summary_stats = {
        "total_samples": len(all_results),
        "psnr": {
            "model_mean": np.mean(model_psnr),
            "model_std": np.std(model_psnr),
            "model_min": np.min(model_psnr),
            "model_max": np.max(model_psnr),
            "bilinear_mean": np.mean(bilinear_psnr),
            "bilinear_std": np.std(bilinear_psnr),
            "bilinear_min": np.min(bilinear_psnr),
            "bilinear_max": np.max(bilinear_psnr),
            "improvement_mean": np.mean(psnr_improvement),
            "improvement_std": np.std(psnr_improvement),
            "improvement_min": np.min(psnr_improvement),
            "improvement_max": np.max(psnr_improvement),
        },
        "ssim": {
            "model_mean": np.mean(model_ssim),
            "model_std": np.std(model_ssim),
            "model_min": np.min(model_ssim),
            "model_max": np.max(model_ssim),
            "bilinear_mean": np.mean(bilinear_ssim),
            "bilinear_std": np.std(bilinear_ssim),
            "bilinear_min": np.min(bilinear_ssim),
            "bilinear_max": np.max(bilinear_ssim),
            "improvement_mean": np.mean(ssim_improvement),
            "improvement_std": np.std(ssim_improvement),
            "improvement_min": np.min(ssim_improvement),
            "improvement_max": np.max(ssim_improvement),
        },
        "lpips": {
            "model_mean": np.mean(model_lpips),
            "model_std": np.std(model_lpips),
            "model_min": np.min(model_lpips),
            "model_max": np.max(model_lpips),
            "bilinear_mean": np.mean(bilinear_lpips),
            "bilinear_std": np.std(bilinear_lpips),
            "bilinear_min": np.min(bilinear_lpips),
            "bilinear_max": np.max(bilinear_lpips),
            "improvement_mean": np.mean(lpips_improvement),
            "improvement_std": np.std(lpips_improvement),
            "improvement_min": np.min(lpips_improvement),
            "improvement_max": np.max(lpips_improvement),
        },
        "transformation_loss": {
            "mean": np.mean(trans_loss_values),
            "std": np.std(trans_loss_values),
            "min": np.min(trans_loss_values),
            "max": np.max(trans_loss_values),
        },
    }

    # Save aggregated statistics to JSON
    with open(output_dir / "summary_statistics.json", "w") as f:
        json.dump(summary_stats, f, indent=2)

    # Save human-readable summary
    summary_text = f"""Multi-Sample Super-Resolution Results Summary
================================================

Total Samples Processed: {len(all_results)}

PSNR Results (dB):
------------------
Model Output:
  Mean: {summary_stats['psnr']['model_mean']:.2f} ± {summary_stats['psnr']['model_std']:.2f}
  Range: {summary_stats['psnr']['model_min']:.2f} - {summary_stats['psnr']['model_max']:.2f}

Bilinear Baseline:
  Mean: {summary_stats['psnr']['bilinear_mean']:.2f} ± {summary_stats['psnr']['bilinear_std']:.2f}
  Range: {summary_stats['psnr']['bilinear_min']:.2f} - {summary_stats['psnr']['bilinear_max']:.2f}

PSNR Improvement (Model - Bilinear):
  Mean: {summary_stats['psnr']['improvement_mean']:.2f} ± {summary_stats['psnr']['improvement_std']:.2f}
  Range: {summary_stats['psnr']['improvement_min']:.2f} - {summary_stats['psnr']['improvement_max']:.2f}

SSIM Results:
-------------
Model Output:
  Mean: {summary_stats['ssim']['model_mean']:.4f} ± {summary_stats['ssim']['model_std']:.4f}
  Range: {summary_stats['ssim']['model_min']:.4f} - {summary_stats['ssim']['model_max']:.4f}

Bilinear Baseline:
  Mean: {summary_stats['ssim']['bilinear_mean']:.4f} ± {summary_stats['ssim']['bilinear_std']:.4f}
  Range: {summary_stats['ssim']['bilinear_min']:.4f} - {summary_stats['ssim']['bilinear_max']:.4f}

SSIM Improvement (Model - Bilinear):
  Mean: {summary_stats['ssim']['improvement_mean']:.4f} ± {summary_stats['ssim']['improvement_std']:.4f}
  Range: {summary_stats['ssim']['improvement_min']:.4f} - {summary_stats['ssim']['improvement_max']:.4f}

LPIPS Results:
--------------
Model Output:
  Mean: {summary_stats['lpips']['model_mean']:.4f} ± {summary_stats['lpips']['model_std']:.4f}
  Range: {summary_stats['lpips']['model_min']:.4f} - {summary_stats['lpips']['model_max']:.4f}

Bilinear Baseline:
  Mean: {summary_stats['lpips']['bilinear_mean']:.4f} ± {summary_stats['lpips']['bilinear_std']:.4f}
  Range: {summary_stats['lpips']['bilinear_min']:.4f} - {summary_stats['lpips']['bilinear_max']:.4f}

LPIPS Improvement (Bilinear - Model):
  Mean: {summary_stats['lpips']['improvement_mean']:.4f} ± {summary_stats['lpips']['improvement_std']:.4f}
  Range: {summary_stats['lpips']['improvement_min']:.4f} - {summary_stats['lpips']['improvement_max']:.4f}

Transformation Loss Results:
----------------------------
Final Transformation Loss:
  Mean: {summary_stats['transformation_loss']['mean']:.6f} ± {summary_stats['transformation_loss']['std']:.6f}
  Range: {summary_stats['transformation_loss']['min']:.6f} - {summary_stats['transformation_loss']['max']:.6f}

Files Generated:
- summary_metrics.png: Bar charts comparing metrics across samples
- metrics_distribution.png: Box plots showing metric distributions
- summary_statistics.json: Detailed numerical statistics
- sample_XXX/: Individual results for each sample
"""

    with open(output_dir / "summary_report.txt", "w") as f:
        f.write(summary_text)

    print(f"\n{'='*60}")
    print("Summary Statistics")
    print(f"{'='*60}")
    print(
        f"PSNR Improvement: {summary_stats['psnr']['improvement_mean']:.2f} ± {summary_stats['psnr']['improvement_std']:.2f} dB"
    )
    print(
        f"SSIM Improvement: {summary_stats['ssim']['improvement_mean']:.4f} ± {summary_stats['ssim']['improvement_std']:.4f}"
    )
    print(
        f"LPIPS Improvement: {summary_stats['lpips']['improvement_mean']:.4f} ± {summary_stats['lpips']['improvement_std']:.4f}"
    )
    print(
        f"Average Transformation Loss: {summary_stats['transformation_loss']['mean']:.6f} ± {summary_stats['transformation_loss']['std']:.6f}"
    )
    print(f"{'='*60}\n")
    print(
        f"📊 Summary visualizations saved to {output_dir}/summary_metrics.png and {output_dir}/metrics_distribution.png"
    )
    print(
        f"📈 Aggregated statistics saved to {output_dir}/summary_statistics.json and {output_dir}/summary_report.txt"
    )


def _single_sample_output_dir(args) -> Path:
    """Directory for metrics/artifacts; matches run_experiment_suite layout when --run_name is set."""
    p = Path("single_samples") / args.dataset / str(args.sample_id)
    if args.run_name:
        return p / args.run_name
    return p


def main():
    parser = argparse.ArgumentParser(description="Minimal Satellite Super-Resolution Training")

    # Essential parameters only
    parser.add_argument(
        "--dataset",
        type=str,
        default="satburst_synth",
        choices=[
            "satburst_synth",
            "satburst_real",
            "worldstrat",
            "burst_synth",
            "worldstrat_test",
            "worldstrat_sweet",
            "worldstrat_bitter",
        ],
    )
    parser.add_argument("--sample_id", default="Landcover-743192_rgb")
    parser.add_argument(
        "--df",
        type=int,
        default=4,
        help="Downsampling factor, or upsampling factor for the data",
    )
    parser.add_argument(
        "--lr_degradation",
        type=str,
        default="area",
        choices=["area", "s2_psf"],
        help=(
            "HR→LR operator for recon loss: ``area`` (PyTorch area resize, default) or "
            "``s2_psf`` (per-band Gaussian blur in HR meters + avg-pool by inferred integer factor)."
        ),
    )
    parser.add_argument(
        "--s2_native_gsd_m",
        type=float,
        default=10.0,
        help="Native Sentinel-2 GSD in meters for PSF σ→pixel conversion (10 for 10 m bands).",
    )
    parser.add_argument(
        "--s2_psf_truncate",
        type=float,
        default=4.0,
        help="Gaussian 1D kernel half-width as multiples of σ (Sentinel-2 PSF blur).",
    )
    parser.add_argument(
        "--s2_psf_sigma_b02_m",
        type=float,
        default=3.8,
        help="PSF σ in meters for channel 2 / blue (S2 B02; RGB ch2).",
    )
    parser.add_argument(
        "--s2_psf_sigma_b03_m",
        type=float,
        default=3.5,
        help="PSF σ in meters for channel 1 / green (S2 B03; RGB ch1).",
    )
    parser.add_argument(
        "--s2_psf_sigma_b04_m",
        type=float,
        default=3.5,
        help="PSF σ in meters for channel 0 / red (S2 B04; RGB ch0).",
    )
    parser.add_argument(
        "--s2_psf_sigma_b08_m",
        type=float,
        default=4.0,
        help="PSF σ in meters for S2 B08 (four-channel paths only; RGB uses B04–B02).",
    )
    parser.add_argument(
        "--scale_factor", type=float, default=4, help="scale factor for the input training grid"
    )
    parser.add_argument(
        "--satburst_data_root",
        type=str,
        default=None,
        help=(
            "Parent folder for satburst-style scenes (subfolder per scene, then scale_*). "
            "Default: ``data`` for dataset satburst_synth, ``data_real`` for satburst_real. "
            "Real exports may use different LR/HR spatial sizes per scene; SRData reads each scene's manifest."
        ),
    )
    parser.add_argument(
        "--worldstrat_data_root",
        type=str,
        default=None,
        help=(
            "Parent folder for WorldStrat areas (subfolder per area with hr/ and lr/). "
            "Default: ``worldstrat_datasets/worldstrat_{sweet,bitter}`` or ``worldstrat_test_data``."
        ),
    )

    # Multi-sample optimization parameters
    parser.add_argument(
        "--multi_sample", action="store_true", help="Optimize against all samples in dataset"
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="multi_sample_results",
        help="Output folder for multi-sample results",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Subfolder under single_samples/<dataset>/<sample_id>/ for outputs (isolates experiment configs).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Training DataLoader batch size (satburst single-sample training typically uses 1).",
    )

    parser.add_argument("--lr_shift", type=float, default=1.0)
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument(
        "--aug", type=str, default="none", choices=["none", "light", "medium", "heavy"]
    )

    # Model parameters
    parser.add_argument(
        "--model",
        type=str,
        default="mlp",
        help="Decoder: mlp | mlp_tcnn | nir | hash_attn | fourier_band_attn.",
    )
    parser.add_argument("--network_depth", type=int, default=4)
    parser.add_argument("--network_hidden_dim", type=int, default=256)
    parser.add_argument("--projection_dim", type=int, default=256)
    parser.add_argument(
        "--input_projection",
        type=str,
        default="fourier_10",
        help="fourier_N, fourier, hashgrid_tcnn (aliases: hashgrid, hash), or none.",
    )
    parser.add_argument("--fourier_scale", type=float, default=10.0)
    parser.add_argument("--tcnn_mlp_dtype", type=str, default="fp32", choices=["fp16", "fp32"])
    parser.add_argument("--mlp_init", type=str, default="kaiming")

    parser.add_argument("--hash_encoding_preset", type=str, default="hash_smoothstep")
    parser.add_argument("--hash_grid_type", type=str, default="Hash")
    parser.add_argument("--hash_n_levels", type=int, default=16)
    parser.add_argument("--hash_n_features_per_level", type=int, default=2)
    parser.add_argument("--hash_log2_hashmap_size", type=int, default=21)
    parser.add_argument(
        "--hash_base_resolution",
        type=int,
        default=0,
        help=(
            "Coarsest HashGrid level resolution. 0 (default) = max(8, LR_side//4) from "
            "transform_log.json for satburst scenes; set explicitly to override."
        ),
    )
    parser.add_argument(
        "--hash_max_resolution",
        type=int,
        default=0,
        help=(
            "Finest HashGrid level resolution. 0 (default) = LR max side from "
            "transform_log.json for satburst scenes; set explicitly to override."
        ),
    )
    parser.add_argument("--hash_encoding_dtype", type=str, default="fp32", choices=["fp16", "fp32"])
    parser.add_argument("--hash_attn_token_dim", type=int, default=32)
    parser.add_argument(
        "--attn_token_dim",
        type=int,
        default=32,
        help="Token width for attention decoders (currently used by fourier_band_attn).",
    )
    parser.add_argument(
        "--fourier_num_bands",
        type=int,
        default=8,
        help=(
            "Number of frequency-band tokens for --model fourier_band_attn. Will fall back "
            "to the nearest divisor of n_bands (sequence 8 -> 4 -> 2 -> 1)."
        ),
    )
    parser.add_argument(
        "--log_attention",
        action="store_true",
        help=(
            "If set, periodically capture decoder attention diagnostics to "
            "<run_dir>/attention_log.json (sampled at the PSNR eval cadence)."
        ),
    )

    parser.add_argument(
        "--heldout_val_frames",
        type=int,
        default=0,
        help="If >0, last K frame ids (by sample_id) get zero recon gradient (held-out val protocol).",
    )
    parser.add_argument("--use_gnll", action="store_true")
    parser.add_argument(
        "--use_separate_ud",
        action="store_true",
        help="Use separate UD parameters for each sample (default: False)",
    )
    parser.add_argument(
        "--visualize_variance",
        action="store_true",
        help="Visualize variance maps for each LR sample when using GNLL (single sample only)",
    )
    parser.add_argument(
        "--no_variance_viz",
        action="store_true",
        help="Skip variance visualizations even when using GNLL (applies to both single and multi-sample modes)",
    )
    parser.add_argument(
        "--no_base_frame",
        action="store_true",
        help="Disable base frame (default: use_base_frame=True)",
    )
    parser.add_argument(
        "--no_direct_param_T",
        action="store_true",
        help="Disable direct parameter T (default: use_direct_param_T=True)",
    )
    parser.add_argument(
        "--use_color_shift",
        action="store_true",
        help="Use color shift (default: use_color_shift=False)",
    )

    # Training parameters
    parser.add_argument("--seed", type=int, default=6)
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--learning_rate", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw", "muon"])
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Adam / AdamW beta1")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="Adam / AdamW beta2")
    parser.add_argument("--adam_eps", type=float, default=1e-8, help="Adam / AdamW epsilon")
    parser.add_argument("--muon_momentum", type=float, default=0.95)
    parser.add_argument("--no_muon_nesterov", action="store_true")
    parser.add_argument("--muon_ns_steps", type=int, default=5)
    parser.add_argument("--muon_eps", type=float, default=1e-8)
    parser.add_argument(
        "--device",
        type=str,
        default="7",
        help="CUDA device number (e.g., '0', '1') or 'cpu' for CPU",
    )
    parser.add_argument(
        "--eval_mixed_precision",
        type=str,
        default="none",
        choices=["none", "auto", "fp16", "bfloat16"],
        help="Use mixed precision (FP16/BF16) for evaluation only; PSNR/metrics reported in this mode. none=float32.",
    )
    parser.add_argument(
        "--eval_crop_lr_size",
        type=int,
        default=0,
        help=(
            "If >0, top-left crop SR/GT/bilinear/LR to (eval_crop_lr_size * df) / eval_crop_lr_size "
            "before headline metrics and comparison.png (detail view for cross-resolution sweeps). "
            "Training is unaffected unless --train_crop_lr_size is set separately. "
            "comparison_full.png is saved when the full patch is larger than this crop. "
            "0 (default) disables eval cropping."
        ),
    )
    parser.add_argument(
        "--train_crop_lr_size",
        type=int,
        default=0,
        help=(
            "If >0, center-crop LR supervision and the HR coordinate grid during training to "
            "train_crop_lr_size (LR) and train_crop_lr_size*df (HR). Use with --eval_crop_lr_size "
            "on large scenes to avoid OOM from full-grid optimization. 0 = train on full resolution."
        ),
    )
    parser.add_argument(
        "--eval_crop_anchor",
        type=str,
        default="topleft",
        choices=["topleft", "center"],
        help=(
            "How to crop when eval_crop_lr_size>0. ``topleft``: [:64,:64] window (aligned across "
            "nested scene_center_v2_* exports). ``center``: independent center crop per scene."
        ),
    )

    args = parser.parse_args()
    args._hash_base_resolution_cli = int(args.hash_base_resolution)
    args._hash_max_resolution_cli = int(args.hash_max_resolution)

    # Setup device - allow "cpu" as explicit device string
    if args.device.lower() == "cpu":
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        cuda_count = torch.cuda.device_count()
        dev = str(args.device).strip().lower()
        try:
            if dev.startswith("cuda:"):
                requested_idx = int(dev.split(":", 1)[1])
            elif dev == "cuda":
                requested_idx = 0
            else:
                requested_idx = int(dev)
        except ValueError:
            requested_idx = 0
            print(f"Warning: invalid CUDA device '{args.device}'. Falling back to cuda:0.")

        if requested_idx < 0 or requested_idx >= cuda_count:
            print(
                f"Warning: CUDA device {requested_idx} is not available "
                f"(found {cuda_count} CUDA device(s)). Falling back to cuda:0."
            )
            requested_idx = 0

        device = torch.device(f"cuda:{requested_idx}")
    else:
        print(f"Warning: CUDA device {args.device} requested but CUDA not available. Using CPU.")
        device = torch.device("cpu")

    args.resolved_device = str(device)
    print(f"Using device: {device}")
    if device.type == "cuda":
        try:
            torch.cuda.set_device(device)
            torch.cuda.empty_cache()
        except Exception:
            pass
    if int(args.train_crop_lr_size) > 0:
        print(
            f"Training center crop: LR {int(args.train_crop_lr_size)} px, "
            f"HR {int(args.train_crop_lr_size) * int(args.df)} px (--train_crop_lr_size).",
            flush=True,
        )
    eval_autocast_dtype = get_eval_autocast_dtype(args.eval_mixed_precision, device)
    if eval_autocast_dtype is not None:
        label = "BF16" if eval_autocast_dtype == torch.bfloat16 else "FP16"
        print(f"Evaluation mixed precision: {label} (PSNR/metrics computed in this mode)")

    # Set seeds
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Setup dataset
    if args.dataset in ("satburst_synth", "satburst_real"):
        args.root_satburst_synth = satburst_scene_dir(args)
    elif args.dataset in WORLDSTRAT_DATASETS:
        args.root_worldstrat_test = resolve_worldstrat_data_root(args)
    elif args.dataset == "burst_synth":
        args.root_burst_synth = "SyntheticBurstVal"
        # Convert sample_id to integer for burst_synth dataset
        try:
            args.sample_id = int(args.sample_id)
        except ValueError:
            print(
                f"Warning: sample_id '{args.sample_id}' cannot be converted to integer for burst_synth dataset. Using 0 instead."
            )
            args.sample_id = 0

    # Handle multi-sample vs single-sample optimization
    if args.multi_sample:
        # Multi-sample optimization
        print(f"Starting multi-sample optimization for dataset: {args.dataset}")

        # Setup output directory
        output_dir = Path(args.output_folder)
        output_dir.mkdir(exist_ok=True)

        # Get all samples in the dataset
        if args.dataset in WORLDSTRAT_DATASETS:
            data_root = resolve_worldstrat_data_root(args)
            os.environ["WORLDSTRAT_TEST_ROOT"] = str(data_root)
            sample_ids = discover_worldstrat_sample_ids(data_root)
            print(f"Found {len(sample_ids)} WorldStrat areas under {data_root}: {sample_ids[:5]}...")
        elif args.dataset == "burst_synth":
            # For burst_synth, get all sample IDs from the gt folder
            if "DATA_DIR_ABSOLUTE" in os.environ:
                data_root = Path(os.environ["DATA_DIR_ABSOLUTE"])
            else:
                data_root = Path("SyntheticBurstVal")

            gt_dir = data_root / "gt"
            if gt_dir.exists():
                sample_dirs = [d for d in gt_dir.iterdir() if d.is_dir()]
                sample_ids = [int(d.name) for d in sample_dirs if d.name.isdigit()]
                sample_ids.sort()
                print(f"Found {len(sample_ids)} samples: {sample_ids[:5]}...")
            else:
                print(f"Error: GT directory {gt_dir} not found!")
                return
        elif args.dataset in ("satburst_synth", "satburst_real"):
            data_root = Path(resolve_satburst_data_root(args))
            if not data_root.exists():
                print(f"Error: satburst data root {data_root} not found!")
                return
            scale_leaf = f"scale_{args.df}_shift_{args.lr_shift:.1f}px_aug_{args.aug}"
            sample_dirs = [
                d
                for d in data_root.iterdir()
                if d.is_dir() and not d.name.startswith(".") and (d / scale_leaf).is_dir()
            ]
            sample_ids = [d.name for d in sample_dirs]
            sample_ids.sort()
            print(
                f"Found {len(sample_ids)} samples under {data_root} (with {scale_leaf}/): {sample_ids[:5]}..."
            )
        else:
            print(f"Error: Unsupported dataset for multi-sample: {args.dataset}")
            return

        output_dim = 3 + args.num_samples * 3 if args.use_gnll and not args.use_separate_ud else 3

        # Run optimization for each sample
        all_results = []
        for sample_idx, sample_id in enumerate(sample_ids):
            print(f"\n{'='*60}")
            print(f"Processing sample {sample_idx + 1}/{len(sample_ids)}: {sample_id}")
            print(f"{'='*60}")

            args.sample_id = sample_id
            if args.dataset in ("satburst_synth", "satburst_real"):
                args.root_satburst_synth = satburst_scene_dir(args)

            if args.dataset in WORLDSTRAT_DATASETS:
                args.root_worldstrat_test = resolve_worldstrat_data_root(args)
            train_data = get_dataset(args=args, name=args.dataset)

            input_projection, decoder = build_input_projection_decoder_bundle(
                args, device, output_dim=output_dim
            )

            print(
                f"🔄 Creating fresh model for sample {sample_id} (sample {sample_idx + 1}/{len(sample_ids)})"
            )
            model = build_inr_model(
                input_projection,
                decoder,
                args,
                device,
            )
            print(f"✅ Fresh model created and initialized")

            result = optimize_and_evaluate_sample(
                model, train_data, device, sample_idx, args, output_dir
            )
            all_results.append(result)

            use_gnll_loss = model.use_gnll
            if use_gnll_loss and not args.no_variance_viz:
                sample_dir = output_dir / f"sample_{sample_idx:03d}"
                print(f"\nGenerating variance visualizations for sample {sample_id}...")
                torch.cuda.empty_cache()  # Clear GPU memory
                visualize_lr_variance(model, train_data, device, sample_dir, sample_id)

        # Create summary visualizations
        create_summary_visualization(all_results, output_dir)
        return

    elif args.dataset == "burst_synth":
        # Set the path to SyntheticBurstVal
        if "DATA_DIR_ABSOLUTE" in os.environ:
            args.root_burst_synth = os.environ["DATA_DIR_ABSOLUTE"]
        else:
            args.root_burst_synth = "SyntheticBurstVal"

    train_data = get_dataset(args=args, name=args.dataset)
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)

    # Setup model
    input_projection, decoder = build_input_projection_decoder_bundle(args, device, output_dim=3)
    model = build_inr_model(
        input_projection,
        decoder,
        args,
        device,
    )
    # model = NIR(input_projection, decoder, args.num_samples, use_gnll=args.use_gnll).to(device)

    # Setup optimizer
    optimizer = build_optimizer(model.parameters(), args)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.iters, eta_min=1e-6)

    print(f"Starting training for {args.iters} iterations...")

    run_start_time = time.time()
    training_start_time = run_start_time

    # Training loop
    iteration = 0
    progress_bar = tqdm(total=args.iters, desc="Training")

    # Lists to store PSNR and losses for plotting
    psnr_list = []
    recon_loss_list = []
    trans_loss_list = []
    total_loss_list = []
    iteration_list = []
    attention_log = []

    while iteration < args.iters:
        for train_sample in train_dataloader:
            if iteration >= args.iters:
                break

            # Train one iteration
            train_losses = train_one_iteration(
                model,
                optimizer,
                train_sample,
                device,
                args,
                iteration,
            )

            # Check for NaN/Inf in losses and break if detected
            if (
                torch.isnan(torch.tensor(train_losses["recon_loss"]))
                or torch.isinf(torch.tensor(train_losses["recon_loss"]))
                or torch.isnan(torch.tensor(train_losses["total_loss"]))
                or torch.isinf(torch.tensor(train_losses["total_loss"]))
            ):
                print(f"\nERROR: NaN/Inf detected in losses at iteration {iteration}")
                print(f"Reconstruction loss: {train_losses['recon_loss']}")
                print(f"Total loss: {train_losses['total_loss']}")
                print("Stopping training to prevent further issues.")
                break

            scheduler.step()
            iteration += 1

            # Update progress bar
            progress_bar.update(1)
            postfix_dict = {
                "recon": f"{train_losses['recon_loss']:.4f}",
                "trans": f"{train_losses['trans_loss']:.4f}",
            }
            progress_bar.set_postfix(postfix_dict)

            # Periodic evaluation
            if iteration % 100 == 0:
                eval_autocast_dtype = get_eval_autocast_dtype(args.eval_mixed_precision, device)
                test_loss, test_psnr = test_one_epoch(
                    model,
                    train_data,
                    device,
                    eval_autocast_dtype,
                    eval_crop_lr_size=int(getattr(args, "eval_crop_lr_size", 0) or 0),
                    df=int(args.df),
                    crop_anchor=str(getattr(args, "eval_crop_anchor", "topleft")),
                )
                print(
                    f"\nIter {iteration}: Train Loss: {train_losses['total_loss']:.6f}, "
                    f"Test Loss: {test_loss:.6f}, Test PSNR: {test_psnr:.2f} dB"
                )

                # Additional debugging for GNLL
                if model.use_gnll and (
                    torch.isnan(torch.tensor(train_losses["recon_loss"]))
                    or torch.isinf(torch.tensor(train_losses["recon_loss"]))
                ):
                    print(
                        f"WARNING: NaN/Inf detected in reconstruction loss at iteration {iteration}"
                    )
                    print(f"Reconstruction loss: {train_losses['recon_loss']}")
                    print(f"Total loss: {train_losses['total_loss']}")

                # Append to lists for plotting
                iteration_list.append(iteration)
                psnr_list.append(test_psnr)
                recon_loss_list.append(train_losses["recon_loss"])
                trans_loss_list.append(train_losses["trans_loss"])
                total_loss_list.append(train_losses["total_loss"])

                if getattr(args, "log_attention", False):
                    aux = getattr(model, "last_decoder_aux", {}) or {}
                    if "attn_mean" in aux:
                        attention_log.append(
                            {
                                "iteration": int(iteration),
                                "attention_mean_per_level": aux["attn_mean"]
                                .detach()
                                .cpu()
                                .tolist(),
                                "attention_entropy": float(
                                    aux["attn_entropy_mean"].detach().cpu()
                                ),
                                "fine_level_mass": float(
                                    aux["attn_fine_mass_last3"].detach().cpu()
                                ),
                            }
                        )

    progress_bar.close()

    training_end_time = time.time()
    training_time = training_end_time - training_start_time
    completed_iters = max(int(iteration), 1)
    time_per_iteration = training_time / completed_iters

    # Final evaluation and save output
    evaluation_start_time = time.time()
    model.eval()
    eval_autocast_dtype = get_eval_autocast_dtype(args.eval_mixed_precision, device)
    with torch.no_grad():
        hr_coords = train_data.get_hr_coordinates().unsqueeze(0).to(device)
        hr_image = train_data.get_original_hr().unsqueeze(0).to(device)
        sample_id = torch.tensor([0]).to(device)

        if eval_autocast_dtype is not None:
            with torch.autocast(device_type="cuda", dtype=eval_autocast_dtype):
                if model.use_gnll:
                    output, _ = model(hr_coords, sample_id, scale_factor=1, training=False)
                else:
                    output, _ = model(hr_coords, sample_id, scale_factor=1, training=False)
        else:
            if model.use_gnll:
                output, _ = model(hr_coords, sample_id, scale_factor=1, training=False)
            else:
                output, _ = model(hr_coords, sample_id, scale_factor=1, training=False)

        # Unstandardize the output
        output = output * train_data.get_lr_std(0).to(device) + train_data.get_lr_mean(0).to(device)

        final_test_loss = F.mse_loss(output, hr_image).item()
        final_psnr = -10 * torch.log10(torch.tensor(final_test_loss)).item()

        pred_tensor = _hwc_to_bchw(output).clamp(0, 1)
        gt_tensor = _hwc_to_bchw(hr_image).clamp(0, 1)

        # Build a 3-channel LR baseline image for visualization
        if hasattr(train_data, "get_lr_sample_hwc"):
            # get_lr_sample_hwc returns standardized HWC format, need to unstandardize
            lr_original = train_data.get_lr_sample_hwc(0).cpu().numpy()  # H x W x 3 (standardized)
            lr_std = train_data.get_lr_std(0).cpu().numpy()
            lr_mean = train_data.get_lr_mean(0).cpu().numpy()
            # Ensure shapes broadcast to HxWx3
            if lr_std.ndim == 1:
                lr_std = lr_std.reshape(1, 1, -1)
            if lr_mean.ndim == 1:
                lr_mean = lr_mean.reshape(1, 1, -1)
            lr_original = lr_original * lr_std + lr_mean
        else:
            # get_lr_sample returns unstandardized CHW format (already unstandardized in data.py line 207)
            lr_original = (
                train_data.get_lr_sample(0).cpu().numpy()
            )  # C x H x W (unstandardized, [0, 1])

            # Convert from CHW to HWC for visualization
            if lr_original.ndim == 3:
                if lr_original.shape[0] in (1, 3, 4):  # CHW format
                    lr_original = lr_original.transpose(1, 2, 0)  # Convert to HWC
                    # Handle multi-frame case if needed (shouldn't happen for satburst_synth, but be safe)
                    if lr_original.shape[2] > 3:
                        H, W, C = lr_original.shape
                        if C % 3 == 0:
                            T = C // 3
                            lr_original = lr_original.reshape(H, W, T, 3)
                            # Use first frame as baseline
                            lr_original = lr_original[:, :, 0, :]
                        else:
                            # Fallback: take first 3 channels
                            lr_original = lr_original[:, :, :3]
            # No unstandardization needed - get_lr_sample already returns unstandardized [0, 1] range

        lr_h, lr_w = lr_original.shape[:2]
        hr_h, hr_w = hr_image.shape[1], hr_image.shape[2]
        lr_bilinear = cv2.resize(lr_original, (hr_w, hr_h), interpolation=cv2.INTER_LINEAR)
        lr_original = np.clip(lr_original, 0, 1)
        lr_bilinear = np.clip(lr_bilinear, 0, 1)

        bilinear_tensor = (
            torch.from_numpy(lr_bilinear).unsqueeze(0).permute(0, 3, 1, 2).to(device)
        )

        # Align outputs for fair comparison (following og_main.py approach)
        # Alignment disabled to avoid OOM errors - can be re-enabled if needed
        print("Skipping alignment (disabled to avoid memory issues)")
        pred_aligned = pred_tensor
        bilinear_aligned = bilinear_tensor

        # Keep full-patch tensors for a second visualization when eval crop is active.
        pred_full_bchw = pred_aligned
        gt_full_bchw = gt_tensor
        bilinear_full_bchw = bilinear_aligned
        lr_full_hwc = np.clip(lr_original, 0.0, 1.0)
        lr_bilinear_full_hwc = np.clip(lr_bilinear, 0.0, 1.0)

        # Optional center-crop of SR/GT/bilinear/LR before metrics+viz so different LR sizes (e.g.
        # 64/128/256/512) are evaluated on the same physical area on the ground.
        eval_crop_info = apply_eval_center_crop(
            pred_bchw=pred_aligned,
            bilinear_bchw=bilinear_aligned,
            gt_bchw=gt_tensor,
            lr_hwc=lr_original,
            lr_bilinear_hwc=lr_bilinear,
            crop_lr_size=int(getattr(args, "eval_crop_lr_size", 0) or 0),
            df=int(args.df),
            crop_anchor=str(getattr(args, "eval_crop_anchor", "topleft")),
        )
        pred_aligned = eval_crop_info["pred_bchw"]
        bilinear_aligned = eval_crop_info["bilinear_bchw"]
        gt_tensor = eval_crop_info["gt_bchw"]
        lr_original = eval_crop_info["lr_hwc"]
        lr_bilinear = eval_crop_info["lr_bilinear_hwc"]
        if eval_crop_info["cropped"]:
            print(
                f"Eval crop ({eval_crop_info.get('crop_anchor', 'topleft')}): LR -> "
                f"{eval_crop_info['lr_crop']}px, HR/SR/GT -> {eval_crop_info['hr_crop']}px "
                f"(--eval_crop_lr_size={int(args.eval_crop_lr_size)} * df={int(args.df)})."
            )
            # Recompute the "final test" PSNR/loss on the cropped region so the saved summary
            # numbers also reflect the same physical area as the headline metrics below.
            final_test_loss = F.mse_loss(pred_aligned, gt_tensor).item()
            final_psnr = -10 * torch.log10(torch.tensor(final_test_loss)).item()

        # Keep gt_np in sync with the (possibly cropped) gt_tensor for visualization below.
        gt_np = gt_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        gt_np = np.clip(gt_np, 0, 1)

        pred_cpu = pred_aligned.detach().cpu()
        gt_cpu = gt_tensor.detach().cpu()
        bilinear_cpu = bilinear_aligned.detach().cpu()

        model_psnr = peak_signal_noise_ratio(pred_cpu, gt_cpu, data_range=1.0).item()
        bilinear_psnr = peak_signal_noise_ratio(bilinear_cpu, gt_cpu, data_range=1.0).item()

        model_ssim = ssim(pred_cpu, gt_cpu, data_range=1.0).item()
        bilinear_ssim = ssim(bilinear_cpu, gt_cpu, data_range=1.0).item()

        lpips_fn = lpips.LPIPS(net="vgg").to(device)
        pred_lpips = lpips_fn((pred_aligned * 2 - 1), (gt_tensor * 2 - 1)).item()
        bilinear_lpips = lpips_fn((bilinear_aligned * 2 - 1), (gt_tensor * 2 - 1)).item()

        # Convert aligned tensors back to numpy for visualization
        pred_aligned_np = pred_aligned.squeeze(0).permute(1, 2, 0).cpu().numpy()
        bilinear_aligned_np = bilinear_aligned.squeeze(0).permute(1, 2, 0).cpu().numpy()

        # Ensure aligned images are in valid range
        pred_aligned_np = np.clip(pred_aligned_np, 0, 1)
        bilinear_aligned_np = np.clip(bilinear_aligned_np, 0, 1)

        # Create structured output directory for single sample results
        sample_dir = _single_sample_output_dir(args)
        sample_dir.mkdir(parents=True, exist_ok=True)

        crop_banner = ""
        if eval_crop_info["cropped"]:
            lr_c, hr_c = eval_crop_info["lr_crop"], eval_crop_info["hr_crop"]
            crop_banner = (
                f"Eval crop ({eval_crop_info.get('crop_anchor', 'topleft')}): "
                f"LR {lr_c}×{lr_c} → HR {hr_c}×{hr_c}"
            )

        _save_comparison_quad(
            sample_dir,
            "comparison.png",
            lr_original,
            bilinear_aligned_np,
            pred_aligned_np,
            gt_np,
            model_psnr=model_psnr,
            bilinear_psnr=bilinear_psnr,
            banner=crop_banner,
        )
        comparison_path = sample_dir / "comparison.png"

        _save_full_comparison_if_needed(
            sample_dir,
            pred_full_bchw=pred_full_bchw,
            gt_full_bchw=gt_full_bchw,
            bilinear_full_bchw=bilinear_full_bchw,
            lr_full_hwc=lr_full_hwc,
            lr_bilinear_full_hwc=lr_bilinear_full_hwc,
            pred_crop_bchw=pred_aligned,
            lr_h_full=lr_h,
            lr_w_full=lr_w,
        )

        # Save individual images for reference (using aligned images)
        plt.figure(figsize=(8, 8))
        plt.imshow(pred_aligned_np)
        plt.axis("off")
        plt.tight_layout(pad=0)
        pred_path = sample_dir / "model_output_aligned.png"
        plt.savefig(pred_path, bbox_inches="tight", pad_inches=0, dpi=300)
        plt.close()

        plt.figure(figsize=(8, 8))
        plt.imshow(gt_np)
        plt.axis("off")
        plt.tight_layout(pad=0)
        gt_path = sample_dir / "ground_truth.png"
        plt.savefig(gt_path, bbox_inches="tight", pad_inches=0, dpi=300)
        plt.close()

        # Save bilinear baseline for reference (aligned version)
        plt.figure(figsize=(8, 8))
        plt.imshow(bilinear_aligned_np)
        plt.axis("off")
        plt.tight_layout(pad=0)
        bilinear_path = sample_dir / "bilinear_baseline.png"
        plt.savefig(bilinear_path, bbox_inches="tight", pad_inches=0, dpi=300)
        plt.close()

        # Save LR original for reference
        plt.figure(figsize=(8, 8))
        plt.imshow(lr_original)
        plt.axis("off")
        plt.tight_layout(pad=0)
        lr_path = sample_dir / "lr_original.png"
        plt.savefig(lr_path, bbox_inches="tight", pad_inches=0, dpi=300)
        plt.close()

        output_path = comparison_path

    print(f"\nFinal Results:")
    print(f"Test Loss: {final_test_loss:.6f}")
    print(f"Test PSNR: {final_psnr:.2f} dB")
    print(f"Model PSNR: {model_psnr:.2f} dB")
    print(f"Bilinear PSNR: {bilinear_psnr:.2f} dB")
    print(f"PSNR Improvement: {model_psnr - bilinear_psnr:.2f} dB")
    print(f"Model output saved to {output_path}")

    evaluation_end_time = time.time()
    evaluation_time = evaluation_end_time - evaluation_start_time
    total_runtime = evaluation_end_time - run_start_time

    # Create structured output directory for single sample results
    sample_dir = _single_sample_output_dir(args)
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Save PSNR results to a text file in the structured directory
    eval_crop_line = (
        f"Eval Center Crop: LR={eval_crop_info['lr_crop']}px, HR={eval_crop_info['hr_crop']}px"
        if eval_crop_info["cropped"]
        else "Eval Center Crop: disabled"
    )

    results_text = f"""Super-Resolution Results
    =======================

    Dataset: {args.dataset}
    Sample ID: {args.sample_id}
    Downsampling Factor: {args.df}
    Model: {args.model}
    Iterations: {args.iters}
    {eval_crop_line}

    PSNR Results:
    - Model Output: {model_psnr:.2f} dB
    - Bilinear Interpolation: {bilinear_psnr:.2f} dB
    - PSNR Improvement: {model_psnr - bilinear_psnr:.2f} dB

    SSIM Results:
    - Model Output: {model_ssim:.4f}
    - Bilinear Interpolation: {bilinear_ssim:.4f}
    - SSIM Improvement: {model_ssim - bilinear_ssim:.4f}

    LPIPS Results:
    - Model Output: {pred_lpips:.4f}
    - Bilinear Interpolation: {bilinear_lpips:.4f}
    - LPIPS Improvement: {bilinear_lpips - pred_lpips:.4f}

    Training Results:
    - Final Test Loss: {final_test_loss:.6f}
    - Final Test PSNR: {final_psnr:.2f} dB
    - Final Reconstruction Loss: {recon_loss_list[-1] if recon_loss_list else 0:.6f}
    - Final Transformation Loss: {trans_loss_list[-1] if trans_loss_list else 0:.6f}
    - Final Total Loss: {total_loss_list[-1] if total_loss_list else 0:.6f}

    Timing:
    - Training time: {training_time:.2f} s ({training_time / 60.0:.2f} min)
    - Completed iterations: {completed_iters}
    - Time per iteration: {time_per_iteration:.4f} s
    - Final evaluation time: {evaluation_time:.2f} s
    - Total runtime: {total_runtime:.2f} s ({total_runtime / 60.0:.2f} min)

    Training Metrics History:
    """

    if len(psnr_list) > 0:
        results_text += f"- Number of evaluation points: {len(psnr_list)}\n"
        results_text += f"- PSNR range: {min(psnr_list):.2f} - {max(psnr_list):.2f} dB\n"
        results_text += f"- Reconstruction loss range: {min(recon_loss_list):.6f} - {max(recon_loss_list):.6f}\n"
        results_text += f"- Transformation loss range: {min(trans_loss_list):.6f} - {max(trans_loss_list):.6f}\n"
        results_text += (
            f"- Total loss range: {min(total_loss_list):.6f} - {max(total_loss_list):.6f}\n"
        )
        results_text += f"- Final PSNR: {psnr_list[-1]:.2f} dB\n"
        results_text += f"- Final reconstruction loss: {recon_loss_list[-1]:.6f}\n"
        results_text += f"- Final transformation loss: {trans_loss_list[-1]:.6f}\n"
        results_text += f"- Final total loss: {total_loss_list[-1]:.6f}\n"
    else:
        results_text += "- No training metrics recorded (training may have been too short)\n"

    # Save to both current directory (for backward compatibility) and structured directory
    with open("psnr_results.txt", "w") as f:
        f.write(results_text)

    with open(sample_dir / "metrics.txt", "w") as f:
        f.write(results_text)

    # Save metrics as JSON for easier parsing
    metrics_dict = {
        "dataset": args.dataset,
        "sample_id": str(args.sample_id),
        "run_name": args.run_name,
        "command": shlex.join(sys.argv),
        "downsampling_factor": args.df,
        "lr_degradation": str(getattr(args, "lr_degradation", "area")),
        "model": args.model,
        "input_projection": args.input_projection,
        "hash_base_resolution": int(getattr(args, "hash_base_resolution", 0)),
        "hash_max_resolution": int(getattr(args, "hash_max_resolution", 0)),
        "iterations": args.iters,
        "learning_rate": args.learning_rate,
        "psnr": {
            "model": model_psnr,
            "bilinear": bilinear_psnr,
            "improvement": model_psnr - bilinear_psnr,
        },
        "ssim": {
            "model": model_ssim,
            "bilinear": bilinear_ssim,
            "improvement": model_ssim - bilinear_ssim,
        },
        "lpips": {
            "model": pred_lpips,
            "bilinear": bilinear_lpips,
            "improvement": bilinear_lpips - pred_lpips,
        },
        "training": {
            "final_test_loss": final_test_loss,
            "final_test_psnr": final_psnr,
            "final_recon_loss": recon_loss_list[-1] if recon_loss_list else 0,
            "final_trans_loss": trans_loss_list[-1] if trans_loss_list else 0,
            "final_total_loss": total_loss_list[-1] if total_loss_list else 0,
            "completed_iterations": completed_iters,
            "training_time_seconds": training_time,
            "time_per_iteration_seconds": time_per_iteration,
            "evaluation_time_seconds": evaluation_time,
            "total_runtime_seconds": total_runtime,
        },
        "train_crop": {
            "requested_lr_size": int(getattr(args, "train_crop_lr_size", 0) or 0),
            "anchor": str(getattr(args, "eval_crop_anchor", "topleft")),
        },
        "eval_crop": {
            "requested_lr_size": int(getattr(args, "eval_crop_lr_size", 0) or 0),
            "anchor": str(eval_crop_info.get("crop_anchor", getattr(args, "eval_crop_anchor", "topleft"))),
            "applied": bool(eval_crop_info["cropped"]),
            "lr_crop": int(eval_crop_info["lr_crop"]),
            "hr_crop": int(eval_crop_info["hr_crop"]),
            "mosaic_origin": getattr(train_data, "mosaic_origin", None),
        },
    }

    aux = getattr(model, "last_decoder_aux", {}) or {}
    if "attn_mean" in aux:
        metrics_dict["attention"] = {
            "mean_by_level": aux["attn_mean"].detach().cpu().tolist(),
            "entropy_mean": float(aux["attn_entropy_mean"].detach().cpu()),
            "fine_mass_last3": float(aux["attn_fine_mass_last3"].detach().cpu()),
        }
        attn_mean = aux["attn_mean"].detach().cpu().numpy()
        is_fourier_attn = str(getattr(args, "model", "")) == "fourier_band_attn"
        token_axis = "Fourier band index" if is_fourier_attn else "HashGrid level index"
        title_axis = "Fourier band" if is_fourier_attn else "HashGrid level"
        fig_attn, ax_attn = plt.subplots(figsize=(8, 4))
        levels = np.arange(len(attn_mean))
        ax_attn.bar(levels, attn_mean)
        ax_attn.set_xlabel(token_axis)
        ax_attn.set_ylabel("Mean attention weight")
        ax_attn.set_title(f"Mean attention by {title_axis}")
        fig_attn.tight_layout()
        fig_attn.savefig(
            sample_dir / "attention_mean_by_level.png", bbox_inches="tight", pad_inches=0.1, dpi=300
        )
        plt.close(fig_attn)

    with open(sample_dir / "metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=2)

    if getattr(args, "log_attention", False) and attention_log:
        with open(sample_dir / "attention_log.json", "w") as f:
            json.dump(attention_log, f, indent=2)
        print(f"Attention log saved to {sample_dir / 'attention_log.json'}")

    print(f"Results saved to: {sample_dir}")
    print(f"PSNR results also saved to psnr_results.txt (current directory)")

    # Plot PSNR and all losses
    if len(psnr_list) > 0:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plot PSNR on top subplot
        ax1.plot(iteration_list, psnr_list, color="blue", linewidth=2, label="PSNR (Test)")
        ax1.set_xlabel("Iteration", fontsize=12)
        ax1.set_ylabel("PSNR (dB)", fontsize=12)
        ax1.set_title("Training PSNR Evolution", fontsize=14, fontweight="bold")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Plot all losses on bottom subplot
        ax2.plot(
            iteration_list, recon_loss_list, color="red", linewidth=2, label="Reconstruction Loss"
        )
        ax2.plot(
            iteration_list, trans_loss_list, color="green", linewidth=2, label="Transformation Loss"
        )
        ax2.plot(iteration_list, total_loss_list, color="purple", linewidth=2, label="Total Loss")
        ax2.set_xlabel("Iteration", fontsize=12)
        ax2.set_ylabel("Loss", fontsize=12)
        ax2.set_title("Training Loss Evolution", fontsize=14, fontweight="bold")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        # Save to both current directory (for backward compatibility) and structured directory
        plt.savefig("training_metrics.png", bbox_inches="tight", pad_inches=0.1, dpi=300)
        plt.savefig(
            sample_dir / "training_metrics.png", bbox_inches="tight", pad_inches=0.1, dpi=300
        )
        plt.close()

        print(
            f"Training metrics plot saved to training_metrics.png and {sample_dir}/training_metrics.png"
        )
    else:
        print("No metrics data available for plotting (training may have been too short)")

    # Generate variance visualizations if requested and using GNLL
    # Note: For multi_sample mode, variance visualization is done in the multi_sample loop above
    use_gnll_loss = model.use_gnll
    if (
        args.visualize_variance
        and use_gnll_loss
        and not args.multi_sample
        and not args.no_variance_viz
    ):
        print("Generating variance visualizations for each LR sample...")
        # Clear GPU memory before variance visualization
        torch.cuda.empty_cache()
        visualize_lr_variance(model, train_data, device, sample_dir, args.sample_id)
    elif args.visualize_variance and not use_gnll_loss:
        print(
            "Warning: --visualize_variance requested but model does not use GNLL. Skipping variance visualization."
        )
    # For multi_sample mode, variance is automatically generated if use_gnll is enabled (unless --no_variance_viz is set)


if __name__ == "__main__":
    main()
