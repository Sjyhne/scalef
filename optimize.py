import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from pathlib import Path
import random
import argparse
import cv2
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchmetrics.functional.image import peak_signal_noise_ratio
import matplotlib.pyplot as plt
import json
from datetime import datetime
import importlib.metadata
import platform
import subprocess
import sys
import threading

from data import get_dataset, resolve_dataset_device
from utils import bilinear_resize_torch, align_output_to_target, get_valid_mask
from losses import BasicLosses, laplace_nll_loss, resolve_recon_criterion
from models.utils import get_decoder
from input_projections.utils import get_input_projection
from models.inr import INR
from models.nir import NIR
from optimizers import build_optimizer
from eval.spot_metrics import DEFAULT_SPOT_HR_PX, compute_fixed_spot_metrics
from eval.masked_metrics import compute_masked_image_metrics
from eval.hr_render import render_hr_rgb_tiled, resolve_hr_render_tile
from eval.visualize import save_eval_visualizations
from eval.export_geotiff import export_qgis_layers
from eval.learned_affines import write_affine_dump
from eval.halo_consistency import (
    attach_halo_to_dataset,
    halo_consistency_loss,
    parse_halo_sides,
    sample_halo_batch,
)
from eval.lr_holdout import (
    EarlyStopState,
    apply_lr_clear_masks,
    compute_holdout_val_loss,
    elementwise_recon,
    gather_train_masks,
    holdout_psf_pad_lr,
    init_early_stop_state,
    masked_mean,
    resolve_early_stop_score,
    default_early_stop_ema,
    default_early_stop_regression,
)
from models.training_schedule import effective_lr_align_args
from models.lr_tile_sampler import (
    build_cross_frame_tile_sampler,
    build_lr_tile_sampler,
    raw_tiles_per_step,
    resolve_lr_tile_mix,
    stack_cross_frame_tiles,
    stack_lr_hr_tiles,
)

import time

import os
import lpips
from torchmetrics.functional.image import structural_similarity_index_measure as ssim

LOG_POSTFIX_INTERVAL = 20
_GNLL_RECON_CRITERION = nn.GaussianNLLLoss()
_GNLL_RECON_NONE = nn.GaussianNLLLoss(reduction="none")


def _reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()


def _peak_memory_gb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return float(torch.cuda.max_memory_allocated(device) / (1024**3))


def _maybe_apply_lr_cloud_masks(holdout_state, train_data, args):
    """AND SCL/cloud clear maps into holdout train/val masks when present."""
    if holdout_state is None:
        return None
    if not bool(getattr(args, "mask_lr_clouds", True)):
        return holdout_state
    if not bool(getattr(train_data, "has_lr_cloud_mask", False)):
        return holdout_state
    clear = getattr(train_data, "lr_clear", None)
    if clear is None:
        return holdout_state
    return apply_lr_clear_masks(holdout_state, clear)


def _json_safe(value):
    """Recursively convert argparse/provenance values into JSON primitives."""
    if isinstance(value, argparse.Namespace):
        value = vars(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_provenance(repo_dir: Path) -> dict:
    result = {"commit": None, "dirty": None}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        result["commit"] = commit.stdout.strip() or None
        result["dirty"] = bool(status.stdout)
    except (OSError, subprocess.SubprocessError):
        pass
    return result


def _experiment_provenance(args, device: torch.device, parsed_args: dict) -> dict:
    gpu_name = None
    if device.type == "cuda":
        try:
            gpu_name = torch.cuda.get_device_name(device)
        except (RuntimeError, AssertionError):
            pass
    return {
        "git": _git_provenance(Path(__file__).resolve().parent),
        "versions": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda": torch.version.cuda,
            "tinycudann": _package_version("tinycudann"),
        },
        "python_executable": sys.executable,
        "gpu_name": gpu_name,
        "seed": int(getattr(args, "seed", 0)),
        "args": _json_safe(parsed_args),
    }


def _model_parameter_counts(model: nn.Module) -> dict[str, int]:
    parameters = list(model.parameters())
    return {
        "total": sum(parameter.numel() for parameter in parameters),
        "trainable": sum(parameter.numel() for parameter in parameters if parameter.requires_grad),
    }


def _training_batch_query_counts(
    coords: torch.Tensor, lr_target: torch.Tensor, supervision_channels: int
) -> dict[str, int]:
    """Count the post-sampling workload passed to one training step."""
    lr_pixels = int(np.prod(lr_target.shape[:-1])) if lr_target.ndim else 0
    lr_elements = int(lr_target.numel())
    hr_pixels = int(np.prod(coords.shape[:-1])) if coords.ndim else 0
    return {
        "lr_pixels": lr_pixels,
        "lr_elements": lr_elements,
        "hr_pixels": hr_pixels,
        "hr_elements": hr_pixels * max(1, int(supervision_channels)),
    }


def _summarize_training_instrumentation(
    step_count: int, step_time_seconds: float, query_totals: dict[str, int]
) -> dict:
    steps = max(0, int(step_count))
    elapsed = max(0.0, float(step_time_seconds))
    return {
        "step_count": steps,
        "total_step_time_seconds": elapsed,
        "average_step_time_seconds": elapsed / steps if steps else None,
        "iterations_per_second": steps / elapsed if elapsed > 0.0 else None,
        "queried": {
            "estimate_basis": "actual_post_sampling_train_one_iteration_inputs",
            "totals": {key: int(value) for key, value in query_totals.items()},
            "average_per_step": {
                key: (float(value) / steps if steps else None)
                for key, value in query_totals.items()
            },
        },
    }


def _sample_process_gpu_memory_bytes() -> tuple[int | None, str | None]:
    """Snapshot this process's device memory, including non-PyTorch arenas."""
    pid = os.getpid()
    try:
        import pynvml  # type: ignore[import-not-found]

        pynvml.nvmlInit()
        used = 0
        found = False
        try:
            for index in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                for process in processes:
                    if int(process.pid) == pid:
                        used += int(process.usedGpuMemory)
                        found = True
        finally:
            pynvml.nvmlShutdown()
        return (used if found else 0), "pynvml"
    except Exception:  # noqa: BLE001 - pynvml versions expose different error classes
        pass

    try:
        snapshot = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        used_mib = 0
        for line in snapshot.stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) == 2 and fields[0].isdigit() and int(fields[0]) == pid:
                used_mib += int(fields[1])
        return used_mib * 1024**2, "nvidia-smi"
    except (OSError, ValueError, subprocess.SubprocessError):
        return None, None


class _ProcessGpuMemorySampler:
    """Low-frequency process-memory sampler for CUDA allocations outside torch."""

    def __init__(self, enabled: bool, interval_seconds: float = 2.0):
        self.enabled = enabled
        self.interval_seconds = interval_seconds
        self.peak_bytes: int | None = None
        self.source: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not self.enabled:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            used, source = _sample_process_gpu_memory_bytes()
            if used is not None:
                self.peak_bytes = max(self.peak_bytes or 0, used)
                self.source = source
            self._stop.wait(self.interval_seconds)

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=max(2.0, self.interval_seconds + 1.0))


def _gpu_memory_metrics(
    device: torch.device, process_sampler: _ProcessGpuMemorySampler
) -> dict:
    if device.type != "cuda":
        return {
            "torch_peak_allocated_gb": None,
            "torch_peak_reserved_gb": None,
            "process_peak_used_gb": None,
            "process_memory_source": None,
            "device_total_memory_gb": None,
        }
    gib = float(1024**3)
    try:
        total = int(torch.cuda.get_device_properties(device).total_memory)
    except (RuntimeError, AssertionError):
        total = 0
    return {
        "torch_peak_allocated_gb": float(torch.cuda.max_memory_allocated(device) / gib),
        "torch_peak_reserved_gb": float(torch.cuda.max_memory_reserved(device) / gib),
        "process_peak_used_gb": (
            float(process_sampler.peak_bytes / gib)
            if process_sampler.peak_bytes is not None
            else None
        ),
        "process_memory_source": process_sampler.source,
        "device_total_memory_gb": float(total / gib) if total else None,
    }


def _uses_hetero_loss(args) -> bool:
    return bool(getattr(args, "use_gnll", False) or getattr(args, "use_laplace_nll", False))


def _hetero_output_dim(args) -> int:
    if not _uses_hetero_loss(args) or getattr(args, "use_separate_ud", False):
        return 3
    if str(getattr(args, "hetero_scale", "pixel")).lower() in ("frame", "region"):
        return 3
    return 3 + int(args.num_samples) * 3


def _hetero_recon_criterion(model, *, elementwise: bool = False):
    if getattr(model, "hetero_loss_type", "gaussian") == "laplace":
        if elementwise:
            return lambda pred, target, scale: laplace_nll_loss(
                pred, target, scale, full=True
            )
        return laplace_nll_loss
    return _GNLL_RECON_NONE if elementwise else _GNLL_RECON_CRITERION


def init_hetero_region_scales(model, train_data, device: torch.device) -> None:
    """Eagerly register region log-scales before the optimizer is built."""
    if not getattr(model, "use_gnll", False):
        return
    if getattr(model, "hetero_scale", "pixel") != "region":
        return
    if getattr(model, "log_scales", None) is not None:
        return
    sample = train_data[0]
    lr = sample["lr_target"]
    if not isinstance(lr, torch.Tensor):
        lr = torch.as_tensor(lr)
    if lr.ndim == 3 and lr.shape[-1] in (1, 3):
        height, width = int(lr.shape[0]), int(lr.shape[1])
    elif lr.ndim == 4:
        height, width = int(lr.shape[1]), int(lr.shape[2])
    else:
        raise ValueError(f"Unexpected lr_target shape for region hetero init: {tuple(lr.shape)}")
    dtype = next(model.parameters()).dtype
    model._ensure_region_log_scales(height, width, device, dtype)
_LPIPS_BY_DEVICE: dict[str, lpips.LPIPS] = {}


def _as_device_tensor(x, device: torch.device):
    if isinstance(x, torch.Tensor):
        return x if x.device == device else x.to(device, non_blocking=True)
    if isinstance(x, (int, float)):
        return torch.tensor(x, device=device)
    return x


def _stack_train_loss_scalars(losses: dict[str, torch.Tensor]) -> dict[str, float]:
    order = (
        "recon_loss",
        "trans_loss",
        "variance_reg_loss",
        "variance_smooth_loss",
        "total_loss",
    )
    stacked = torch.stack([losses[k].detach().float() for k in order]).cpu()
    values = stacked.tolist()
    return dict(zip(order, values))


def _train_postfix_from_scalars(scalars: dict[str, float]) -> dict[str, str]:
    postfix = {
        "recon": f"{scalars['recon_loss']:.4f}",
        "trans": f"{scalars['trans_loss']:.4f}",
    }
    if scalars.get("variance_reg_loss", 0.0) > 0.0:
        postfix["var_reg"] = f"{scalars['variance_reg_loss']:.4f}"
    if scalars.get("variance_smooth_loss", 0.0) > 0.0:
        postfix["var_smooth"] = f"{scalars['variance_smooth_loss']:.4f}"
    return postfix


def get_lpips_model(device: torch.device) -> lpips.LPIPS:
    key = str(device)
    if key not in _LPIPS_BY_DEVICE:
        model = lpips.LPIPS(net="vgg").to(device)
        model.eval()
        _LPIPS_BY_DEVICE[key] = model
    return _LPIPS_BY_DEVICE[key]


def _lr_rgb_hwc_unstandardized(train_data) -> np.ndarray:
    """Return the first LR frame as HWC float RGB in [0, 1]."""
    if hasattr(train_data, "get_lr_sample_hwc"):
        lr_hwc = train_data.get_lr_sample_hwc(0).cpu().numpy()
        lr_needs_unstandardize = True
    else:
        lr_any = train_data.get_lr_sample(0).cpu().numpy()
        if lr_any.ndim == 3 and lr_any.shape[0] == 3:
            lr_hwc = np.transpose(lr_any, (1, 2, 0))
        elif lr_any.ndim == 3 and lr_any.shape[2] > 3:
            h, w, c = lr_any.shape
            if c % 3 == 0:
                lr_hwc = lr_any.reshape(h, w, c // 3, 3)[:, :, 0, :]
            else:
                lr_hwc = lr_any[:, :, :3]
        else:
            lr_hwc = lr_any
        lr_needs_unstandardize = False

    if lr_needs_unstandardize:
        lr_std = train_data.get_lr_std(0).cpu().numpy()
        lr_mean = train_data.get_lr_mean(0).cpu().numpy()
        if lr_std.ndim == 1:
            lr_std = lr_std.reshape(1, 1, -1)
        if lr_mean.ndim == 1:
            lr_mean = lr_mean.reshape(1, 1, -1)
        lr_hwc = lr_hwc * lr_std + lr_mean
    return np.clip(lr_hwc, 0.0, 1.0)


def _hr_spatial_hw(hr_coords: torch.Tensor) -> tuple[int, int]:
    if hr_coords.dim() == 4:
        return int(hr_coords.shape[1]), int(hr_coords.shape[2])
    if hr_coords.dim() == 3:
        return int(hr_coords.shape[0]), int(hr_coords.shape[1])
    raise ValueError(f"Unexpected hr_coords shape {tuple(hr_coords.shape)}")


def _forward_hr_output(
    model,
    hr_coords,
    hr_image,
    sample_id,
    device,
    eval_autocast_dtype=None,
    hr_render_tile: int = 0,
    **fwd_kwargs,
):
    """Full-frame HR decode; auto-tiles when the grid exceeds ~2048² (INR path)."""
    kwargs = dict(fwd_kwargs)
    # NIR needs full-frame lr_frames — keep single-shot path.
    if isinstance(model, NIR):
        if eval_autocast_dtype is not None and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=eval_autocast_dtype):
                output, _ = model(
                    hr_coords, sample_id, scale_factor=1, training=False, lr_frames=hr_image, **kwargs
                )
        else:
            output, _ = model(
                hr_coords, sample_id, scale_factor=1, training=False, lr_frames=hr_image, **kwargs
            )
        return output.reshape(hr_image.shape[1], hr_image.shape[2], 3).unsqueeze(0)

    h, w = _hr_spatial_hw(hr_coords)
    tile = resolve_hr_render_tile(h, w, hr_render_tile)
    if tile < max(h, w):
        print(f"HR render tiled at {tile}×{tile} over {h}×{w}", flush=True)
        return render_hr_rgb_tiled(
            model,
            hr_coords,
            sample_id,
            device=device,
            tile=tile,
            eval_autocast_dtype=eval_autocast_dtype,
            **kwargs,
        )

    if eval_autocast_dtype is not None and device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=eval_autocast_dtype):
            output, _ = model(hr_coords, sample_id, scale_factor=1, training=False, **kwargs)
    else:
        output, _ = model(hr_coords, sample_id, scale_factor=1, training=False, **kwargs)
    return output


def _get_hr_eval_mask(dataset) -> torch.Tensor | None:
    if not getattr(dataset, "use_masked_eval", False):
        return None
    getter = getattr(dataset, "get_hr_eval_mask", None)
    if getter is None:
        return None
    return getter()


def _compute_full_frame_metrics(
    pred_tensor: torch.Tensor,
    gt_tensor: torch.Tensor,
    bilinear_tensor: torch.Tensor,
    device: torch.device,
    lpips_fn: lpips.LPIPS,
    eval_mask_hw: torch.Tensor | None = None,
) -> dict:
    if eval_mask_hw is not None:
        return compute_masked_image_metrics(
            pred_tensor,
            gt_tensor,
            bilinear_tensor,
            eval_mask_hw.to(device),
            device=device,
            lpips_fn=lpips_fn,
        )

    # Full-frame PSNR/MSE/MAE; SSIM/LPIPS center-cropped when huge.
    from eval.masked_metrics import _center_crop_bchw

    pred_cpu = pred_tensor.detach().cpu()
    gt_cpu = gt_tensor.detach().cpu()
    bil_cpu = bilinear_tensor.detach().cpu()
    model_psnr = peak_signal_noise_ratio(pred_cpu, gt_cpu, data_range=1.0).item()
    bilinear_psnr = peak_signal_noise_ratio(bil_cpu, gt_cpu, data_range=1.0).item()
    model_mse = F.mse_loss(pred_tensor, gt_tensor).item()
    bilinear_mse = F.mse_loss(bilinear_tensor, gt_tensor).item()
    pred_p, gt_p, bil_p = _center_crop_bchw(
        pred_tensor, gt_tensor, bilinear_tensor, max_side=2048
    )
    model_ssim = ssim(pred_p.detach().cpu(), gt_p.detach().cpu(), data_range=1.0).item()
    bilinear_ssim = ssim(bil_p.detach().cpu(), gt_p.detach().cpu(), data_range=1.0).item()
    model_lpips = lpips_fn((pred_p * 2 - 1), (gt_p * 2 - 1)).item()
    bilinear_lpips = lpips_fn((bil_p * 2 - 1), (gt_p * 2 - 1)).item()
    return {
        "masked": False,
        "valid_fraction": 1.0,
        "test_loss": model_mse,
        "test_psnr": model_psnr,
        "model_psnr": model_psnr,
        "bilinear_psnr": bilinear_psnr,
        "model_ssim": model_ssim,
        "bilinear_ssim": bilinear_ssim,
        "model_lpips": model_lpips,
        "bilinear_lpips": bilinear_lpips,
        "model_mse": model_mse,
        "bilinear_mse": bilinear_mse,
        "model_mae": F.l1_loss(pred_tensor, gt_tensor).item(),
        "bilinear_mae": F.l1_loss(bilinear_tensor, gt_tensor).item(),
    }


def eval_hr_metrics(
    model,
    test_loader,
    device,
    eval_autocast_dtype=None,
    args=None,
    iteration: int | None = None,
) -> dict:
    """Full-frame HR metrics vs GT (and bilinear baseline) for periodic eval."""
    model.eval()
    fwd_kwargs = _step_schedule_kwargs(args, iteration) if args is not None and iteration else {}
    hr_tile = int(getattr(args, "hr_render_tile", 0) or 0) if args is not None else 0
    with torch.no_grad():
        hr_coords = test_loader.get_hr_coordinates().unsqueeze(0).to(device)
        hr_image = test_loader.get_original_hr().unsqueeze(0).to(device)
        sample_id = torch.tensor([0]).to(device)

        output = _forward_hr_output(
            model,
            hr_coords,
            hr_image,
            sample_id,
            device,
            eval_autocast_dtype,
            hr_render_tile=hr_tile,
            **fwd_kwargs,
        )
        output = (
            output * test_loader.get_lr_std(0).to(output.device)
            + test_loader.get_lr_mean(0).to(output.device)
        )

        pred_tensor = output if output.ndim == 4 else output.unsqueeze(0)
        if pred_tensor.shape[-1] == 3:
            pred_tensor = pred_tensor.permute(0, 3, 1, 2)
        gt_tensor = hr_image if hr_image.ndim == 4 else hr_image.unsqueeze(0)
        if gt_tensor.shape[-1] == 3:
            gt_tensor = gt_tensor.permute(0, 3, 1, 2)

        hr_h, hr_w = int(gt_tensor.shape[-2]), int(gt_tensor.shape[-1])
        lr_original = _lr_rgb_hwc_unstandardized(test_loader)
        lr_bilinear = cv2.resize(lr_original, (hr_w, hr_h), interpolation=cv2.INTER_LINEAR)
        bilinear_tensor = (
            torch.from_numpy(np.clip(lr_bilinear, 0.0, 1.0))
            .unsqueeze(0)
            .permute(0, 3, 1, 2)
            .to(device)
        )

        eval_mask_hw = _get_hr_eval_mask(test_loader)
        lpips_fn = get_lpips_model(device)
        metrics = _compute_full_frame_metrics(
            pred_tensor, gt_tensor, bilinear_tensor, device, lpips_fn, eval_mask_hw
        )
        window = _parse_eval_subwindow(args)
        if window is not None:
            r0, c0, side = window
            df = hr_h // int(test_loader.lr_height)
            ys = slice(r0 * df, (r0 + side) * df)
            xs = slice(c0 * df, (c0 + side) * df)
            sub_gt = gt_tensor[..., ys, xs]
            sub_mask = eval_mask_hw[ys, xs] if eval_mask_hw is not None else None
            reference = getattr(test_loader, "sub_reference", None)
            if reference is not None:
                sub_gt = reference["gt_bchw"].to(device)
                sub_mask = reference["mask_hw"]
            sub = _compute_full_frame_metrics(
                pred_tensor[..., ys, xs],
                sub_gt,
                bilinear_tensor[..., ys, xs],
                device,
                lpips_fn,
                sub_mask,
            )
            metrics.update({f"sub_{k}": v for k, v in sub.items()})

    return metrics


def resolve_lr_tile_halo(args) -> int:
    if int(getattr(args, "lr_tile", 0) or 0) <= 0:
        return 0
    raw = int(getattr(args, "lr_tile_halo", -1))
    if raw >= 0:
        return raw
    if str(getattr(args, "lr_degradation", "s2_psf")).lower().strip() == "area":
        return 0
    sigma = max(
        float(getattr(args, f"s2_psf_sigma_{b}_m", 0.0) or 0.0)
        for b in ("b02", "b03", "b04")
    ) * float(getattr(args, "psf_sigma_scale", 1.0) or 1.0)
    return holdout_psf_pad_lr(
        int(getattr(args, "df", 4) or 4),
        sigma_m=sigma,
        truncate=float(getattr(args, "s2_psf_truncate", 4.0)),
        native_gsd_m=float(getattr(args, "s2_native_gsd_m", 10.0)),
    )


def _fourier_matrix_provenance(model, args) -> dict:
    proj = getattr(model, "input_projection", None)
    B = getattr(proj, "B", None)
    if B is None:
        return {}
    import hashlib

    data = B.detach().float().cpu().contiguous().numpy().tobytes()
    return {
        "fourier_matrix": {
            "shape": list(B.shape),
            "scale": float(getattr(proj, "scale", float("nan"))),
            "matrix_seed": int(getattr(args, "fourier_matrix_seed", -1)),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    }


def load_subwindow_reference(args, dataset, device) -> dict | None:
    """HR reference and eval mask for ``--eval_subwindow_lr`` taken from another field.

    The orthophoto is harmonized to the base frame over the whole field, so its
    radiometry depends on the field. Cross-size comparisons therefore score the shared
    window of every field against one parent field's reference.
    """
    ref_dir = str(getattr(args, "eval_reference_s2_dir", "") or "").strip()
    window = _parse_eval_subwindow(args)
    if not ref_dir or window is None:
        return None
    import copy

    own_meta = json.loads((Path(args.s2_dir) / "meta.json").read_text())
    ref_meta = json.loads((Path(ref_dir) / "meta.json").read_text())
    ref_args = copy.copy(args)
    ref_args.s2_dir = ref_dir
    ref_args.num_samples = len(ref_meta["frames"])
    ref_args.lr_height = ref_args.lr_width = 0
    ref = get_dataset(args=ref_args, name=args.dataset, training_device=device)
    r0, c0, side = window
    rr = own_meta["aoi_window"]["row_off"] + r0 - ref_meta["aoi_window"]["row_off"]
    cc = own_meta["aoi_window"]["col_off"] + c0 - ref_meta["aoi_window"]["col_off"]
    df = int(ref.get_original_hr().shape[0]) // int(ref.lr_height)
    if df != int(dataset.get_original_hr().shape[0]) // int(dataset.lr_height):
        raise ValueError("reference field must use the same df")
    ys, xs = slice(rr * df, (rr + side) * df), slice(cc * df, (cc + side) * df)
    gt = ref.get_original_hr()[ys, xs].float()
    mask = _get_hr_eval_mask(ref)
    out = {
        "gt_bchw": gt.permute(2, 0, 1).unsqueeze(0).cpu(),
        "mask_hw": mask[ys, xs].cpu() if mask is not None else None,
        "provenance": {"s2_dir": ref_dir, "window_lr_in_reference": [rr, cc, side]},
    }
    del ref
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def _parse_eval_subwindow(args) -> tuple[int, int, int] | None:
    raw = str(getattr(args, "eval_subwindow_lr", "") or "").strip() if args is not None else ""
    if not raw:
        return None
    parts = [int(p) for p in raw.replace(" ", "").split(",")]
    if len(parts) != 3:
        raise ValueError(f"--eval_subwindow_lr expects ROW,COL,SIZE, got {raw!r}")
    return parts[0], parts[1], parts[2]


def _format_periodic_eval_line(
    iteration: int,
    scalars: dict,
    metrics: dict | None = None,
    *,
    val_loss: float | None = None,
) -> str:
    parts = [f"\nIter {iteration}: Train Loss: {scalars['total_loss']:.6f}"]
    if val_loss is not None:
        parts.append(f"Val(holdout): {val_loss:.6f}")
    if metrics is not None:
        parts.append(
            f"Test Loss: {metrics['test_loss']:.6f}, "
            f"PSNR: {metrics['test_psnr']:.2f} dB (bil {metrics['bilinear_psnr']:.2f}), "
            f"SSIM: {metrics['model_ssim']:.4f} (bil {metrics['bilinear_ssim']:.4f}), "
            f"LPIPS: {metrics['model_lpips']:.4f} (bil {metrics['bilinear_lpips']:.4f})"
        )
    return ", ".join(parts)


def _step_schedule_kwargs(args, iteration: int) -> dict:
    return {"lr_align_args": effective_lr_align_args(args, iteration)}


def _final_schedule_kwargs(args, iteration: int) -> dict:
    del iteration
    return {"lr_align_args": args}


def _resolve_early_stop_max_regression(args) -> float | None:
    raw = float(getattr(args, "early_stop_max_regression", -1.0) or -1.0)
    if raw >= 0.0:
        return raw
    metric = str(getattr(args, "early_stop_metric", "lpips") or "lpips")
    return default_early_stop_regression(metric)


def _resolve_early_stop_ema(args) -> float | None:
    raw = float(getattr(args, "early_stop_ema", -1.0) or -1.0)
    if raw >= 0.0:
        return raw
    metric = str(getattr(args, "early_stop_metric", "lpips") or "lpips")
    return default_early_stop_ema(metric)


def _skip_periodic_hr_eval(args) -> bool:
    """Skip full HR-vs-GT during training (LPIPS/PSNR/MAE). Final eval still runs."""
    if bool(getattr(args, "skip_eval", False)):
        return True
    if bool(getattr(args, "force_hr_eval", False)):
        return False
    if bool(getattr(args, "allow_no_hr", False)):
        return True
    metric = str(getattr(args, "early_stop_metric", "lpips") or "lpips").lower().strip()
    return metric == "holdout_mse"


def _dataset_has_hr_gt(dataset) -> bool:
    return bool(getattr(dataset, "has_hr_gt", True))


def _nan_hr_metrics() -> dict:
    return {
        "masked": False,
        "valid_fraction": 0.0,
        "test_loss": float("nan"),
        "test_psnr": float("nan"),
        "model_psnr": float("nan"),
        "bilinear_psnr": float("nan"),
        "model_ssim": float("nan"),
        "bilinear_ssim": float("nan"),
        "model_lpips": float("nan"),
        "bilinear_lpips": float("nan"),
        "model_mse": float("nan"),
        "bilinear_mse": float("nan"),
        "model_mae": float("nan"),
        "bilinear_mae": float("nan"),
    }


def _holdout_val_interval(args) -> int:
    """How often to score held-out LR pixels (defaults to eval_every, min 1)."""
    every = int(getattr(args, "eval_every", 100) or 0)
    return max(1, every) if every > 0 else 100


def _run_holdout_val_step(
    *,
    model,
    dataset,
    holdout_state: EarlyStopState,
    args,
    device: torch.device,
    iteration: int,
    scalars: dict,
    skip_hr_eval: bool,
) -> tuple[float, dict | None, bool]:
    """Compute holdout val, optional HR metrics; return (val_loss, hr_metrics, should_stop)."""
    val_loss = compute_holdout_val_loss(
        model,
        dataset,
        holdout_state,
        args,
        device,
        step_kwargs=_step_schedule_kwargs(args, iteration),
    )
    hr_metrics = None
    if not skip_hr_eval:
        eval_autocast_dtype = get_eval_autocast_dtype(
            getattr(args, "eval_mixed_precision", "none"), device
        )
        hr_metrics = eval_hr_metrics(
            model, dataset, device, eval_autocast_dtype, args=args, iteration=iteration
        )

    metric = str(getattr(args, "early_stop_metric", "lpips") or "lpips").lower().strip()
    resolved = resolve_early_stop_score(metric, holdout_mse=val_loss, hr_metrics=hr_metrics)
    if resolved is None and metric != "holdout_mse":
        print(
            f"WARN: early_stop_metric={metric} needs HR eval; falling back to holdout_mse."
        )
        resolved = resolve_early_stop_score("holdout_mse", holdout_mse=val_loss, hr_metrics=None)
    stop_metric, stop_score = resolved if resolved is not None else ("holdout_mse", val_loss)

    should_stop = holdout_state.observe(
        iteration,
        stop_score,
        model,
        holdout_mse=val_loss,
    )
    print(
        _format_periodic_eval_line(
            iteration, scalars, hr_metrics, val_loss=val_loss
        )
        + (
            f", stop({stop_metric})={stop_score:.6f}"
            if stop_metric == "holdout_mse"
            else f", stop({stop_metric})={stop_score:.4f}"
        )
    )
    if should_stop:
        print(
            f"Early stop at iter {iteration}: {stop_metric} stalled "
            f"(best {holdout_state.best_score:.6f} @ {holdout_state.best_iter}, "
            f"patience={holdout_state.patience})"
        )
    return val_loss, hr_metrics, should_stop


def _maybe_fixed_spot_metrics(
    pred_aligned: torch.Tensor,
    gt_tensor: torch.Tensor,
    bilinear_aligned: torch.Tensor,
    device: torch.device,
    lpips_fn: lpips.LPIPS,
    args,
    dataset=None,
) -> dict:
    spot_hr_px = int(getattr(args, "eval_spot_hr_px", DEFAULT_SPOT_HR_PX) or 0)
    if spot_hr_px <= 0:
        return {}
    eval_mask_hw = _get_hr_eval_mask(dataset) if dataset is not None else None
    return compute_fixed_spot_metrics(
        pred_aligned,
        gt_tensor,
        bilinear_aligned,
        spot_hr_px=spot_hr_px,
        device=device,
        lpips_fn=lpips_fn,
        eval_mask_hw=eval_mask_hw,
    )


def resolve_hash_resolutions(args) -> tuple[int, int, int, int]:
    """Return (base_h, max_h, base_w, max_w) for the hashgrid.

    If per-axis overrides are given they take priority. Otherwise falls back to
    the isotropic --hash_max_resolution, then to the dataset LR shape, then to 48.
    """
    lr_h = int(getattr(args, "lr_height", 0) or 0)
    lr_w = int(getattr(args, "lr_width", 0) or 0)
    lr = int(getattr(args, "lr_size", 0) or 0)
    # Use lr_size as fallback if dataset shape not yet known
    if lr_h <= 0:
        lr_h = lr
    if lr_w <= 0:
        lr_w = lr

    explicit_max = int(getattr(args, "hash_max_resolution", 0) or 0)
    explicit_max_h = int(getattr(args, "hash_max_resolution_h", 0) or 0)
    explicit_max_w = int(getattr(args, "hash_max_resolution_w", 0) or 0)

    fallback = explicit_max if explicit_max > 0 else 48
    max_h = explicit_max_h if explicit_max_h > 0 else (lr_h if lr_h > 0 else fallback)
    max_w = explicit_max_w if explicit_max_w > 0 else (lr_w if lr_w > 0 else fallback)

    # Finest level defaults to the LR grid; >1 lets the encoder represent
    # detail above LR Nyquist (mult=scale_factor puts the finest level at HR).
    mult = float(getattr(args, "hash_max_resolution_mult", 1.0) or 1.0)
    if mult > 0 and mult != 1.0:
        max_h = max(8, int(round(max_h * mult)))
        max_w = max(8, int(round(max_w * mult)))

    explicit_base = int(getattr(args, "hash_base_resolution", 0) or 0)
    base_h = explicit_base if explicit_base > 0 else max(8, max_h // 4)
    base_w = explicit_base if explicit_base > 0 else max(8, max_w // 4)
    if base_h >= max_h:
        base_h = max(8, max_h // 4)
    if base_w >= max_w:
        base_w = max(8, max_w // 4)
    return base_h, max_h, base_w, max_w


def build_projection_and_decoder(args, device, *, output_dim: int = 3):
    base_h, max_h, base_w, max_w = resolve_hash_resolutions(args)
    rectangular = (max_h != max_w)
    input_projection = get_input_projection(
        args.input_projection,
        2,
        args.projection_dim,
        device,
        args.fourier_scale,
        hash_n_levels=int(getattr(args, "hash_n_levels", 16)),
        hash_n_features_per_level=int(getattr(args, "hash_n_features_per_level", 2)),
        hash_log2_hashmap_size=int(getattr(args, "hash_log2_hashmap_size", 19)),
        hash_base_resolution=base_h,
        hash_base_resolution_h=base_h if rectangular else 0,
        hash_base_resolution_w=base_w if rectangular else 0,
        hash_max_resolution=max(max_h, max_w),
        hash_max_resolution_h=max_h if rectangular else 0,
        hash_max_resolution_w=max_w if rectangular else 0,
        hash_interpolation=str(getattr(args, "hash_interpolation", "smoothstep")),
    )
    matrix_seed = int(getattr(args, "fourier_matrix_seed", -1))
    if hasattr(input_projection, "B") and matrix_seed >= 0:
        gen = torch.Generator().manual_seed(matrix_seed)
        with torch.no_grad():
            input_projection.B.copy_(
                torch.randn(tuple(input_projection.B.shape), generator=gen).to(input_projection.B)
            )
    if input_projection is None:
        decoder_in = 2
    elif hasattr(input_projection, "projection_output_dim"):
        decoder_in = int(input_projection.projection_output_dim)
    else:
        decoder_in = int(args.projection_dim)
    decoder = get_decoder(
        args.model,
        args.network_depth,
        decoder_in,
        args.network_hidden_dim,
        output_dim=output_dim,
        tcnn_mlp_dtype=str(getattr(args, "tcnn_mlp_dtype", "fp16")),
    )
    if args.model == "mlp_tcnn":
        decoder = decoder.to(device)
    return input_projection, decoder


def build_model(args, input_projection, decoder, device):
    from models.inr import get_inr

    model = get_inr(
        input_projection,
        decoder,
        args.num_samples,
        use_gnll=bool(getattr(args, "use_gnll", False)),
        use_laplace_nll=bool(getattr(args, "use_laplace_nll", False)),
        hetero_scale=str(getattr(args, "hetero_scale", "pixel")),
        hetero_region_size=int(getattr(args, "hetero_region_size", 4)),
    )
    model = model.to(device)
    model.lr_degradation = str(getattr(args, "lr_degradation", "s2_psf"))
    if bool(getattr(args, "freeze_affines", False)):
        for parameter in model.affine_params.parameters():
            parameter.requires_grad_(False)
        print("Control: per-frame affine registration frozen at initialization", flush=True)
    if bool(getattr(args, "freeze_radiometry", False)):
        for parameter in model.color_transforms.parameters():
            parameter.requires_grad_(False)
        print("Control: per-frame radiometric transforms frozen at identity", flush=True)
    return model


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


def build_grad_scaler(args, device):
    """Loss scaling for the fp16 tcnn decoder, or None to disable.

    Training is nominally fp32, but tcnn's kernels compute in fp16 internally
    where torch's autocast machinery never sees them. With a mean-reduced loss
    dL/dout falls as 1/pixels-per-step, so past a few million HR rows per step
    it underflows fp16 and the decoder stops receiving gradient entirely.
    """
    mode = str(getattr(args, "grad_scaler", "auto") or "auto").lower()
    if device.type != "cuda" or mode in {"off", "none", "false", "0"}:
        return None
    init_scale = float(getattr(args, "grad_scaler_init_scale", 0.0) or 2.0**15)
    return torch.amp.GradScaler("cuda", init_scale=init_scale)


def resolve_grad_accum_groups(args, batch: int) -> int:
    """How many micro-batches to split a fused tile batch into.

    Full coverage at LR2048 needs k=16 tiles of 512, which is 33.5M HR query
    rows in one step and OOMs inside tinycudann's allocator. Splitting the
    batch keeps peak memory at the k-per-group level while still taking one
    optimizer step over the whole batch.
    """
    requested = int(getattr(args, "grad_accum", 1) or 1)
    if requested <= 1 or batch <= 1:
        return 1
    groups = min(requested, batch)
    while groups > 1 and batch % groups:
        groups -= 1
    return max(1, groups)


def _split_tile_batch(groups: int, coords, lr_target, sample_id, train_mask,
                      gt_dx, gt_dy):
    """Yield (coords, lr_target, sample_id, mask, gt_dx, gt_dy) micro-batches."""
    if groups <= 1:
        yield coords, lr_target, sample_id, train_mask, gt_dx, gt_dy
        return
    per = coords.shape[0] // groups
    for start in range(0, coords.shape[0], per):
        stop = start + per
        yield (
            coords[start:stop],
            lr_target[start:stop],
            sample_id[start:stop],
            None if train_mask is None else train_mask[start:stop],
            gt_dx[start:stop],
            gt_dy[start:stop],
        )


def build_train_dataloader(train_data, args):
    batch_size = max(1, int(getattr(args, "batch_size", 1) or 1))
    return DataLoader(train_data, batch_size=batch_size, shuffle=False)


def _train_forward_losses(
    model,
    coords,
    lr_target,
    sample_id,
    train_mask,
    gt_dx,
    gt_dy,
    full_lr_hw: tuple[int, int],
    args,
    model_kwargs: dict,
    variance_reg: float,
    variance_smooth_reg: float,
    halo: int = 0,
):
    use_gnll_loss = model.use_gnll
    frames_ref = lr_target
    if halo > 0:
        b, h, w, c = lr_target.shape
        frames_ref = lr_target.new_empty(b, h + 2 * halo, w + 2 * halo, c)

    def crop(t):
        return t[:, halo:-halo, halo:-halo] if halo > 0 else t

    if use_gnll_loss:
        output, pred_shifts, pred_variance = model(
            coords, sample_id, lr_frames=frames_ref, **model_kwargs
        )
        output, pred_variance = crop(output), crop(pred_variance)
        if train_mask is not None:
            elem = _hetero_recon_criterion(model, elementwise=True)(
                output, lr_target, pred_variance
            )
            recon_loss = masked_mean(elem, train_mask)
        else:
            recon_loss = _hetero_recon_criterion(model)(output, lr_target, pred_variance)

        variance_reg_loss = torch.zeros((), device=recon_loss.device, dtype=recon_loss.dtype)
        variance_smooth_loss = torch.zeros((), device=recon_loss.device, dtype=recon_loss.dtype)
        if variance_reg > 0.0 or variance_smooth_reg > 0.0:
            if hasattr(model, 'use_separate_ud') and model.use_separate_ud and hasattr(model, 'variances'):
                idx = sample_id.reshape(-1).long()
                log_vars = torch.stack([model.variances[i] for i in idx], dim=0)
                if variance_reg > 0.0:
                    variance_reg_loss = variance_reg * torch.mean(log_vars ** 2)
                if variance_smooth_reg > 0.0:
                    if log_vars.shape[1] > 1 and log_vars.shape[2] > 1:
                        h_diff = log_vars[:, 1:, :, :] - log_vars[:, :-1, :, :]
                        v_diff = log_vars[:, :, 1:, :] - log_vars[:, :, :-1, :]
                        variance_smooth_loss = variance_smooth_reg * (
                            torch.mean(h_diff ** 2) + torch.mean(v_diff ** 2)
                        )
            elif hasattr(model, "log_scales") and variance_reg > 0.0:
                variance_reg_loss = variance_reg * torch.mean(model.log_scales ** 2)
    else:
        output, pred_shifts = model(
            coords, sample_id, lr_frames=frames_ref, **model_kwargs
        )
        output = crop(output)
        if train_mask is not None:
            elem = elementwise_recon(
                getattr(args, "recon_loss", "mse"),
                output,
                lr_target,
                charbonnier_eps=float(getattr(args, "charbonnier_eps", 1e-3)),
                huber_delta=float(getattr(args, "huber_delta", 0.05)),
            )
            recon_loss = masked_mean(elem, train_mask)
        else:
            recon_criterion = resolve_recon_criterion(
                getattr(args, "recon_loss", "mse"),
                charbonnier_eps=float(getattr(args, "charbonnier_eps", 1e-3)),
                huber_delta=float(getattr(args, "huber_delta", 0.05)),
            )
            recon_loss = recon_criterion(output, lr_target)
        variance_reg_loss = torch.zeros((), device=recon_loss.device, dtype=recon_loss.dtype)
        variance_smooth_loss = torch.zeros((), device=recon_loss.device, dtype=recon_loss.dtype)

    if isinstance(model, INR):
        pred_dx, pred_dy = pred_shifts
        lr_h, lr_w = full_lr_hw
        pred_dx_percent = pred_dx / lr_w
        pred_dy_percent = pred_dy / lr_h
        trans_loss = torch.mean(torch.sqrt((pred_dx_percent - gt_dx)**2 + (pred_dy_percent - gt_dy)**2))
    else:
        trans_loss = torch.zeros((), device=recon_loss.device, dtype=recon_loss.dtype)

    total_loss = recon_loss + variance_reg_loss + variance_smooth_loss
    return {
        'recon_loss': recon_loss,
        'trans_loss': trans_loss,
        'variance_reg_loss': variance_reg_loss,
        'variance_smooth_loss': variance_smooth_loss,
        'total_loss': total_loss,
    }


def train_one_iteration(
    model,
    optimizer,
    train_sample,
    device,
    args,
    variance_reg=0.0,
    variance_smooth_reg=0.0,
    holdout_state: EarlyStopState | None = None,
    iteration: int = 0,
    tile_sampler=None,
    cross_sampler=None,
    dataset=None,
    grad_scaler=None,
    instrumentation: dict | None = None,
):
    model.train()

    tile = int(getattr(args, "lr_tile", 0) or 0)
    halo = 0
    if cross_sampler is not None and tile > 0 and dataset is not None:
        halo = resolve_lr_tile_halo(args)
        pairs = cross_sampler.next_pairs()
        masks = holdout_state.train_masks if holdout_state is not None else None
        coords, lr_target, train_mask, sample_id, gt_dx, gt_dy = stack_cross_frame_tiles(
            dataset,
            pairs,
            tile,
            device,
            train_masks=masks,
            halo=halo,
        )
        full_lr_hw = (
            int(getattr(dataset, "lr_height", 0) or lr_target.shape[1]),
            int(getattr(dataset, "lr_width", 0) or lr_target.shape[2]),
        )
    else:
        coords = _as_device_tensor(train_sample["input"], device)
        lr_target = _as_device_tensor(train_sample["lr_target"], device)
        sample_id = _as_device_tensor(train_sample["sample_id"], device)
        if "shifts" in train_sample and "dx_percent" in train_sample["shifts"]:
            gt_dx = _as_device_tensor(train_sample["shifts"]["dx_percent"], device)
            gt_dy = _as_device_tensor(train_sample["shifts"]["dy_percent"], device)
        else:
            gt_dx = torch.zeros(lr_target.shape[0], device=device)
            gt_dy = torch.zeros(lr_target.shape[0], device=device)

        train_mask = None
        if holdout_state is not None:
            train_mask = gather_train_masks(holdout_state.train_masks, sample_id, device)

        full_lr_hw = (int(lr_target.shape[1]), int(lr_target.shape[2]))
        origins = tile_sampler.next_origins() if tile_sampler is not None and tile > 0 else None
        if origins:
            halo = resolve_lr_tile_halo(args)
            coords, lr_target, train_mask, sample_id, gt_dx, gt_dy = stack_lr_hr_tiles(
                coords,
                lr_target,
                origins,
                tile,
                mask=train_mask,
                sample_id=sample_id,
                gt_dx=gt_dx,
                gt_dy=gt_dy,
                halo=halo,
            )
    if instrumentation is not None:
        counts = _training_batch_query_counts(
            coords,
            lr_target,
            int(getattr(args, "supervision_channels", lr_target.shape[-1])),
        )
        instrumentation["step_count"] = int(instrumentation.get("step_count", 0)) + 1
        totals = instrumentation.setdefault("query_totals", {})
        for key, value in counts.items():
            totals[key] = int(totals.get(key, 0)) + value
    model_kwargs = dict(_step_schedule_kwargs(args, iteration))

    optimizer.zero_grad()
    groups = resolve_grad_accum_groups(args, int(coords.shape[0]))
    accumulated: dict[str, torch.Tensor] = {}
    for chunk in _split_tile_batch(
        groups, coords, lr_target, sample_id, train_mask, gt_dx, gt_dy
    ):
        c_coords, c_target, c_sid, c_mask, c_dx, c_dy = chunk
        losses = _train_forward_losses(
            model,
            c_coords,
            c_target,
            c_sid,
            c_mask,
            c_dx,
            c_dy,
            full_lr_hw,
            args,
            model_kwargs,
            variance_reg,
            variance_smooth_reg,
            halo=halo,
        )
        # Each group holds an equal number of tiles, so averaging the group
        # losses reproduces the full-batch loss (up to per-group differences
        # in how many pixels the holdout mask leaves valid).
        loss = losses["total_loss"] / groups
        if grad_scaler is None:
            loss.backward()
        else:
            # The tcnn decoder computes in fp16 while the loss is a mean, so
            # dL/dout falls as 1/pixels-per-step and underflows to zero once a
            # step covers a few million supervision pixels. Scale before
            # backward; the scaler unscales before the step.
            grad_scaler.scale(loss).backward()
        for k, v in losses.items():
            accumulated[k] = accumulated.get(k, 0.0) + v.detach() / groups

    halo_w = float(getattr(args, "halo_weight", 0.0) or 0.0)
    if (
        halo_w > 0.0
        and dataset is not None
        and getattr(dataset, "halo_target", None) is not None
    ):
        batch = sample_halo_batch(
            dataset,
            device=device,
            chunk_lr=int(getattr(args, "halo_chunk_lr", 128) or 128),
        )
        if batch is not None:
            h_coords, h_target, h_mask = batch
            h_sid = torch.zeros(1, device=device, dtype=torch.long)
            out = model(h_coords, h_sid, lr_frames=h_target, **model_kwargs)
            pred = out[0] if isinstance(out, (tuple, list)) else out
            pred = pred[..., :3]
            mean0 = dataset.get_lr_mean(0).to(device)
            std0 = dataset.get_lr_std(0).to(device)
            pred_rgb = pred * std0 + mean0
            h_loss = halo_consistency_loss(
                pred_rgb,
                h_target,
                h_mask,
                recon_loss=str(getattr(args, "recon_loss", "charbonnier") or "charbonnier"),
                charbonnier_eps=float(getattr(args, "charbonnier_eps", 0.01) or 0.01),
                mode=str(getattr(args, "halo_mode", "full") or "full"),
                lf_hr_px=int(getattr(args, "halo_lf_hr_px", 16) or 16),
            )
            scaled = (halo_w * h_loss)
            if grad_scaler is None:
                scaled.backward()
            else:
                grad_scaler.scale(scaled).backward()
            accumulated["halo_loss"] = h_loss.detach()

    if grad_scaler is None:
        optimizer.step()
    else:
        grad_scaler.step(optimizer)
        grad_scaler.update()
    return accumulated


def test_one_epoch(model, test_loader, device, eval_autocast_dtype=None, args=None):
    metrics = eval_hr_metrics(model, test_loader, device, eval_autocast_dtype, args=args)
    return metrics["test_loss"], metrics["test_psnr"]


def optimize_and_evaluate_sample(model, train_data, device, sample_idx, args, output_dir):
    print(f"\n{'='*60}")
    print(f"Optimizing sample {sample_idx + 1}")
    print(f"{'='*60}")
    
    # Record start time for timing metrics
    start_time = time.time()
    
    # Setup optimizer for this sample
    optimizer = build_optimizer(model.parameters(), args)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.iters, eta_min=1e-6)
    grad_scaler = build_grad_scaler(args, device)

    # Training loop for this sample
    iteration = 0
    progress_bar = tqdm(total=args.iters, desc=f"Training Sample {sample_idx + 1}")
    
    # Lists to store training metrics
    psnr_list = []
    ssim_list = []
    lpips_list = []
    recon_loss_list = []
    trans_loss_list = []
    total_loss_list = []
    iteration_list = []
    
    # Track timing for different phases
    training_start_time = time.time()
    
    train_dataloader = build_train_dataloader(train_data, args)
    tile_sampler = build_lr_tile_sampler(train_data, args)
    cross_sampler = build_cross_frame_tile_sampler(train_data, args)
    init_hetero_region_scales(model, train_data, device)
    _reset_peak_memory(device)

    eval_every = int(getattr(args, "eval_every", 100) or 0)
    skip_eval = bool(getattr(args, "skip_eval", False))
    holdout_state = init_early_stop_state(
        num_frames=int(getattr(train_data, "num_samples", len(train_data))),
        lr_height=int(getattr(train_data, "lr_height", 0) or getattr(args, "lr_height", 0)),
        lr_width=int(getattr(train_data, "lr_width", 0) or getattr(args, "lr_width", 0)),
        spatial_holdout=float(getattr(args, "spatial_holdout", 0.0) or 0.0),
        holdout_block=int(getattr(args, "holdout_block", 0) or 0),
        patience=int(getattr(args, "early_stop_patience", 0) or 0),
        min_iters=int(getattr(args, "early_stop_min_iters", 1000) or 0),
        min_delta=float(getattr(args, "early_stop_min_delta", 0.0) or 0.0),
        metric=str(getattr(args, "early_stop_metric", "lpips") or "lpips"),
        max_regression=_resolve_early_stop_max_regression(args),
        ema_alpha=_resolve_early_stop_ema(args),
        device=device,
    )
    holdout_state = _maybe_apply_lr_cloud_masks(holdout_state, train_data, args)
    val_every = _holdout_val_interval(args)
    val_loss_list: list[float] = []
    
    while iteration < args.iters:
        stop_training = False
        # Cross-frame mix: one fused (tile×frame) mini-batch per step; else walk frames.
        step_samples = [None] if cross_sampler is not None else train_dataloader
        for train_sample in step_samples:
            if iteration >= args.iters:
                break
                
            train_losses = train_one_iteration(
                model,
                optimizer,
                train_sample,
                device,
                args,
                variance_reg=args.variance_reg,
                variance_smooth_reg=args.variance_smooth_reg,
                holdout_state=holdout_state,
                iteration=iteration + 1,
                tile_sampler=tile_sampler,
                cross_sampler=cross_sampler,
                dataset=train_data,
                grad_scaler=grad_scaler,
            )
            scheduler.step()
            iteration += 1

            progress_bar.update(1)
            if iteration % LOG_POSTFIX_INTERVAL == 0:
                scalars = _stack_train_loss_scalars(train_losses)
                postfix = _train_postfix_from_scalars(scalars)
                if val_loss_list:
                    postfix["val"] = f"{val_loss_list[-1]:.4f}"
                progress_bar.set_postfix(postfix)
            
            # Holdout val / early stop (even when HR eval is skipped).
            if holdout_state is not None and iteration % val_every == 0:
                scalars = _stack_train_loss_scalars(train_losses)
                val_loss, eval_metrics, should_stop = _run_holdout_val_step(
                    model=model,
                    dataset=train_data,
                    holdout_state=holdout_state,
                    args=args,
                    device=device,
                    iteration=iteration,
                    scalars=scalars,
                    skip_hr_eval=_skip_periodic_hr_eval(args),
                )
                val_loss_list.append(val_loss)
                if eval_metrics is not None:
                    iteration_list.append(iteration)
                    psnr_list.append(eval_metrics["test_psnr"])
                    ssim_list.append(eval_metrics["model_ssim"])
                    lpips_list.append(eval_metrics["model_lpips"])
                    recon_loss_list.append(scalars['recon_loss'])
                    trans_loss_list.append(scalars['trans_loss'])
                    total_loss_list.append(scalars['total_loss'])
                if should_stop:
                    stop_training = True
                    break
            elif eval_every > 0 and not _skip_periodic_hr_eval(args) and iteration % eval_every == 0:
                scalars = _stack_train_loss_scalars(train_losses)
                eval_metrics = eval_hr_metrics(
                    model, train_data, device, args=args, iteration=iteration
                )
                print(_format_periodic_eval_line(iteration, scalars, eval_metrics))

                iteration_list.append(iteration)
                psnr_list.append(eval_metrics["test_psnr"])
                ssim_list.append(eval_metrics["model_ssim"])
                lpips_list.append(eval_metrics["model_lpips"])
                recon_loss_list.append(scalars['recon_loss'])
                trans_loss_list.append(scalars['trans_loss'])
                total_loss_list.append(scalars['total_loss'])

        if stop_training:
            break

    progress_bar.close()

    if holdout_state is not None and holdout_state.restore_best(model):
        print(
            f"Restored best holdout checkpoint from iter {holdout_state.best_iter} "
            f"(val={holdout_state.best_val:.6f})"
        )
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

        output = _forward_hr_output(
            model,
            hr_coords,
            hr_image,
            sample_id,
            device,
            eval_autocast_dtype,
            hr_render_tile=int(getattr(args, "hr_render_tile", 0) or 0),
        )

        # Unstandardize the output
        output = (
            output * train_data.get_lr_std(0).to(output.device)
            + train_data.get_lr_mean(0).to(output.device)
        )
        
        final_test_loss = F.mse_loss(output, hr_image).item()   
        final_psnr = -10 * torch.log10(torch.tensor(final_test_loss)).item()
        
        # Convert tensors to numpy for alignment and color matching
        pred_tensor = torch.from_numpy(output.squeeze().cpu().numpy()).unsqueeze(0).permute(0, 3, 1, 2).to(device)
        gt_tensor = torch.from_numpy(hr_image.squeeze().cpu().numpy()).unsqueeze(0).permute(0, 3, 1, 2).to(device)
        
        # Get LR for bilinear comparison – always work in HWC
        if hasattr(train_data, 'get_lr_sample_hwc'):
            lr_standardized_hwc = train_data.get_lr_sample_hwc(0).cpu().numpy()  # H, W, 3 (standardized)
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
            lr_needs_unstandardize = False

        # Unstandardize only if the LR we fetched is standardized
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

        eval_mask_hw = _get_hr_eval_mask(train_data)
        if eval_mask_hw is not None:
            print(
                f"Using masked HR eval on {train_data.hr_valid_fraction * 100:.1f}% valid GT pixels"
            )

        lpips_fn = get_lpips_model(device)
        frame_metrics = _compute_full_frame_metrics(
            pred_aligned, gt_tensor, bilinear_aligned, device, lpips_fn, eval_mask_hw
        )
        final_test_loss = frame_metrics["test_loss"]
        final_psnr = frame_metrics["test_psnr"]
        model_psnr = frame_metrics["model_psnr"]
        bilinear_psnr = frame_metrics["bilinear_psnr"]
        model_ssim = frame_metrics["model_ssim"]
        bilinear_ssim = frame_metrics["bilinear_ssim"]
        model_lpips = frame_metrics["model_lpips"]
        bilinear_lpips = frame_metrics["bilinear_lpips"]
        model_mse = frame_metrics["model_mse"]
        bilinear_mse = frame_metrics["bilinear_mse"]
        model_mae = frame_metrics["model_mae"]
        bilinear_mae = frame_metrics["bilinear_mae"]

        fixed_spot = _maybe_fixed_spot_metrics(
            pred_aligned, gt_tensor, bilinear_aligned, device, lpips_fn, args, dataset=train_data
        )
        if fixed_spot and not fixed_spot.get("skipped"):
            print(
                f"Fixed spot ({fixed_spot['hr_pixels']}×{fixed_spot['hr_pixels']} HR, center): "
                f"PSNR {fixed_spot['model_psnr']:.2f} dB "
                f"(bil {fixed_spot['bilinear_psnr']:.2f}, Δ {fixed_spot['psnr_improvement']:+.2f}), "
                f"SSIM {fixed_spot['model_ssim']:.4f} "
                f"(bil {fixed_spot['bilinear_ssim']:.4f}), "
                f"LPIPS {fixed_spot['model_lpips']:.4f} "
                f"(bil {fixed_spot['bilinear_lpips']:.4f})"
            )
        elif fixed_spot and fixed_spot.get("skipped"):
            print(
                f"Fixed spot skipped ({fixed_spot.get('skip_reason', 'unavailable')})"
            )
        
        # Convert aligned tensors back to numpy for visualization
        pred_aligned_np = pred_aligned.squeeze(0).permute(1, 2, 0).cpu().numpy()
        bilinear_aligned_np = bilinear_aligned.squeeze(0).permute(1, 2, 0).cpu().numpy()
        gt_np = hr_image.squeeze().cpu().numpy()
        
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

        save_eval_visualizations(
            sample_dir,
            lr_hwc=lr_original,
            bilinear_hwc=bilinear_aligned_np,
            pred_hwc=pred_aligned_np,
            gt_hwc=gt_np,
            image_metrics={
                "model_psnr": model_psnr,
                "bilinear_psnr": bilinear_psnr,
                "model_ssim": model_ssim,
                "bilinear_ssim": bilinear_ssim,
                "model_lpips": model_lpips,
                "bilinear_lpips": bilinear_lpips,
            },
            fixed_spot=(None if not fixed_spot or fixed_spot.get("skipped") else fixed_spot),
            sample_label=f"Sample {sample_idx + 1}",
        )
        
        # Plot training curves if we have data
        if len(psnr_list) > 0:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            ax1.plot(iteration_list, psnr_list, color='blue', linewidth=2, label='PSNR (Test)')
            ax1.set_xlabel('Iteration', fontsize=12)
            ax1.set_ylabel('PSNR (dB)', fontsize=12)
            ax1.set_title(f'Sample {sample_idx + 1} - Training PSNR Evolution', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            ax2.plot(iteration_list, recon_loss_list, color='red', linewidth=2, label='Reconstruction Loss')
            ax2.plot(iteration_list, trans_loss_list, color='green', linewidth=2, label='Transformation Loss')
            ax2.plot(iteration_list, total_loss_list, color='purple', linewidth=2, label='Total Loss')
            ax2.set_xlabel('Iteration', fontsize=12)
            ax2.set_ylabel('Loss', fontsize=12)
            ax2.set_title(f'Sample {sample_idx + 1} - Training Loss Evolution', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            plt.tight_layout()
            plt.savefig(sample_dir / "training_metrics.png", bbox_inches='tight', pad_inches=0.1, dpi=300)
            plt.close()
    
    # Variance maps are opt-in only (--visualize_variance); off by default even with GNLL.
    if model.use_gnll and args.visualize_variance:
        print(f"\nGenerating variance visualizations for sample {sample_idx + 1}...")
        visualize_lr_variance(model, train_data, device, sample_dir, sample_idx)
    
    # Record evaluation end time
    evaluation_end_time = time.time()
    evaluation_time = evaluation_end_time - evaluation_start_time
    total_time = evaluation_end_time - start_time
    
    # Return comprehensive results for this sample
    return {
        'sample_idx': sample_idx,
        'sample_info': {
            'dataset': args.dataset,
            'sample_id': getattr(args, 'sample_id', f'sample_{sample_idx}'),
            'num_lr_frames': len(train_data),
            'iterations': args.iters,
            'model_type': args.model,
            'input_projection': args.input_projection,
            'network_depth': args.network_depth,
            'network_hidden_dim': args.network_hidden_dim,
        },
        'image_metrics': {
            'model_psnr': model_psnr,
            'bilinear_psnr': bilinear_psnr,
            'psnr_improvement': model_psnr - bilinear_psnr,
            'model_ssim': model_ssim,
            'bilinear_ssim': bilinear_ssim,
            'ssim_improvement': model_ssim - bilinear_ssim,
            'model_lpips': model_lpips,
            'bilinear_lpips': bilinear_lpips,
            'lpips_improvement': bilinear_lpips - model_lpips,
            'model_mse': model_mse,
            'bilinear_mse': bilinear_mse,
            'mse_improvement': bilinear_mse - model_mse,
            'model_mae': model_mae,
            'bilinear_mae': bilinear_mae,
            'mae_improvement': bilinear_mae - model_mae,
            'fixed_spot': fixed_spot,
        },
        'training_metrics': {
            'final_test_loss': final_test_loss,
            'final_test_psnr': final_psnr,
            'iterations': iteration_list,
            'psnr': psnr_list,
            'recon_loss': recon_loss_list,
            'trans_loss': trans_loss_list,
            'total_loss': total_loss_list,
            'convergence_iteration': len(psnr_list),  # Number of evaluation points
            'final_recon_loss': recon_loss_list[-1] if recon_loss_list else None,
            'final_trans_loss': trans_loss_list[-1] if trans_loss_list else None,
            'final_total_loss': total_loss_list[-1] if total_loss_list else None,
        },
        'timing_metrics': {
            'training_time_seconds': training_time,
            'training_time_minutes': training_time / 60.0,
            'evaluation_time_seconds': evaluation_time,
            'evaluation_time_minutes': evaluation_time / 60.0,
            'total_time_seconds': total_time,
            'total_time_minutes': total_time / 60.0,
            'time_per_iteration_seconds': training_time / args.iters if args.iters > 0 else 0,
        },
        'image_dimensions': {
            'hr_height': hr_image.shape[1],
            'hr_width': hr_image.shape[2],
            'lr_height': lr_original.shape[0],
            'lr_width': lr_original.shape[1],
            'scale_factor': hr_image.shape[1] / lr_original.shape[0],
        }
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
        if hasattr(train_data, 'num_samples'):
            num_samples = train_data.num_samples
        elif hasattr(train_data, 'lr_paths'):
            num_samples = len(train_data.lr_paths)
        else:
            print("Warning: Cannot determine number of LR samples. Skipping variance visualization.")
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
            output, _, variance = model(hr_coords, sample_id_tensor, scale_factor=1, training=False, lr_frames=hr_image)

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
                # Some datasets may not index per-sample; fall back to 0
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
                if hasattr(std_i, 'cpu'):
                    std_i = std_i.cpu().numpy()
                if hasattr(mean_i, 'cpu'):
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
                    if hasattr(train_data, 'get_lr_sample_hwc'):
                        lr_sample = train_data.get_lr_sample_hwc(i)
                        if hasattr(lr_sample, 'cpu'):
                            lr_np = lr_sample.cpu().numpy()
                        else:
                            lr_np = np.array(lr_sample)
                    elif hasattr(train_data, 'get_lr_sample'):
                        lr_sample = train_data.get_lr_sample(i)
                        if hasattr(lr_sample, 'cpu'):
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
            axes[0, 0].set_title(f'Ground Truth HR', fontsize=12, fontweight='bold')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(output_np)
            axes[0, 1].set_title(f'Model Output (Sample {i})', fontsize=12, fontweight='bold')
            axes[0, 1].axis('off')
            
            # Row 2: Standard deviation analysis
            # Raw std map - upsample to match output size if needed
            std_display = std_np.copy()
            if std_np.shape[:2] != output_np.shape[:2]:
                # Std is at different resolution, upsample to match output
                if std_np.ndim == 3:
                    # Resize each channel
                    std_display = np.zeros((output_np.shape[0], output_np.shape[1], std_np.shape[2]))
                    for c in range(std_np.shape[2]):
                        std_display[:, :, c] = cv2.resize(
                            std_np[:, :, c], 
                            (output_np.shape[1], output_np.shape[0]), 
                            interpolation=cv2.INTER_LINEAR
                        )
                else:
                    std_display = cv2.resize(
                        std_np, 
                        (output_np.shape[1], output_np.shape[0]), 
                        interpolation=cv2.INTER_LINEAR
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
                global_max_deviation = max(abs(global_vmax - 1), abs(global_vmin - 1), max_deviation)
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
                lr_vis_grid = cv2.resize(lr_np_grid, (output_np.shape[1], output_np.shape[0]), interpolation=cv2.INTER_LINEAR)
                # Brighten LR image for better visibility (scale + shift)
                lr_vis_grid = lr_vis_grid * 1.2 + 0.15
                lr_vis_grid = np.clip(lr_vis_grid, 0.0, 1.0)
                lr_samples_for_grid.append(lr_vis_grid)
            else:
                lr_samples_for_grid.append(None)
            
            # Display std map with Blues colormap
            im_var = axes[1, 0].imshow(std_map, cmap='Blues', vmin=vmin, vmax=vmax)
            axes[1, 0].set_title(f'Standard Deviation Map (Sample {i})', fontsize=12, fontweight='bold')
            axes[1, 0].axis('off')
            cbar = plt.colorbar(im_var, ax=axes[1, 0], fraction=0.046, pad=0.04)
            cbar.set_label('Standard Deviation', rotation=270, labelpad=15)
            
            # Show the LR sample alongside
            if lr_np is not None:
                # Resize LR to HR size for visualization
                if lr_np.ndim == 2:
                    lr_np = np.repeat(lr_np[..., None], 3, axis=-1)
                elif lr_np.shape[-1] == 1:
                    lr_np = np.repeat(lr_np, 3, axis=-1)
                lr_vis = cv2.resize(lr_np, (output_np.shape[1], output_np.shape[0]), interpolation=cv2.INTER_LINEAR)
                # Brighten LR image for better visibility (scale + shift)
                lr_vis = lr_vis * 1.2 + 0.15
                lr_vis = np.clip(lr_vis, 0.0, 1.0)
                axes[1, 1].imshow(lr_vis)
                axes[1, 1].set_title(f'LR Sample (Sample {i})', fontsize=12, fontweight='bold')
                axes[1, 1].axis('off')
            else:
                # Fallback: show std stats if LR not available
                axes[1, 1].text(0.1, 0.8, f'Standard Deviation Statistics:', fontsize=12, fontweight='bold', transform=axes[1, 1].transAxes)
                axes[1, 1].text(0.1, 0.7, f'Mean: {np.mean(std_np):.6f}', fontsize=10, transform=axes[1, 1].transAxes)
                axes[1, 1].text(0.1, 0.6, f'Std: {np.std(std_np):.6f}', fontsize=10, transform=axes[1, 1].transAxes)
                axes[1, 1].text(0.1, 0.5, f'Min: {np.min(std_np):.6f}', fontsize=10, transform=axes[1, 1].transAxes)
                axes[1, 1].text(0.1, 0.4, f'Max: {np.max(std_np):.6f}', fontsize=10, transform=axes[1, 1].transAxes)
                axes[1, 1].text(0.1, 0.3, f'75th percentile: {np.percentile(std_np, 75):.6f}', fontsize=10, transform=axes[1, 1].transAxes)
                axes[1, 1].text(0.1, 0.2, f'95th percentile: {np.percentile(std_np, 95):.6f}', fontsize=10, transform=axes[1, 1].transAxes)
                axes[1, 1].set_xlim(0, 1)
                axes[1, 1].set_ylim(0, 1)
                axes[1, 1].axis('off')
            
            plt.tight_layout(pad=2.0)
            
            # Save individual std visualization
            variance_path = variance_dir / f"sample_{i:03d}_variance_analysis.png"
            plt.savefig(variance_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
            plt.close()
            
            # Save individual std map as an image (for later 2x8 grid visualization)
            fig_var_only = plt.figure(figsize=(8, 8))
            ax_var_only = fig_var_only.add_subplot(111)
            im_var_only = ax_var_only.imshow(std_map, cmap='Blues', vmin=vmin, vmax=vmax)
            ax_var_only.axis('off')
            cbar_var_only = plt.colorbar(im_var_only, ax=ax_var_only, fraction=0.046, pad=0.04)
            cbar_var_only.set_label('Standard Deviation', rotation=270, labelpad=15)
            plt.tight_layout(pad=0)
            variance_map_path = variance_dir / f"sample_{i:03d}_variance_map.png"
            plt.savefig(variance_map_path, bbox_inches='tight', pad_inches=0, dpi=300)
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
                lr_vis_resized = cv2.resize(lr_np_vis, (output_np.shape[1], output_np.shape[0]), interpolation=cv2.INTER_LINEAR)
                # Brighten LR image for better visibility (scale + shift)
                lr_vis_resized = lr_vis_resized * 1.2 + 0.15
                lr_vis_resized = np.clip(lr_vis_resized, 0.0, 1.0)
                ax_lr_only.imshow(lr_vis_resized)
                ax_lr_only.axis('off')
                plt.tight_layout(pad=0)
                lr_sample_path = variance_dir / f"sample_{i:03d}_lr_sample.png"
                plt.savefig(lr_sample_path, bbox_inches='tight', pad_inches=0, dpi=300)
                plt.close(fig_lr_only)
            
            # Save individual std map as numpy array (also save variance for reference)
            np.save(variance_dir / f"sample_{i:03d}_std.npy", std_np)
            np.save(variance_dir / f"sample_{i:03d}_variance.npy", variance_np)
            np.save(variance_dir / f"sample_{i:03d}_output.npy", output_np)
        
        # Create a summary visualization showing all variance maps side by side
        create_variance_summary(train_data, variance_dir, device)
        
        # Create 2x8 grid: top row = LR samples, bottom row = std maps
        if len(lr_samples_for_grid) >= 8 and len(variance_maps_for_grid) >= 8:
            create_lr_variance_grid(lr_samples_for_grid[:8], variance_maps_for_grid[:8], 
                                   global_vmin, global_vmax, variance_dir)
        
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
    
    fig, axes = plt.subplots(2, 8, figsize=(24, 8))  # Increased height from 6 to 8 for less compact y direction
    
    # Top row: LR samples
    for i in range(8):
        ax = axes[0, i]
        if lr_samples[i] is not None:
            ax.imshow(lr_samples[i])
        else:
            ax.text(0.5, 0.5, f'LR {i}', ha='center', va='center', transform=ax.transAxes)
        # Get image bounds for border
        if lr_samples[i] is not None:
            h, w = lr_samples[i].shape[:2]
            rect = Rectangle((-0.5, -0.5), w, h, 
                            fill=False, edgecolor='gray', linewidth=0.5, clip_on=False)
            ax.add_patch(rect)
        ax.axis('off')
    
    # Bottom row: Standard deviation maps
    for i in range(8):
        ax = axes[1, i]
        im = ax.imshow(variance_maps[i], cmap='Blues', vmin=vmin, vmax=vmax)
        # Get image bounds for border
        h, w = variance_maps[i].shape[:2]
        rect = Rectangle((-0.5, -0.5), w, h, 
                        fill=False, edgecolor='gray', linewidth=0.5, clip_on=False)
        ax.add_patch(rect)
        ax.axis('off')
    
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
    cbar.set_label('Standard Deviation (1 = neutral)', rotation=270, labelpad=20)
    grid_path = variance_dir / "lr_variance_grid_2x8.png"
    plt.savefig(grid_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
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
    if hasattr(train_data, 'num_samples'):
        num_samples = train_data.num_samples
    elif hasattr(train_data, 'lr_paths'):
        num_samples = len(train_data.lr_paths)
    else:
        num_samples = "Unknown"
    
    with open(summary_path, 'w') as f:
        f.write("Standard Deviation Analysis Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Number of LR samples: {num_samples}\n")
        f.write(f"Each sample has been processed individually to show model uncertainty.\n")
        f.write(f"High standard deviation regions indicate where the model is less confident.\n")
        f.write(f"Standard deviation is computed as sqrt(variance) for easier interpretation.\n")
        f.write(f"Check individual sample_XXX_variance_analysis.png files for detailed analysis.\n")
    
    print(f"Variance summary saved to {summary_path}")


def get_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal Satellite Super-Resolution Training")
    
    # Essential parameters only
    parser.add_argument("--dataset", type=str, default="s2",
                       help="Dataset name or city id (s2 / bergen / kristiansand) used in output paths.")
    parser.add_argument(
        "--s2-dir",
        dest="s2_dir",
        type=str,
        default=None,
        help="Sentinel-2 revisit directory with meta.json (default: data/s2_revisits/bergen).",
    )
    parser.add_argument(
        "--hr-path",
        dest="hr_path",
        type=str,
        default=None,
        help="NIB HR ortho GeoTIFF. Default: data/nib_resampled/<city>/*_{10/df}m.tif",
    )
    parser.add_argument(
        "--no_mask_lr_clouds",
        dest="mask_lr_clouds",
        action="store_false",
        help=(
            "Do not AND per-frame SCL/cloud masks into the reconstruction loss. "
            "Default is on when a frame cloud_mask exists."
        ),
    )
    parser.set_defaults(mask_lr_clouds=True)
    parser.add_argument(
        "--lr_stats_pixels",
        choices=["clear", "all"],
        default="clear",
        help=(
            "Pixels used for each LR frame's per-channel standardization mean/std: "
            "'clear' excludes cloud-mask and nodata pixels (default); 'all' uses the "
            "whole frame."
        ),
    )
    parser.add_argument(
        "--halo_sr",
        type=str,
        default=None,
        help="Frozen neighbor sr_pred.tif for train-time halo consistency.",
    )
    parser.add_argument(
        "--halo_lr_px",
        type=int,
        default=32,
        help="Halo width in LR pixels (default 32 ≈ 320 m).",
    )
    parser.add_argument(
        "--halo_sides",
        type=str,
        default="west",
        help="Comma-separated halo sides: west, north, or west,north.",
    )
    parser.add_argument(
        "--halo_weight",
        type=float,
        default=1.0,
        help="Weight on halo consistency vs S2 recon (used only if --halo_sr is set).",
    )
    parser.add_argument(
        "--halo_chunk_lr",
        type=int,
        default=128,
        help="Random LR-sized chunk along the halo strip each step.",
    )
    parser.add_argument(
        "--halo_mode",
        type=str,
        default="full",
        choices=["full", "lowfreq", "mean"],
        help=(
            "Halo match: full=per-pixel (legacy); lowfreq=avg-pool then match; "
            "mean=RGB mean over the overlap (DC/colour only)."
        ),
    )
    parser.add_argument(
        "--halo_lf_hr_px",
        type=int,
        default=16,
        help="Pool size in HR pixels for --halo_mode lowfreq (default 16 = 4 LR px).",
    )
    parser.add_argument(
        "--allow_no_hr",
        action="store_true",
        help=(
            "Production SR without HR GT: do not load NIB even if a city package "
            "exists. Training + GeoTIFF export use Sentinel-2 revisits only."
        ),
    )
    parser.add_argument(
        "--hr-gsd-m",
        dest="hr_gsd_m",
        type=float,
        default=0.0,
        help="HR GSD in meters (default: --s2-native-gsd-m / --df, e.g. 2.5 for df=4).",
    )
    parser.add_argument(
        "--no_hr_harmonize",
        action="store_true",
        help="Disable NIB->S2 HR radiometric harmonization (SEN2NAIP per-band histogram matching) used for eval GT.",
    )
    parser.add_argument(
        "--no_hr_spatial_align",
        action="store_true",
        help="Disable NIB->S2 HR spatial shift from eval/spatial_alignment.json (eval GT only).",
    )
    parser.add_argument(
        "--spatial_alignment_path",
        type=str,
        default=None,
        help="JSON with per-city hr_shift_hr_px (default: eval/spatial_alignment.json).",
    )
    parser.add_argument("--sample_id", default="sample")
    parser.add_argument("--df", type=int, default=4, help="Downsampling factor, or upsampling factor for the data")
    parser.add_argument("--scale_factor", type=float, default=4, help="scale factor for the input training grid")
    parser.add_argument("--num_samples", type=int, default=16)
    
    # Model parameters
    parser.add_argument("--model", type=str, default="mlp_tcnn",
                       choices=["mlp", "mlp_tcnn", "nir"])
    parser.add_argument("--network_depth", type=int, default=4)
    parser.add_argument("--network_hidden_dim", type=int, default=256)
    parser.add_argument("--projection_dim", type=int, default=256)
    parser.add_argument("--input_projection", type=str, default="hashgrid_tcnn",
                       help="fourier, fourier_N, hashgrid, hashgrid_tcnn, none")
    parser.add_argument("--fourier_scale", type=float, default=10.0)
    parser.add_argument("--use_gnll", action="store_true")
    parser.add_argument(
        "--use_laplace_nll",
        action="store_true",
        help="Heteroscedastic Laplace NLL (robust L1 + learned scale). Mutually exclusive with --use_gnll.",
    )
    parser.add_argument(
        "--hetero_scale",
        type=str,
        default="pixel",
        choices=["pixel", "frame", "region"],
        help="Hetero uncertainty: per-pixel maps, one scalar per LR frame, or tiled regions.",
    )
    parser.add_argument(
        "--hetero_region_size",
        type=int,
        default=4,
        help="Tile size (LR pixels) for --hetero_scale region (default: 4 → 4x4 regions).",
    )
    parser.add_argument("--use_separate_ud", action="store_true", help="Use separate UD parameters for each sample (default: False)")
    parser.add_argument("--variance_reg", type=float, default=0.0, help="L2 regularization strength for log-variances (default: 0.0)")
    parser.add_argument("--variance_smooth_reg", type=float, default=0.0, help="Smoothness regularization strength for variance maps (default: 0.0)")
    parser.add_argument("--visualize_variance", action="store_true", help="Opt in: save GNLL variance maps (slow; off by default)")
    parser.add_argument("--no_variance_viz", action="store_true", help="Deprecated alias: variance viz is already off unless --visualize_variance is set")
    parser.add_argument("--no_base_frame", action="store_true", help="Disable base frame (default: use_base_frame=True)")
    parser.add_argument(
        "--max_base_cloud_frac",
        type=float,
        default=0.02,
        help=(
            "Frozen base frame: closest to center/NIB date among frames whose "
            "cell SCL cloud fraction is <= this (default 0.02). If none qualify, "
            "use the clearest frame. Affine/radiometry identity is that frame."
        ),
    )
    parser.add_argument(
        "--force_base_date",
        type=str,
        default=None,
        help=(
            "If this YYYY-MM-DD is in the cell stack, freeze it as identity "
            "even when cell SCL cloud exceeds --max_base_cloud_frac. Missing "
            "date falls back to the default identity rule."
        ),
    )
    parser.add_argument("--no_direct_param_T", action="store_true", help="Disable direct parameter T (default: use_direct_param_T=True)")
    parser.add_argument(
        "--freeze_affines",
        action="store_true",
        help="Control ablation: keep all per-frame geometric transforms at identity.",
    )
    parser.add_argument(
        "--freeze_radiometry",
        action="store_true",
        help="Control ablation: keep all per-frame band gain/bias transforms at identity.",
    )
    
    parser.add_argument("--lr_size", type=int, default=0)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument(
        "--lr_degradation",
        type=str,
        default="s2_psf_m",
        choices=["area", "s2_psf", "s2_psf_m"],
        help="HR→LR operator during training (default: s2_psf_m).",
    )
    parser.add_argument(
        "--recon_loss",
        type=str,
        default="mae",
        choices=["mse", "mae", "charbonnier", "huber"],
        help="LR reconstruction loss (default: mae). mae is more robust to hashgrid ringing than mse.",
    )
    parser.add_argument(
        "--charbonnier_eps",
        type=float,
        default=1e-3,
        help="Epsilon for Charbonnier loss (default: 1e-3).",
    )
    parser.add_argument(
        "--huber_delta",
        type=float,
        default=0.05,
        help="Quadratic/linear transition for Huber loss in normalized reflectance units (default: 0.05).",
    )
    parser.add_argument("--s2-native-gsd-m", dest="s2_native_gsd_m", type=float, default=10.0)
    parser.add_argument("--s2-psf-truncate", dest="s2_psf_truncate", type=float, default=4.0)
    parser.add_argument("--s2-psf-sigma-b02-m", dest="s2_psf_sigma_b02_m", type=float, default=2.8)
    parser.add_argument("--s2-psf-sigma-b03-m", dest="s2_psf_sigma_b03_m", type=float, default=3.25)
    parser.add_argument("--s2-psf-sigma-b04-m", dest="s2_psf_sigma_b04_m", type=float, default=4.2)
    parser.add_argument("--s2-psf-sigma-b08-m", dest="s2_psf_sigma_b08_m", type=float, default=3.5)
    parser.add_argument("--hash_max_resolution", type=int, default=0)
    parser.add_argument("--hash_max_resolution_h", type=int, default=0,
                        help="Hashgrid max resolution for the height (row) axis. "
                             "0 = auto from dataset LR height.")
    parser.add_argument("--hash_max_resolution_w", type=int, default=0,
                        help="Hashgrid max resolution for the width (col) axis. "
                             "0 = auto from dataset LR width.")
    parser.add_argument("--hash_base_resolution", type=int, default=0,
                        help="Hashgrid coarsest level resolution. 0 = auto (max/4).")
    parser.add_argument("--hash_max_resolution_mult", type=float, default=1.0,
                        help="Scale the auto-derived max resolution. 1.0 (default) caps the "
                             "finest level at the LR grid; use the scale factor (e.g. 4) to "
                             "let the encoder represent detail up to the HR grid.")
    parser.add_argument("--hash_n_levels", type=int, default=16)
    parser.add_argument("--hash_n_features_per_level", type=int, default=2)
    parser.add_argument("--hash_log2_hashmap_size", type=int, default=21)
    parser.add_argument(
        "--hash_interpolation",
        type=str,
        default="linear",
        choices=["smoothstep", "linear"],
        help=(
            "Hash grid vertex interpolation (default: linear). "
            "smoothstep applies NGP Appendix A half-voxel per-level offset; "
            "linear matches the paper's main multilinear default."
        ),
    )
    parser.add_argument(
        "--schedule_horizon_iters",
        type=int,
        default=3000,
        help="Horizon for PSF curriculum / sigma schedules (default: 3000).",
    )
    parser.add_argument(
        "--schedule_boundaries",
        type=str,
        default="0.27,0.53",
        help="Step-schedule phase boundaries as progress fractions (default: 0.27,0.53).",
    )
    parser.add_argument(
        "--psf_curriculum",
        type=str,
        default="none",
        choices=["none", "step"],
        help="PSF curriculum: area -> s2_psf -> lr_degradation target (default: none).",
    )
    parser.add_argument(
        "--psf_sigma_schedule",
        type=str,
        default="none",
        choices=["none", "linear", "step"],
        help="Ramp s2_psf_m sigma scale from psf_sigma_min_scale to 1.0 (default: none).",
    )
    parser.add_argument(
        "--psf_sigma_min_scale",
        type=float,
        default=0.0,
        help="Starting sigma scale for psf_sigma_schedule (0 ≈ area pool).",
    )
    parser.add_argument("--tcnn_mlp_dtype", type=str, default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--supervision_channels", type=int, default=3)
    parser.add_argument("--eval_every", type=int, default=200,
                        help="Periodic HR eval / holdout-val interval in iterations (default: 200).")
    parser.add_argument(
        "--hr_render_tile",
        type=int,
        default=0,
        help=(
            "HR decode tile side in pixels for final/periodic eval (0=auto: full if ≤2048², "
            "else 2048). Needed for large AOIs like LR2048→HR8192."
        ),
    )
    parser.add_argument(
        "--query_gsd_m",
        type=str,
        default="",
        help=(
            "Comma-separated extra output GSDs in metres to decode after training "
            "(e.g. '5,1'). The field is still fitted at --df; this is query-time scoring, "
            "not a 1 m training recipe."
        ),
    )
    parser.add_argument("--no_multiband_diagnostics", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--skip_artifacts", action="store_true")
    parser.add_argument(
        "--no_qgis_export",
        action="store_true",
        help="Skip writing georeferenced GeoTIFFs (hr_gt / sr_pred / s2_bilinear) for QGIS.",
    )
    parser.add_argument(
        "--spatial_holdout",
        type=float,
        default=0.1,
        help=(
            "Fraction of LR pixel blocks held out of the train loss and used for "
            "validation / early stopping (0=disable). Independent per frame."
        ),
    )
    parser.add_argument(
        "--holdout_block",
        type=int,
        default=0,
        help=(
            "Side length in LR pixels of each held-out block. "
            "0=auto (scales with AOI; ~8 on LR512, ~32 on LR2048)."
        ),
    )
    parser.add_argument(
        "--holdout_patch_batch",
        type=int,
        default=0,
        help=(
            "Holdout patches per val forward (0=auto from holdout_block). "
            "Larger = fewer launches, more VRAM during val."
        ),
    )
    parser.add_argument(
        "--early_stop_metric",
        type=str,
        default="lpips",
        choices=["holdout_mse", "lpips", "psnr", "mae"],
        help=(
            "Metric for checkpoint restore / early stopping when spatial_holdout>0. "
            "lpips/mae/psnr use full-frame HR GT (needs periodic eval); holdout_mse uses "
            "held-out LR blocks only."
        ),
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=3,
        help=(
            "Stop after this many val checks without improvement on early_stop_metric "
            "(0=never early-stop; still logs val when spatial_holdout>0)."
        ),
    )
    parser.add_argument(
        "--early_stop_max_regression",
        type=float,
        default=-1.0,
        help=(
            "Force stop when the (possibly EMA-smoothed) metric regresses more than this "
            "from the best score (-1 = auto: 0.003 LPIPS, 0.01 holdout_mse, …)."
        ),
    )
    parser.add_argument(
        "--early_stop_ema",
        type=float,
        default=-1.0,
        help=(
            "EMA alpha on the stop metric before patience/regression "
            "(-1 = auto: 0.4 for holdout_mse, 0 otherwise; 0 disables)."
        ),
    )
    parser.add_argument(
        "--force_hr_eval",
        action="store_true",
        help=(
            "Log full-frame HR metrics every eval_every even when early_stop_metric "
            "is holdout_mse (for stopping-rule validation; not for production)."
        ),
    )
    parser.add_argument(
        "--lr_tile_halo",
        type=int,
        default=-1,
        help=(
            "LR px of blur context added around each training window (--lr_tile > 0); "
            "the prediction is cropped back to the window before the loss so window "
            "edges match the full-field forward. -1 = PSF support (default); 0 = none."
        ),
    )
    parser.add_argument(
        "--fourier_matrix_seed",
        type=int,
        default=-1,
        help=(
            "Draw the Fourier frequency matrix from a dedicated generator with this seed, "
            "so it is identical across field sizes; -1 keeps the global-RNG draw."
        ),
    )
    parser.add_argument(
        "--eval_reference_s2_dir",
        type=str,
        default="",
        help=(
            "Score --eval_subwindow_lr against the HR reference (and eval mask) of this "
            "field instead of this run's own reference; offsets come from both meta.json."
        ),
    )
    parser.add_argument(
        "--eval_subwindow_lr",
        type=str,
        default="",
        help=(
            "ROW,COL,SIZE in LR pixels of this field. Periodic HR evaluation also scores "
            "this window (history keys sub_*), e.g. a footprint shared across field sizes."
        ),
    )
    parser.add_argument(
        "--early_stop_min_iters",
        type=int,
        default=1000,
        help="Do not early-stop before this many iterations (default: 1000).",
    )
    parser.add_argument(
        "--early_stop_min_delta",
        type=float,
        default=0.0005,
        help="Minimum improvement in the stop metric to reset patience (default: 0.0005 for LPIPS).",
    )
    parser.add_argument(
        "--eval-spot-hr-px",
        dest="eval_spot_hr_px",
        type=int,
        default=DEFAULT_SPOT_HR_PX,
        help="Center HR patch size for cross-LR-size spot metrics (0=disable).",
    )

    # Training parameters
    parser.add_argument("--seed", type=int, default=6)
    parser.add_argument(
        "--iters",
        type=int,
        default=3000,
        help="Max training iterations (default: 3000). LPIPS early stop usually fires earlier.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Training DataLoader batch size (frames per optimizer step). Default: 1.",
    )
    parser.add_argument(
        "--lr_tile",
        type=int,
        default=0,
        help="LR spatial tile size for training (0=full field). E.g. 64 → HR 256 at df=4.",
    )
    parser.add_argument(
        "--lr_tiles_per_step",
        type=int,
        default=1,
        help="How many LR tiles per optimizer step (0=all complete tiles, shuffled). Default: 1.",
    )
    parser.add_argument(
        "--lr_tile_mix",
        type=str,
        default="within",
        choices=["within", "cross_epoch", "cross_iid", "cross_same_tile"],
        help=(
            "How fused tiles pick frames: within=current DataLoader frame; "
            "cross_epoch=shuffle spatial index + random frame IDs then mini-batch; "
            "cross_iid=k independent (tile, frame) draws each step; "
            "cross_same_tile=one spatial tile × k frames per step."
        ),
    )
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "muon"])
    parser.add_argument("--muon_momentum", type=float, default=0.95)
    parser.add_argument("--no_muon_nesterov", action="store_true")
    parser.add_argument("--muon_ns_steps", type=int, default=5)
    parser.add_argument("--muon_eps", type=float, default=1e-8)
    parser.add_argument("--device", type=str, default="7", help="CUDA device number (e.g., '0', '1') or 'cpu' for CPU")
    parser.add_argument(
        "--dataset_device",
        type=str,
        default="auto",
        help="Device for cached LR frames and coord grids (default: auto = same as --device).",
    )
    parser.add_argument(
        "--grad_scaler", type=str, default="auto",
        help=(
            "Loss scaling for the fp16 tcnn decoder: 'auto' (on for CUDA) or "
            "'off'. Without it, dL/dout underflows fp16 once a step covers a "
            "few million HR rows and the decoder stops training."
        ),
    )
    parser.add_argument(
        "--grad_scaler_init_scale", type=float, default=0.0,
        help="Initial loss scale (0 = 2**15).",
    )
    parser.add_argument(
        "--grad_accum", type=int, default=1,
        help=(
            "Split each fused tile batch into this many micro-batches before "
            "stepping. Lets high --lr_tiles_per_step run without the single "
            "huge forward that OOMs tinycudann (k8/k16 at LR2048 tile 512)."
        ),
    )
    parser.add_argument("--eval_mixed_precision", type=str, default="none",
                        choices=["none", "auto", "fp16", "bfloat16"],
                        help="Use mixed precision (FP16/BF16) for evaluation only; PSNR/metrics reported in this mode. none=float32.")
    return parser


def main():
    args = get_argparser().parse_args()
    if getattr(args, "halo_sr", None):
        west, north = parse_halo_sides(getattr(args, "halo_sides", "west"), int(getattr(args, "halo_lr_px", 32) or 0))
        args.halo_lr_px_west = west
        args.halo_lr_px_north = north
    else:
        args.halo_sr = None
        args.halo_lr_px_west = 0
        args.halo_lr_px_north = 0
    parsed_args = _json_safe(vars(args))

    # Setup device - allow "cpu" as explicit device string
    if args.device.lower() == "cpu":
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        cuda_count = torch.cuda.device_count()
        try:
            requested_idx = int(args.device)
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
        torch.cuda.set_device(device)
    else:
        print(f"Warning: CUDA device {args.device} requested but CUDA not available. Using CPU.")
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    provenance = _experiment_provenance(args, device, parsed_args)
    args.dataset_device = resolve_dataset_device(args, training_device=device)
    print(f"Dataset cache device: {args.dataset_device}", flush=True)
    eval_autocast_dtype = get_eval_autocast_dtype(args.eval_mixed_precision, device)
    if eval_autocast_dtype is not None:
        label = "BF16" if eval_autocast_dtype == torch.bfloat16 else "FP16"
        print(f"Evaluation mixed precision: {label} (PSNR/metrics computed in this mode)")
    
    # Set seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    input_projection_name = args.input_projection.lower()
    if input_projection_name.startswith("fourier_"):
        args.fourier_scale = float(input_projection_name.split("_")[1])
        args.input_projection = "fourier"
    elif input_projection_name == "fourier":
        args.input_projection = "fourier"
    elif input_projection_name in {"hashgrid", "hash", "ngp_hash", "hashgrid_tcnn", "hash_tcnn"}:
        args.input_projection = input_projection_name
    elif input_projection_name == "none":
        args.input_projection = "none"
    else:
        raise ValueError(f"Unknown input projection: {args.input_projection}")

    if _uses_hetero_loss(args) and getattr(args, "use_gnll", False) and getattr(args, "use_laplace_nll", False):
        print("Error: --use_gnll and --use_laplace_nll are mutually exclusive.")
        return

    train_data = get_dataset(args=args, name=args.dataset, training_device=device)
    attach_halo_to_dataset(train_data, args)
    train_dataloader = build_train_dataloader(train_data, args)

    # Expose dataset LR shape into args so resolve_hash_resolutions can auto-size the hashgrid.
    if not getattr(args, "lr_height", 0):
        args.lr_height = int(getattr(train_data, "lr_height", 0) or 0)
    if not getattr(args, "lr_width", 0):
        args.lr_width = int(getattr(train_data, "lr_width", 0) or 0)
    if getattr(args, "eval_reference_s2_dir", ""):
        np_state, py_state = np.random.get_state(), random.getstate()
        with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
            train_data.sub_reference = load_subwindow_reference(args, train_data, device)
        np.random.set_state(np_state)
        random.setstate(py_state)

    # Setup model
    output_dim = _hetero_output_dim(args)
    input_projection, decoder = build_projection_and_decoder(args, device, output_dim=output_dim)
    model = build_model(args, input_projection, decoder, device)
    init_hetero_region_scales(model, train_data, device)
    parameter_counts = _model_parameter_counts(model)
    # model = NIR(input_projection, decoder, args.num_samples, use_gnll=args.use_gnll).to(device)

    # Setup optimizer
    optimizer = build_optimizer(model.parameters(), args)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.iters, eta_min=1e-6)
    grad_scaler = build_grad_scaler(args, device)
    print(
        f"Gradient loss scaling: {'off' if grad_scaler is None else f'on (init {grad_scaler.get_scale():.0f})'}"
        f", grad_accum={max(1, int(getattr(args, 'grad_accum', 1) or 1))}",
        flush=True,
    )

    print(f"Starting training for {args.iters} iterations (batch_size={args.batch_size})...")
    tile_sampler = build_lr_tile_sampler(train_data, args)
    cross_sampler = build_cross_frame_tile_sampler(train_data, args)
    if cross_sampler is not None:
        print(
            f"LR tile sampling: mix={resolve_lr_tile_mix(args)} tile={args.lr_tile} "
            f"tiles_per_step={cross_sampler.tiles_per_step} "
            f"n_origins={len(cross_sampler.origins)} n_frames={cross_sampler.num_frames}",
            flush=True,
        )
    elif tile_sampler is not None:
        print(
            f"LR tile sampling: mix=within tile={args.lr_tile} "
            f"tiles_per_step={tile_sampler.tiles_per_step} "
            f"n_origins={len(tile_sampler.origins)}",
            flush=True,
        )
    _reset_peak_memory(device)
    process_memory_sampler = _ProcessGpuMemorySampler(enabled=device.type == "cuda")
    process_memory_sampler.start()
    
    # Training loop
    iteration = 0
    progress_bar = tqdm(total=args.iters, desc="Training")
    training_start_time = time.time()
    eval_every = int(getattr(args, "eval_every", 100) or 0)
    skip_eval = bool(getattr(args, "skip_eval", False))
    holdout_state = init_early_stop_state(
        num_frames=int(getattr(train_data, "num_samples", len(train_data))),
        lr_height=int(getattr(train_data, "lr_height", 0) or getattr(args, "lr_height", 0)),
        lr_width=int(getattr(train_data, "lr_width", 0) or getattr(args, "lr_width", 0)),
        spatial_holdout=float(getattr(args, "spatial_holdout", 0.0) or 0.0),
        holdout_block=int(getattr(args, "holdout_block", 0) or 0),
        patience=int(getattr(args, "early_stop_patience", 0) or 0),
        min_iters=int(getattr(args, "early_stop_min_iters", 1000) or 0),
        min_delta=float(getattr(args, "early_stop_min_delta", 0.0) or 0.0),
        metric=str(getattr(args, "early_stop_metric", "lpips") or "lpips"),
        max_regression=_resolve_early_stop_max_regression(args),
        ema_alpha=_resolve_early_stop_ema(args),
        device=device,
    )
    holdout_state = _maybe_apply_lr_cloud_masks(holdout_state, train_data, args)
    val_every = _holdout_val_interval(args)
    val_loss_list: list[float] = []
    
    # Lists to store PSNR and losses for plotting
    psnr_list = []
    ssim_list = []
    lpips_list = []
    recon_loss_list = []
    trans_loss_list = []
    total_loss_list = []
    iteration_list = []
    instrumentation = {
        "step_count": 0,
        "query_totals": {
            "lr_pixels": 0,
            "lr_elements": 0,
            "hr_pixels": 0,
            "hr_elements": 0,
        },
    }
    cpu_step_time_seconds = 0.0
    cuda_step_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    sub_history: dict[str, list[float]] = {
        "sub_psnr": [], "sub_model_ssim": [], "sub_model_lpips": [],
        "sub_bilinear_psnr": [], "sub_bilinear_lpips": [],
    }
    elapsed_step_list: list[float] = []
    elapsed_wall_list: list[float] = []

    def _record_eval_extras(eval_metrics: dict) -> None:
        if cuda_step_events:
            torch.cuda.synchronize(device)
            elapsed_step_list.append(
                sum(start.elapsed_time(end) for start, end in cuda_step_events) / 1000.0
            )
        else:
            elapsed_step_list.append(cpu_step_time_seconds)
        elapsed_wall_list.append(time.time() - training_start_time)
        if "sub_model_lpips" in eval_metrics:
            sub_history["sub_psnr"].append(eval_metrics["sub_test_psnr"])
            sub_history["sub_model_ssim"].append(eval_metrics["sub_model_ssim"])
            sub_history["sub_model_lpips"].append(eval_metrics["sub_model_lpips"])
            sub_history["sub_bilinear_psnr"].append(eval_metrics["sub_bilinear_psnr"])
            sub_history["sub_bilinear_lpips"].append(eval_metrics["sub_bilinear_lpips"])

    while iteration < args.iters:
        stop_training = False
        step_samples = [None] if cross_sampler is not None else train_dataloader
        for train_sample in step_samples:
            if iteration >= args.iters:
                break
                
            # Train one iteration
            cpu_step_start = time.perf_counter()
            cuda_step_start = None
            cuda_step_end = None
            if device.type == "cuda":
                cuda_step_start = torch.cuda.Event(enable_timing=True)
                cuda_step_end = torch.cuda.Event(enable_timing=True)
                cuda_step_start.record()
            train_losses = train_one_iteration(
                model,
                optimizer,
                train_sample,
                device,
                args,
                variance_reg=args.variance_reg,
                variance_smooth_reg=args.variance_smooth_reg,
                holdout_state=holdout_state,
                iteration=iteration + 1,
                tile_sampler=tile_sampler,
                cross_sampler=cross_sampler,
                dataset=train_data,
                grad_scaler=grad_scaler,
                instrumentation=instrumentation,
            )
            if cuda_step_end is not None and cuda_step_start is not None:
                cuda_step_end.record()
                cuda_step_events.append((cuda_step_start, cuda_step_end))
            else:
                cpu_step_time_seconds += time.perf_counter() - cpu_step_start

            # Check for NaN/Inf in losses and break if detected
            if (torch.isnan(train_losses['recon_loss']) or 
                torch.isinf(train_losses['recon_loss']) or
                torch.isnan(train_losses['total_loss']) or 
                torch.isinf(train_losses['total_loss'])):
                scalars = _stack_train_loss_scalars(train_losses)
                print(f"\nERROR: NaN/Inf detected in losses at iteration {iteration}")
                print(f"Reconstruction loss: {scalars['recon_loss']}")
                print(f"Total loss: {scalars['total_loss']}")
                print("Stopping training to prevent further issues.")
                stop_training = True
                break
            
            scheduler.step()
            iteration += 1

            # Update progress bar
            progress_bar.update(1)
            if iteration % LOG_POSTFIX_INTERVAL == 0:
                scalars = _stack_train_loss_scalars(train_losses)
                postfix = _train_postfix_from_scalars(scalars)
                if val_loss_list:
                    postfix["val"] = f"{val_loss_list[-1]:.4f}"
                progress_bar.set_postfix(postfix)
            
            if holdout_state is not None and iteration % val_every == 0:
                scalars = _stack_train_loss_scalars(train_losses)
                val_loss, eval_metrics, should_stop = _run_holdout_val_step(
                    model=model,
                    dataset=train_data,
                    holdout_state=holdout_state,
                    args=args,
                    device=device,
                    iteration=iteration,
                    scalars=scalars,
                    skip_hr_eval=_skip_periodic_hr_eval(args),
                )
                val_loss_list.append(val_loss)
                if eval_metrics is not None:
                    if model.use_gnll and (
                        torch.isnan(train_losses['recon_loss'])
                        or torch.isinf(train_losses['recon_loss'])
                    ):
                        print(
                            f"WARNING: NaN/Inf detected in reconstruction loss "
                            f"at iteration {iteration}"
                        )
                    iteration_list.append(iteration)
                    _record_eval_extras(eval_metrics)
                    psnr_list.append(eval_metrics["test_psnr"])
                    ssim_list.append(eval_metrics["model_ssim"])
                    lpips_list.append(eval_metrics["model_lpips"])
                    recon_loss_list.append(scalars['recon_loss'])
                    trans_loss_list.append(scalars['trans_loss'])
                    total_loss_list.append(scalars['total_loss'])
                if should_stop:
                    stop_training = True
                    break
            elif eval_every > 0 and not _skip_periodic_hr_eval(args) and iteration % eval_every == 0:
                scalars = _stack_train_loss_scalars(train_losses)
                eval_autocast_dtype = get_eval_autocast_dtype(args.eval_mixed_precision, device)
                eval_metrics = eval_hr_metrics(
                    model, train_data, device, eval_autocast_dtype, args=args, iteration=iteration
                )
                print(_format_periodic_eval_line(iteration, scalars, eval_metrics))
                
                # Additional debugging for GNLL
                if model.use_gnll and (torch.isnan(train_losses['recon_loss']) or 
                                     torch.isinf(train_losses['recon_loss'])):
                    print(f"WARNING: NaN/Inf detected in reconstruction loss at iteration {iteration}")
                    print(f"Reconstruction loss: {scalars['recon_loss']}")
                    print(f"Total loss: {scalars['total_loss']}")

                # Append to lists for plotting
                iteration_list.append(iteration)
                _record_eval_extras(eval_metrics)
                psnr_list.append(eval_metrics["test_psnr"])
                ssim_list.append(eval_metrics["model_ssim"])
                lpips_list.append(eval_metrics["model_lpips"])
                recon_loss_list.append(scalars['recon_loss'])
                trans_loss_list.append(scalars['trans_loss'])
                total_loss_list.append(scalars['total_loss'])

        if stop_training:
            break

    progress_bar.close()
    training_time = time.time() - training_start_time
    if cuda_step_events:
        torch.cuda.synchronize(device)
        step_time_seconds = sum(
            start.elapsed_time(end) for start, end in cuda_step_events
        ) / 1000.0
    else:
        step_time_seconds = cpu_step_time_seconds
    training_instrumentation = _summarize_training_instrumentation(
        instrumentation["step_count"],
        step_time_seconds,
        instrumentation["query_totals"],
    )
    training_instrumentation["timing_source"] = (
        "cuda_events" if cuda_step_events else "host_perf_counter"
    )
    process_memory_sampler.stop()
    peak_memory_gb = _peak_memory_gb(device)
    gpu_memory = _gpu_memory_metrics(device, process_memory_sampler)
    if peak_memory_gb is not None:
        process_peak = gpu_memory["process_peak_used_gb"]
        process_text = (
            f", process/device {process_peak:.2f} GB ({gpu_memory['process_memory_source']})"
            if process_peak is not None
            else ""
        )
        print(
            f"Peak training GPU memory: torch allocated {peak_memory_gb:.2f} GB, "
            f"torch reserved {gpu_memory['torch_peak_reserved_gb']:.2f} GB"
            f"{process_text}",
            flush=True,
        )

    if holdout_state is not None and holdout_state.restore_best(model):
        print(
            f"Restored best holdout checkpoint from iter {holdout_state.best_iter} "
            f"(val={holdout_state.best_val:.6f})"
        )
    checkpoint_metadata = {
        "final_iteration": int(iteration),
        "completed_iterations": int(iteration),
        "stopped_early": bool(holdout_state.stopped) if holdout_state is not None else False,
        "stopped_iteration": (
            int(holdout_state.stopped_iter)
            if holdout_state is not None and holdout_state.stopped_iter
            else None
        ),
        "best_iteration": (
            int(holdout_state.best_iter)
            if holdout_state is not None and holdout_state.best_iter
            else None
        ),
        "best_score": (
            float(holdout_state.best_score)
            if holdout_state is not None and holdout_state.best_iter
            else None
        ),
        "restored_best": bool(holdout_state.restored) if holdout_state is not None else False,
    }

    final_iter = iteration
    if holdout_state is not None and holdout_state.best_iter:
        final_iter = int(holdout_state.best_iter)
    checkpoint_metadata["final_iteration"] = int(final_iter)
    final_kwargs = _final_schedule_kwargs(args, final_iter)
    final_progress = final_kwargs.get("progress")
    
    # Final evaluation and save output
    model.eval()
    eval_autocast_dtype = get_eval_autocast_dtype(args.eval_mixed_precision, device)
    has_hr = _dataset_has_hr_gt(train_data)
    with torch.no_grad():
        hr_coords = train_data.get_hr_coordinates().unsqueeze(0).to(device)
        hr_image = train_data.get_original_hr().unsqueeze(0).to(device)
        sample_id = torch.tensor([0]).to(device)
        
        fwd_kwargs = {"lr_align_args": final_kwargs["lr_align_args"]}
        output = _forward_hr_output(
            model,
            hr_coords,
            hr_image,
            sample_id,
            device,
            eval_autocast_dtype,
            hr_render_tile=int(getattr(args, "hr_render_tile", 0) or 0),
            **fwd_kwargs,
        )

        # Unstandardize the output
        output = (
            output * train_data.get_lr_std(0).to(output.device)
            + train_data.get_lr_mean(0).to(output.device)
        )

        pred_tensor = output if output.ndim == 4 else output.unsqueeze(0)
        if pred_tensor.shape[-1] == 3:
            pred_tensor = pred_tensor.permute(0, 3, 1, 2)
        gt_tensor = hr_image if hr_image.ndim == 4 else hr_image.unsqueeze(0)
        if gt_tensor.shape[-1] == 3:
            gt_tensor = gt_tensor.permute(0, 3, 1, 2)

        pred_np = pred_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        gt_np = gt_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

        # Build a 3-channel LR baseline image for visualization
        if hasattr(train_data, 'get_lr_sample_hwc'):
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
            lr_original = train_data.get_lr_sample(0).cpu().numpy()  # C x H x W (unstandardized, [0, 1])
            
            # Convert from CHW to HWC for visualization
            if lr_original.ndim == 3:
                if lr_original.shape[0] in (1, 3, 4):  # CHW format
                    lr_original = lr_original.transpose(1, 2, 0)  # Convert to HWC
                    # Handle multi-frame case if needed
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
        hr_h, hr_w = pred_np.shape[:2]
        lr_bilinear = cv2.resize(lr_original, (hr_w, hr_h), interpolation=cv2.INTER_LINEAR)
        pred_np = np.clip(pred_np, 0, 1)
        lr_original = np.clip(lr_original, 0, 1)
        lr_bilinear = np.clip(lr_bilinear, 0, 1)
        if has_hr:
            gt_np = np.clip(gt_np, 0, 1)
        else:
            gt_np = None

        # Convert numpy arrays to torch tensors for alignment and color matching
        pred_tensor = torch.from_numpy(pred_np).unsqueeze(0).permute(0, 3, 1, 2).to(device)  # [1, C, H, W]
        bilinear_tensor = torch.from_numpy(lr_bilinear).unsqueeze(0).permute(0, 3, 1, 2).to(device)  # [1, C, H, W]
        
        # Align outputs for fair comparison (following og_main.py approach)
        # Alignment disabled to avoid OOM errors - can be re-enabled if needed
        print("Skipping alignment (disabled to avoid memory issues)")
        pred_aligned = pred_tensor
        bilinear_aligned = bilinear_tensor

        if has_hr:
            gt_tensor = torch.from_numpy(gt_np).unsqueeze(0).permute(0, 3, 1, 2).to(device)
            eval_mask_hw = _get_hr_eval_mask(train_data)
            if eval_mask_hw is not None:
                print(
                    f"Using masked HR eval on {train_data.hr_valid_fraction * 100:.1f}% valid GT pixels"
                )

            lpips_fn = get_lpips_model(device)
            frame_metrics = _compute_full_frame_metrics(
                pred_aligned, gt_tensor, bilinear_aligned, device, lpips_fn, eval_mask_hw
            )
            fixed_spot = _maybe_fixed_spot_metrics(
                pred_aligned, gt_tensor, bilinear_aligned, device, lpips_fn, args, dataset=train_data
            )
        else:
            print("No HR GT (--allow_no_hr): skipping LPIPS/PSNR/SSIM vs GT; writing SR GeoTIFFs only.")
            frame_metrics = _nan_hr_metrics()
            fixed_spot = {"skipped": True, "skip_reason": "no_hr_gt"}
            lpips_fn = None

        query_gsd_metrics = None

        final_test_loss = frame_metrics["test_loss"]
        final_psnr = frame_metrics["test_psnr"]
        model_psnr = frame_metrics["model_psnr"]
        bilinear_psnr = frame_metrics["bilinear_psnr"]
        model_ssim = frame_metrics["model_ssim"]
        bilinear_ssim = frame_metrics["bilinear_ssim"]
        pred_lpips = frame_metrics["model_lpips"]
        bilinear_lpips = frame_metrics["bilinear_lpips"]

        if fixed_spot and not fixed_spot.get("skipped"):
            print(
                f"Fixed spot ({fixed_spot['hr_pixels']}×{fixed_spot['hr_pixels']} HR, center): "
                f"PSNR {fixed_spot['model_psnr']:.2f} dB "
                f"(bil {fixed_spot['bilinear_psnr']:.2f}, Δ {fixed_spot['psnr_improvement']:+.2f}), "
                f"SSIM {fixed_spot['model_ssim']:.4f} "
                f"(bil {fixed_spot['bilinear_ssim']:.4f}), "
                f"LPIPS {fixed_spot['model_lpips']:.4f} "
                f"(bil {fixed_spot['bilinear_lpips']:.4f})"
            )
        elif fixed_spot and fixed_spot.get("skipped"):
            print(
                f"Fixed spot skipped ({fixed_spot.get('skip_reason', 'unavailable')})"
            )

        # Convert aligned tensors back to numpy for visualization
        pred_aligned_np = pred_aligned.squeeze(0).permute(1, 2, 0).cpu().numpy()
        bilinear_aligned_np = bilinear_aligned.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        # Ensure aligned images are in valid range
        pred_aligned_np = np.clip(pred_aligned_np, 0, 1)
        bilinear_aligned_np = np.clip(bilinear_aligned_np, 0, 1)

        query_gsd_metrics = None
        if has_hr:
            from eval.query_gsd import eval_query_gsds, parse_query_gsd_m

            query_gsds = parse_query_gsd_m(getattr(args, "query_gsd_m", None))
            if query_gsds:
                query_gsd_metrics = eval_query_gsds(
                    model,
                    train_data,
                    query_gsds,
                    device=device,
                    args=args,
                    fwd_kwargs={
                        **fwd_kwargs,
                        "eval_autocast_dtype": eval_autocast_dtype,
                    },
                    lpips_fn=lpips_fn,
                    train_pred_hwc=pred_aligned_np,
                )
        
        # Create structured output directory for single sample results
        output_base_dir = Path("single_samples")
        dataset_dir = output_base_dir / args.dataset
        sample_dir = dataset_dir / str(args.sample_id)
        if getattr(args, "run_name", None):
            sample_dir = sample_dir / str(args.run_name)
        sample_dir.mkdir(parents=True, exist_ok=True)

        save_eval_visualizations(
            sample_dir,
            lr_hwc=lr_original,
            bilinear_hwc=bilinear_aligned_np,
            pred_hwc=pred_aligned_np,
            gt_hwc=gt_np,
            image_metrics={
                "model_psnr": model_psnr,
                "bilinear_psnr": bilinear_psnr,
                "model_ssim": model_ssim,
                "bilinear_ssim": bilinear_ssim,
                "model_lpips": pred_lpips,
                "bilinear_lpips": bilinear_lpips,
            },
            fixed_spot=(None if not fixed_spot or fixed_spot.get("skipped") else fixed_spot),
            sample_label=str(args.sample_id),
        )
        comparison_path = sample_dir / "comparison.png"
        output_path = comparison_path

        if not bool(getattr(args, "no_qgis_export", False)):
            try:
                qgis_dir = sample_dir / "qgis"
                written = export_qgis_layers(
                    qgis_dir,
                    hr_gt_hwc=gt_np,
                    sr_pred_hwc=pred_aligned_np,
                    s2_bilinear_hwc=bilinear_aligned_np,
                    dataset=train_data,
                    lr_hwc=lr_original,
                )
                print(f"QGIS GeoTIFFs written to {qgis_dir}: {', '.join(sorted(written))}")
            except Exception as exc:  # noqa: BLE001
                print(f"WARN: QGIS GeoTIFF export skipped: {exc}")
        
    if has_hr:
        print(f"\nFinal Results:")
        print(f"Test Loss: {final_test_loss:.6f}")
        print(f"Test PSNR: {final_psnr:.2f} dB")
        print(f"Model PSNR: {model_psnr:.2f} dB (bilinear {bilinear_psnr:.2f}, Δ {model_psnr - bilinear_psnr:+.2f} dB)")
        print(f"Model SSIM: {model_ssim:.4f} (bilinear {bilinear_ssim:.4f}, Δ {model_ssim - bilinear_ssim:+.4f})")
        print(f"Model LPIPS: {pred_lpips:.4f} (bilinear {bilinear_lpips:.4f}, Δ {bilinear_lpips - pred_lpips:+.4f}; lower is better)")
    else:
        print(f"\nFinal Results (production, no HR GT):")
        print(f"SR + bilinear GeoTIFFs written under {output_path.parent}")
    print(f"Model output saved to {output_path}")
    
    # Create structured output directory for single sample results
    output_base_dir = Path("single_samples")
    dataset_dir = output_base_dir / args.dataset
    sample_dir = dataset_dir / str(args.sample_id)
    if getattr(args, "run_name", None):
        sample_dir = sample_dir / str(args.run_name)
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Save PSNR results to a text file in the structured directory
    if has_hr:
        results_text = f"""Super-Resolution Results
    =======================

    Dataset: {args.dataset}
    Sample ID: {args.sample_id}
    Downsampling Factor: {args.df}
    Model: {args.model}
    Iterations: {args.iters}

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
"""
    else:
        results_text = f"""Super-Resolution Results (production, no HR GT)
    ==============================================

    Dataset: {args.dataset}
    Sample ID: {args.sample_id}
    Downsampling Factor: {args.df}
    Model: {args.model}
    Iterations: {args.iters}
    Completed iters: {iteration}

    No NIB/HR ground truth — LPIPS/PSNR/SSIM vs GT are N/A.
    Deliverables: qgis/sr_pred.tif, qgis/s2_bilinear.tif, qgis/s2_lr.tif
"""
    if has_hr and fixed_spot and not fixed_spot.get("skipped"):
        results_text += f"""
    Fixed Spot ({fixed_spot['hr_pixels']}×{fixed_spot['hr_pixels']} HR, center):
    - Model PSNR: {fixed_spot['model_psnr']:.2f} dB
    - Bilinear PSNR: {fixed_spot['bilinear_psnr']:.2f} dB
    - PSNR Improvement: {fixed_spot['psnr_improvement']:+.2f} dB
    - Model SSIM: {fixed_spot['model_ssim']:.4f}
    - Bilinear SSIM: {fixed_spot['bilinear_ssim']:.4f}
    - Model LPIPS: {fixed_spot['model_lpips']:.4f}
    - Bilinear LPIPS: {fixed_spot['bilinear_lpips']:.4f}
"""
    elif fixed_spot and fixed_spot.get("skipped"):
        results_text += f"""
    Fixed Spot skipped: {fixed_spot.get('skip_reason', 'unavailable')}
"""
    results_text += f"""
    Training Results:
    - Final Test Loss: {final_test_loss:.6f}
    - Final Test PSNR: {final_psnr:.2f} dB
    - Final Reconstruction Loss: {recon_loss_list[-1] if recon_loss_list else 0:.6f}
    - Final Transformation Loss: {trans_loss_list[-1] if trans_loss_list else 0:.6f}
    - Final Total Loss: {total_loss_list[-1] if total_loss_list else 0:.6f}

    Training Metrics History:
    """
    
    if len(psnr_list) > 0:
        results_text += f"- Number of evaluation points: {len(psnr_list)}\n"
        results_text += f"- PSNR range: {min(psnr_list):.2f} - {max(psnr_list):.2f} dB\n"
        if len(ssim_list) > 0:
            results_text += f"- SSIM range: {min(ssim_list):.4f} - {max(ssim_list):.4f}\n"
        if len(lpips_list) > 0:
            results_text += f"- LPIPS range: {min(lpips_list):.4f} - {max(lpips_list):.4f}\n"
        results_text += f"- Reconstruction loss range: {min(recon_loss_list):.6f} - {max(recon_loss_list):.6f}\n"
        results_text += f"- Transformation loss range: {min(trans_loss_list):.6f} - {max(trans_loss_list):.6f}\n"
        results_text += f"- Total loss range: {min(total_loss_list):.6f} - {max(total_loss_list):.6f}\n"
        results_text += f"- Final PSNR: {psnr_list[-1]:.2f} dB\n"
        if len(ssim_list) > 0:
            results_text += f"- Final SSIM: {ssim_list[-1]:.4f}\n"
        if len(lpips_list) > 0:
            results_text += f"- Final LPIPS: {lpips_list[-1]:.4f}\n"
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
        'dataset': args.dataset,
        'sample_id': str(args.sample_id),
        'run_name': getattr(args, "run_name", None),
        'has_hr_gt': bool(has_hr),
        'allow_no_hr': bool(getattr(args, "allow_no_hr", False)),
        'lr_degradation': str(getattr(args, "lr_degradation", "s2_psf")),
        'recon_loss': str(getattr(args, "recon_loss", "mse")),
        'use_gnll': bool(getattr(args, "use_gnll", False)),
        'use_laplace_nll': bool(getattr(args, "use_laplace_nll", False)),
        'hetero_scale': str(getattr(args, "hetero_scale", "pixel")),
        'hetero_region_size': int(getattr(args, "hetero_region_size", 4)),
        'hetero_loss': (
            'laplace' if getattr(args, "use_laplace_nll", False)
            else 'gaussian' if getattr(args, "use_gnll", False)
            else None
        ),
        'downsampling_factor': args.df,
        'model': args.model,
        'input_projection': args.input_projection,
        'iterations': args.iters,
        'completed_iters': iteration,
        'learning_rate': args.learning_rate,
        'model_psnr': model_psnr,
        'bilinear_psnr': bilinear_psnr,
        'final_test_psnr': final_psnr,
        'training_time_seconds': training_time,
        'peak_memory_gb': peak_memory_gb,
        'provenance': provenance,
        'eval_alignment': getattr(train_data, "hr_spatial_alignment", None),
        'model_parameters': parameter_counts,
        **_fourier_matrix_provenance(model, args),
        'training_instrumentation': training_instrumentation,
        'gpu_memory': gpu_memory,
        'checkpoint': checkpoint_metadata,
        'lr_tile': int(getattr(args, "lr_tile", 0) or 0),
        'lr_tile_halo': resolve_lr_tile_halo(args),
        'lr_tiles_per_step': raw_tiles_per_step(args),
        'lr_tile_mix': resolve_lr_tile_mix(args),
        'spatial_holdout': float(getattr(args, "spatial_holdout", 0.0) or 0.0),
        'lr_cloud_mask': {
            'enabled': bool(getattr(args, "mask_lr_clouds", True)),
            'has_mask': bool(getattr(train_data, "has_lr_cloud_mask", False)),
            'clear_fractions': list(getattr(train_data, "lr_clear_fractions", []) or []),
            'stats_pixels': getattr(train_data, "lr_stats_pixels", None),
            'stats_masked': list(getattr(train_data, "lr_stats_masked", []) or []),
        },
        'halo': {
            'sr': getattr(args, "halo_sr", None),
            'west_px': int(getattr(args, "halo_lr_px_west", 0) or 0),
            'north_px': int(getattr(args, "halo_lr_px_north", 0) or 0),
            'weight': float(getattr(args, "halo_weight", 0.0) or 0.0),
            'mode': str(getattr(args, "halo_mode", "full") or "full"),
            'lf_hr_px': int(getattr(args, "halo_lf_hr_px", 16) or 16),
        },
        'base_frame': {
            'date': getattr(train_data, 'base_frame_date', None),
            'cloud_frac': getattr(train_data, 'base_frame_cloud_frac', None),
            'max_cloud_frac': getattr(train_data, 'max_base_cloud_frac', None),
            'force_date': getattr(train_data, 'force_base_date', None),
        },
        'holdout_block': int(
            (holdout_state.holdout_block if holdout_state is not None else 0)
            or getattr(args, "holdout_block", 0)
            or 0
        ),
        'schedule_horizon_iters': int(getattr(args, "schedule_horizon_iters", 3000) or 3000),
        'psf_curriculum': str(getattr(args, "psf_curriculum", "none") or "none"),
        'psf_sigma_schedule': str(getattr(args, "psf_sigma_schedule", "none") or "none"),
        'early_stop': (
            {
                **holdout_state.summary(),
                'patience': int(getattr(args, "early_stop_patience", 0) or 0),
                'min_iters': int(getattr(args, "early_stop_min_iters", 0) or 0),
                'min_delta': float(getattr(args, "early_stop_min_delta", 0.0) or 0.0),
                'max_regression': _resolve_early_stop_max_regression(args),
                'metric': str(getattr(args, "early_stop_metric", "lpips") or "lpips"),
            }
            if holdout_state is not None
            else None
        ),
        'eval_mask': {
            'masked': bool(getattr(train_data, 'use_masked_eval', False)),
            'valid_fraction': float(getattr(train_data, 'hr_valid_fraction', 1.0)),
        },
        'psnr': {
            'model': model_psnr,
            'bilinear': bilinear_psnr,
            'improvement': model_psnr - bilinear_psnr
        },
        'ssim': {
            'model': model_ssim,
            'bilinear': bilinear_ssim,
            'improvement': model_ssim - bilinear_ssim
        },
        'lpips': {
            'model': pred_lpips,
            'bilinear': bilinear_lpips,
            'improvement': bilinear_lpips - pred_lpips
        },
        'query_gsd': query_gsd_metrics,
        'training': {
            'final_test_loss': final_test_loss,
            'final_test_psnr': final_psnr,
            'final_recon_loss': recon_loss_list[-1] if recon_loss_list else 0,
            'final_trans_loss': trans_loss_list[-1] if trans_loss_list else 0,
            'final_total_loss': total_loss_list[-1] if total_loss_list else 0,
            'history': {
                'iterations': iteration_list,
                'psnr': psnr_list,
                'model_ssim': ssim_list,
                'model_lpips': lpips_list,
                'recon_loss': recon_loss_list,
                'trans_loss': trans_loss_list,
                'total_loss': total_loss_list,
                'val_loss': val_loss_list if holdout_state is not None else [],
                'elapsed_step_seconds': elapsed_step_list,
                'elapsed_wall_seconds': elapsed_wall_list,
                **({k: v for k, v in sub_history.items()} if sub_history["sub_model_lpips"] else {}),
            },
            **(
                {'eval_subwindow_lr': list(_parse_eval_subwindow(args))}
                if _parse_eval_subwindow(args) is not None else {}
            ),
            **(
                {'eval_subwindow_reference': train_data.sub_reference["provenance"]}
                if getattr(train_data, "sub_reference", None) is not None else {}
            ),
        }
    }
    if int(getattr(args, "lr_size", 0) or 0) > 0:
        metrics_dict["lr_size"] = int(args.lr_size)
    if fixed_spot:
        metrics_dict["fixed_spot"] = fixed_spot
    
    with open(sample_dir / "metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=2)
    write_affine_dump(model, train_data, sample_dir)
    
    print(f"Results saved to: {sample_dir}")
    print(f"PSNR results also saved to psnr_results.txt (current directory)")

    # Plot PSNR / SSIM / LPIPS and losses
    if len(psnr_list) > 0:
        nrows = 3 if len(ssim_list) > 0 else 2
        fig, axes = plt.subplots(nrows, 1, figsize=(12, 4 * nrows))
        if nrows == 2:
            ax1, ax2 = axes
        else:
            ax1, ax_mid, ax2 = axes

        ax1.plot(iteration_list, psnr_list, color='blue', linewidth=2, label='PSNR (Test)')
        ax1.set_xlabel('Iteration', fontsize=12)
        ax1.set_ylabel('PSNR (dB)', fontsize=12)
        ax1.set_title('Training PSNR Evolution', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        if nrows == 3:
            ax_mid.plot(iteration_list, ssim_list, color='purple', linewidth=2, label='SSIM')
            ax_mid.set_xlabel('Iteration', fontsize=12)
            ax_mid.set_ylabel('SSIM', fontsize=12, color='purple')
            ax_mid.tick_params(axis='y', labelcolor='purple')
            ax_mid.grid(True, alpha=0.3)
            ax_mid_lpips = ax_mid.twinx()
            ax_mid_lpips.plot(iteration_list, lpips_list, color='brown', linewidth=2, label='LPIPS')
            ax_mid_lpips.set_ylabel('LPIPS (lower better)', fontsize=12, color='brown')
            ax_mid_lpips.tick_params(axis='y', labelcolor='brown')
            ax_mid.set_title('Training SSIM / LPIPS Evolution', fontsize=14, fontweight='bold')
            lines_l, labels_l = ax_mid.get_legend_handles_labels()
            lines_r, labels_r = ax_mid_lpips.get_legend_handles_labels()
            ax_mid.legend(lines_l + lines_r, labels_l + labels_r, loc='best')

        ax2.plot(iteration_list, recon_loss_list, color='red', linewidth=2, label='Reconstruction Loss')
        ax2.plot(iteration_list, trans_loss_list, color='green', linewidth=2, label='Transformation Loss')
        ax2.plot(iteration_list, total_loss_list, color='purple', linewidth=2, label='Total Loss')
        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.set_title('Training Loss Evolution', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        # Save to both current directory (for backward compatibility) and structured directory
        plt.savefig("training_metrics.png", bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.savefig(sample_dir / "training_metrics.png", bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close()
        
        print(f"Training metrics plot saved to training_metrics.png and {sample_dir}/training_metrics.png")
    else:
        print("No metrics data available for plotting (training may have been too short)")
    
    # Generate variance visualizations only when explicitly requested.
    use_gnll_loss = model.use_gnll
    if args.visualize_variance and use_gnll_loss:
        print("Generating variance visualizations for each LR sample...")
        # Clear GPU memory before variance visualization
        torch.cuda.empty_cache()
        visualize_lr_variance(model, train_data, device, sample_dir, args.sample_id)
    elif args.visualize_variance and not use_gnll_loss:
        print("Warning: --visualize_variance requested but model does not use GNLL. Skipping variance visualization.")


if __name__ == "__main__":
    main() 
