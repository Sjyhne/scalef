import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def _bootstrap_nvjitlink_from_venv() -> None:
    if os.environ.get("SCALEF_NVJITLINK_BOOTSTRAP", "1") == "0":
        return
    if sys.platform != "linux":
        return
    if os.environ.get("SCALEF_NVJITLINK_BOOTSTRAPPED", "0") == "1":
        return

    venv_site = (
        Path(sys.prefix)
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    nvjitlink_lib = venv_site / "nvidia" / "nvjitlink" / "lib"
    if not nvjitlink_lib.exists():
        return
    if not any(
        (nvjitlink_lib / name).exists() for name in ("libnvJitLink.so.12", "libnvJitLink.so")
    ):
        return

    cur = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [p for p in cur.split(":") if p]
    nvjit_str = str(nvjitlink_lib)
    if parts and parts[0] == nvjit_str:
        return
    if nvjit_str in parts:
        parts.remove(nvjit_str)

    os.environ["LD_LIBRARY_PATH"] = ":".join([nvjit_str] + parts)
    os.environ["SCALEF_NVJITLINK_BOOTSTRAPPED"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)


_bootstrap_nvjitlink_from_venv()
import lpips
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchmetrics.functional.image import (
    peak_signal_noise_ratio,
)
from torchmetrics.functional.image import (
    structural_similarity_index_measure as ssim,
)
from tqdm import tqdm

from data import get_dataset
from input_projections.utils import get_input_projection, normalize_input_projection_name
from losses import BasicLosses
from models.inr import INR, get_inr
from models.nir import NIR
from models.utils import get_decoder
from optimizers import build_optimizer
from s2_reflectance_utils import reflectance_b432_chw_to_dn_style


def strict_lr_valid_mask(
    warped_grid: torch.Tensor, scale: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return [B,H_lr,W_lr,1] mask where every df x df warped HR coord is inside [0,1]."""
    if warped_grid.ndim != 4 or warped_grid.shape[-1] != 2:
        raise ValueError(f"Expected warped grid [B,H,W,2], got {tuple(warped_grid.shape)}")
    x = warped_grid[..., 0]
    y = warped_grid[..., 1]
    valid_hr = ((x >= 0.0) & (x <= 1.0) & (y >= 0.0) & (y <= 1.0)).float().unsqueeze(1)
    coverage_lr = F.avg_pool2d(valid_hr, kernel_size=scale, stride=scale)
    valid_lr = (coverage_lr >= (1.0 - 1e-6)).permute(0, 2, 3, 1)
    return valid_lr, coverage_lr.permute(0, 2, 3, 1)


def unstandardize_hwc(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Unstandardize an HWC/BHWC tensor using SRData per-channel stats."""
    while mean.ndim < x.ndim:
        mean = mean.unsqueeze(1)
    while std.ndim < x.ndim:
        std = std.unsqueeze(1)
    return x * std.to(x.device) + mean.to(x.device)


def make_reflectance_display_stretch(images: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel robust stretch for low-valued S2 B4/B3/B2 reflectance images."""
    vals = []
    for image in images:
        arr = np.asarray(image, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[-1] >= 3:
            vals.append(np.clip(arr[..., :3], 0.0, None).reshape(-1, 3))
    if not vals:
        return np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32)

    stacked = np.concatenate(vals, axis=0)
    lo = np.nanpercentile(stacked, 0.5, axis=0).astype(np.float32)
    hi = np.nanpercentile(stacked, 99.0, axis=0).astype(np.float32)
    lo = np.clip(lo, 0.0, None)
    hi = np.maximum(hi, lo + 1e-4)
    return lo, hi


def display_hwc(
    image_hwc: np.ndarray,
    use_raw_b432: bool,
    reflectance_stretch: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Return display RGB HWC float [0,1] for either normal RGB or S2 B4/B3/B2 reflectance."""
    image_hwc = np.asarray(image_hwc, dtype=np.float32)
    if use_raw_b432:
        _ = reflectance_stretch  # kept for backward compatibility with older callers
        chw = np.transpose(np.asarray(image_hwc[..., :3], dtype=np.float32), (2, 0, 1))
        dn = reflectance_b432_chw_to_dn_style(chw)
        rgb = np.transpose(dn, (1, 2, 0))  # HWC
        min_value, q99_value = np.quantile(rgb, q=[0.0, 0.99])
        rgb01 = (rgb - min_value) / (q99_value - min_value + 1e-6)
        rgb01 = np.clip(rgb01, 0.0, 1.0)
        rgb01 = np.power(rgb01, 0.7)
        return rgb01.astype(np.float32)
    return np.clip(image_hwc, 0.0, 1.0)


def save_lr_reprojection_diagnostic(
    pred_hr_hwc: np.ndarray,
    lr_target_hwc: np.ndarray,
    df: int,
    out_path: Path,
    use_raw_b432: bool = False,
) -> None:
    """
    Save pred_hr vs avg-pooled pred_lr vs target_lr diagnostic.

    This separates "HR has unconstrained high-frequency junk" from "LR reprojection is wrong".
    """
    df = max(int(df), 1)
    pred_hr_hwc = np.asarray(pred_hr_hwc, dtype=np.float32)
    lr_target_hwc = np.asarray(lr_target_hwc, dtype=np.float32)

    pred_hr_bchw = torch.from_numpy(pred_hr_hwc).permute(2, 0, 1).unsqueeze(0)
    pred_lr_bchw = F.avg_pool2d(pred_hr_bchw, kernel_size=df, stride=df)
    pred_lr_up_bchw = F.interpolate(pred_lr_bchw, scale_factor=df, mode="nearest")
    lr_target_bchw = torch.from_numpy(lr_target_hwc).permute(2, 0, 1).unsqueeze(0)
    lr_target_up_bchw = F.interpolate(lr_target_bchw, scale_factor=df, mode="nearest")

    pred_lr_hwc = pred_lr_bchw.squeeze(0).permute(1, 2, 0).numpy()
    pred_lr_up_hwc = pred_lr_up_bchw.squeeze(0).permute(1, 2, 0).numpy()
    lr_target_up_hwc = lr_target_up_bchw.squeeze(0).permute(1, 2, 0).numpy()
    lr_mse = float(np.mean((pred_lr_hwc - lr_target_hwc) ** 2))
    lr_psnr = float(-10.0 * np.log10(max(lr_mse, 1e-12)))

    reflectance_stretch = (
        make_reflectance_display_stretch([pred_hr_hwc, pred_lr_up_hwc, lr_target_up_hwc])
        if use_raw_b432
        else None
    )
    pred_hr_display = display_hwc(pred_hr_hwc, use_raw_b432, reflectance_stretch)
    pred_lr_up_display = display_hwc(pred_lr_up_hwc, use_raw_b432, reflectance_stretch)
    lr_target_up_display = display_hwc(lr_target_up_hwc, use_raw_b432, reflectance_stretch)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(pred_hr_display)
    axes[0].set_title("1. pred_hr", fontsize=13, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(pred_lr_up_display)
    axes[1].set_title(
        f"2. avg_pool(pred_hr), nearest up\nLR PSNR: {lr_psnr:.2f} dB",
        fontsize=13,
        fontweight="bold",
    )
    axes[1].axis("off")

    axes[2].imshow(lr_target_up_display)
    axes[2].set_title("3. LR target, nearest up", fontsize=13, fontweight="bold")
    axes[2].axis("off")

    plt.tight_layout(pad=1.0)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.1, dpi=300)
    plt.close(fig)


def train_one_iteration(
    model,
    optimizer,
    train_sample,
    device,
    iteration=0,
    total_iterations=1,
    **_ignored_kwargs,
):
    model.train()
    recon_criterion = BasicLosses.mse_loss

    input_coords = train_sample["input"].to(device)
    lr_target = train_sample["lr_target"].to(device)
    sample_id = train_sample["sample_id"].to(device)
    scale_factor = train_sample["scale_factor"].to(device)

    if "shifts" in train_sample and "dx_percent" in train_sample["shifts"]:
        gt_dx = train_sample["shifts"]["dx_percent"].to(device)
        gt_dy = train_sample["shifts"]["dy_percent"].to(device)
    else:
        gt_dx = torch.zeros(lr_target.shape[0], device=device)
        gt_dy = torch.zeros(lr_target.shape[0], device=device)

    progress = float(iteration) / max(float(total_iterations - 1), 1.0)

    optimizer.zero_grad()
    valid_fraction = 1.0
    if isinstance(model, INR) and scale_factor.unique().numel() == 1:
        output_hr, pred_shifts = model(
            input_coords, sample_id, scale_factor=None, lr_frames=lr_target, progress=progress
        )
        affines = model.get_direct_affine(sample_id)
        warped_grid = model.apply_affine(input_coords, affines)

        scale = max(int(round(float(scale_factor.unique().item()))), 1)
        pred_hr = output_hr.permute(0, 3, 1, 2)  # [B,C,H_hr,W_hr]
        pred_lr = F.avg_pool2d(pred_hr, kernel_size=scale, stride=scale).permute(0, 2, 3, 1)
        valid_lr, _ = strict_lr_valid_mask(warped_grid, scale=scale)
        valid = valid_lr.expand_as(pred_lr)
        valid_fraction = float(valid_lr.float().mean().detach().item())

        residual = (pred_lr - lr_target) ** 2
        recon_loss = residual[valid].mean() if valid.any() else pred_lr.sum() * 0.0
    else:
        output, pred_shifts = model(
            input_coords,
            sample_id,
            scale_factor=1 / scale_factor,
            lr_frames=lr_target,
            progress=progress,
        )
        recon_loss = recon_criterion(output, lr_target)

    if isinstance(model, INR):
        pred_dx, pred_dy = pred_shifts
        lr_h, lr_w = lr_target.shape[1:3]
        pred_dx_percent = pred_dx / lr_w
        pred_dy_percent = pred_dy / lr_h
        trans_loss = torch.mean(
            torch.sqrt((pred_dx_percent - gt_dx) ** 2 + (pred_dy_percent - gt_dy) ** 2)
        )
    else:
        trans_loss = torch.zeros(1, device=device)

    total_loss = recon_loss
    total_loss.backward()
    optimizer.step()
    if isinstance(model, INR):
        model.clamp_reference_frame()

    return {
        "recon_loss": recon_loss.item(),
        "trans_loss": trans_loss.item(),
        "total_loss": total_loss.item(),
        "valid_fraction": valid_fraction,
    }


def capture_affine_motion_rows(model, device, num_samples, iteration):
    """Snapshot each learnable 2x3 affine (canonical -> frame) for INR; no-op for NIR."""
    if not isinstance(model, INR):
        return []
    n = min(int(num_samples), len(model.affine_params))
    rows = []
    with torch.no_grad():
        for sample_idx in range(n):
            sid = torch.tensor([sample_idx], device=device, dtype=torch.long)
            A = model.get_direct_affine(sid)
            rows.append(
                {
                    "iteration": int(iteration),
                    "sample_idx": sample_idx,
                    "a00": float(A[0, 0, 0].item()),
                    "a01": float(A[0, 0, 1].item()),
                    "tx": float(A[0, 0, 2].item()),
                    "a10": float(A[0, 1, 0].item()),
                    "a11": float(A[0, 1, 1].item()),
                    "ty": float(A[0, 1, 2].item()),
                }
            )
    return rows


def save_affine_motion_artifacts(rows, sample_dir, num_samples, title_prefix=""):
    """Write affine_motion.csv plus translation time-series and scatter plots; empty rows -> (None,)*3."""
    sample_dir = Path(sample_dir)
    if not rows:
        return None, None, None

    fieldnames = ["iteration", "sample_idx", "a00", "a01", "tx", "a10", "a11", "ty"]
    csv_path = sample_dir / "affine_motion.csv"
    sorted_rows = sorted(rows, key=lambda r: (r["iteration"], r["sample_idx"]))
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in sorted_rows:
            writer.writerow(row)

    by_sample = {}
    for row in rows:
        by_sample.setdefault(int(row["sample_idx"]), []).append(row)
    for pts in by_sample.values():
        pts.sort(key=lambda r: r["iteration"])

    plot_path = sample_dir / "affine_motion_timeseries.png"
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ts_title = (
        f"{title_prefix}: affine translation vs iteration"
        if title_prefix
        else "Affine translation vs iteration"
    )
    fig.suptitle(ts_title, fontsize=12)
    for sid in sorted(by_sample.keys()):
        pts = by_sample[sid]
        it = [p["iteration"] for p in pts]
        axes[0].plot(it, [p["tx"] for p in pts], marker=".", label=f"{sid}")
        axes[1].plot(it, [p["ty"] for p in pts], marker=".", label=f"{sid}")
    axes[0].set_ylabel("tx")
    axes[0].grid(True, alpha=0.3)
    ncol = max(1, min(int(num_samples), 8))
    axes[0].legend(title="sample", ncol=ncol, fontsize=8)
    axes[1].set_ylabel("ty")
    axes[1].set_xlabel("iteration")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, bbox_inches="tight", dpi=200)
    plt.close(fig)

    spatial_path = sample_dir / "affine_motion_spatial.png"
    last_by_sid = {}
    for row in sorted_rows:
        last_by_sid[int(row["sample_idx"])] = row
    fig2, ax = plt.subplots(figsize=(6, 6))
    sids = sorted(last_by_sid.keys())
    for i, sid in enumerate(sids):
        r = last_by_sid[sid]
        ax.scatter(
            [r["tx"]],
            [r["ty"]],
            color=plt.cm.tab10(i % 10),
            s=80,
            label=f"{sid}",
        )
        ax.annotate(
            str(sid), (r["tx"], r["ty"]), textcoords="offset points", xytext=(4, 4), fontsize=8
        )
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel("tx")
    ax.set_ylabel("ty")
    sp_title = (
        f"{title_prefix}: final per-frame translation"
        if title_prefix
        else "Final per-frame translation"
    )
    ax.set_title(sp_title, fontsize=11)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(title="sample", fontsize=8, ncol=min(len(sids), 4))
    plt.tight_layout()
    plt.savefig(spatial_path, bbox_inches="tight", dpi=200)
    plt.close(fig2)

    return csv_path, plot_path, spatial_path


def test_one_epoch(model, test_loader, device):
    model.eval()
    with torch.no_grad():
        hr_coords = test_loader.get_hr_coordinates().unsqueeze(0).to(device)
        hr_image = test_loader.get_original_hr().unsqueeze(0).to(device)
        sample_id = torch.tensor([0], device=device)

        if isinstance(model, INR):
            output, _ = model(hr_coords, sample_id, scale_factor=None, training=False)
        else:
            output, _ = model(
                hr_coords, sample_id, scale_factor=1, training=False, lr_frames=hr_image
            )
            output = output.reshape(hr_image.shape[1], hr_image.shape[2], 3).unsqueeze(0)

        output = output * test_loader.get_lr_std(0).to(device) + test_loader.get_lr_mean(0).to(
            device
        )
        loss = F.mse_loss(output, hr_image)
        psnr = -10 * torch.log10(loss.clamp_min(1e-12))
        return float(loss.item()), float(psnr.item())


def evaluate_lr_reprojection(model, dataset, device):
    # Deprecated: LR reprojection metric evaluation removed.
    _ = (model, dataset, device)
    return None


def _is_cuda_busy_or_unavailable(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "busy" in msg or "unavailable" in msg


def _parse_device(device_str: str) -> torch.device:
    s = str(device_str).strip().lower()
    if s == "cpu":
        return torch.device("cpu")
    if s.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        dev = torch.device(s if ":" in s else "cuda:0")
        try:
            torch.cuda.set_device(dev.index)
            return dev
        except RuntimeError as e:
            # Common on shared boxes: requested device is busy/unavailable.
            if "busy or unavailable" not in str(e).lower():
                raise

            n = torch.cuda.device_count()
            for idx in range(n):
                if idx == dev.index:
                    continue
                try:
                    torch.cuda.set_device(idx)
                    fallback = torch.device(f"cuda:{idx}")
                    print(f"Warning: {dev} unavailable; falling back to {fallback}")
                    return fallback
                except RuntimeError:
                    continue
            raise
    # accept plain index: "7"
    idx = int(s)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA index requested but torch.cuda.is_available() is False.")
    dev = torch.device(f"cuda:{idx}")
    try:
        torch.cuda.set_device(dev.index)
        return dev
    except RuntimeError as e:
        if "busy or unavailable" not in str(e).lower():
            raise
        n = torch.cuda.device_count()
        for j in range(n):
            if j == idx:
                continue
            try:
                torch.cuda.set_device(j)
                fallback = torch.device(f"cuda:{j}")
                print(f"Warning: cuda:{idx} unavailable; falling back to {fallback}")
                return fallback
            except RuntimeError:
                continue
        raise


def _probe_cuda_alloc(dev: torch.device) -> bool:
    """Return True if we can set the device and allocate a modest HR-sized tensor."""
    if dev.type != "cuda":
        return True
    try:
        torch.cuda.set_device(dev.index)
        torch.empty((), device=dev, dtype=torch.float32)
        # Larger than a scalar: catches "busy" that only appears on real transfers (e.g. SRData HR tensor).
        t = torch.empty(512, 512, 3, device=dev, dtype=torch.float32)
        del t
        return True
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "oom" in str(e).lower():
            raise
        return False


def _cuda_devices_to_try(preferred: torch.device) -> list[torch.device]:
    if preferred.type != "cuda":
        return []
    order = [preferred]
    for j in range(torch.cuda.device_count()):
        d = torch.device(f"cuda:{j}")
        if d.index != preferred.index:
            order.append(d)
    return order


def _resolve_training_device(device_str: str) -> torch.device:
    """
    Pick a usable device, probing CUDA with allocations (scalar + HR-sized chunk).

    Shared/multi-tenant GPUs often report "busy or unavailable" only on larger
    copies; try other GPU indices before falling back to CPU (non-CUDA workloads).
    """
    try:
        primary = _parse_device(device_str)
    except RuntimeError as e:
        if device_str.strip().lower() != "cpu" and _is_cuda_busy_or_unavailable(e):
            print(f"Warning: could not select CUDA ({e}). Using CPU.")
            return torch.device("cpu")
        raise

    if primary.type != "cuda":
        return primary

    for dev in _cuda_devices_to_try(primary):
        if _probe_cuda_alloc(dev):
            if dev.index != primary.index:
                print(f"Warning: {primary} failed allocation probe; using {dev} instead.")
            return dev

    print(
        "Warning: no CUDA device passed the allocation probe (busy or unavailable on all). "
        "Using CPU."
    )
    return torch.device("cpu")


def _first_usable_cuda_device(preferred_idx: int) -> int | None:
    """
    Return a CUDA device index we can set and allocate on, preferring ``preferred_idx``.

    Tries a tiny tensor allocation so "busy or unavailable" GPUs are skipped the same way
    as in ``_resolve_training_device``.
    """
    if not torch.cuda.is_available():
        return None
    n = torch.cuda.device_count()
    if n == 0:
        return None
    preferred_idx = int(preferred_idx)
    order = [preferred_idx] + [j for j in range(n) if j != preferred_idx]
    for idx in order:
        if idx < 0 or idx >= n:
            continue
        dev = torch.device(f"cuda:{idx}")
        try:
            torch.cuda.set_device(idx)
            torch.empty((), device=dev)
            return idx
        except RuntimeError:
            continue
    return None


def get_eval_autocast_dtype(mode: str, device: torch.device) -> torch.dtype | None:
    """Return autocast dtype for eval on CUDA, or ``None`` for full float32 (or CPU)."""
    m = (mode or "none").lower().strip()
    if m == "none" or device.type != "cuda":
        return None
    if m == "fp16":
        return torch.float16
    if m in ("bfloat16", "bf16"):
        return torch.bfloat16
    if m == "auto":
        try:
            major, _minor = torch.cuda.get_device_capability(device.index)
            if major >= 8 and getattr(torch.cuda, "is_bf16_supported", lambda: False)():
                return torch.bfloat16
        except Exception:
            pass
        return torch.float16
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Satellite super-resolution training (single sample)"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="satburst_synth",
        choices=[
            "satburst_synth",
            "worldstrat",
            "burst_synth",
            "worldstrat_test",
            "worldstrat_sweet",
            "worldstrat_bitter",
        ],
    )
    parser.add_argument("--sample_id", default="Landcover-743192_rgb")
    parser.add_argument("--df", type=int, default=4)
    parser.add_argument("--lr_shift", type=float, default=1.0)
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument(
        "--aug", type=str, default="none", choices=["none", "light", "medium", "heavy"]
    )
    parser.add_argument("--use_raw_b432", action="store_true")

    parser.add_argument("--model", type=str, default="mlp", choices=["mlp", "mlp_tcnn", "nir"])
    parser.add_argument("--network_depth", type=int, default=4)
    parser.add_argument("--network_hidden_dim", type=int, default=256)
    parser.add_argument("--mlp_init", type=str, default="kaiming", choices=["kaiming", "xavier"])
    parser.add_argument("--tcnn_mlp_dtype", type=str, default="fp16", choices=["fp16", "fp32"])

    parser.add_argument("--projection_dim", type=int, default=256)
    parser.add_argument(
        "--input_projection",
        type=str,
        default="fourier_10",
        choices=[
            "fourier_10",
            "fourier_5",
            "fourier_20",
            "fourier_40",
            "fourier",
            "hashgrid",
            "hash",
            "ngp_hash",
            "hashgrid_tcnn",
            "hash_tcnn",
            "ngp_hash_tcnn",
            "none",
            "None",
        ],
    )
    parser.add_argument("--fourier_scale", type=float, default=10.0)
    parser.add_argument("--hash_n_levels", type=int, default=16)
    parser.add_argument("--hash_n_features_per_level", type=int, default=2)
    parser.add_argument("--hash_log2_hashmap_size", type=int, default=19)
    parser.add_argument("--hash_base_resolution", type=int, default=16)
    parser.add_argument("--hash_max_resolution", type=int, default=64)
    parser.add_argument("--hash_encoding_dtype", type=str, default="fp32", choices=["fp16", "fp32"])
    parser.add_argument(
        "--hash_encoding_preset",
        type=str,
        default="hashgrid",
        choices=["hashgrid", "smoothstep_grid"],
    )
    parser.add_argument(
        "--hash_grid_type",
        type=str,
        default="Hash",
        choices=["Hash", "Dense", "Tiled", "hash", "dense", "tiled"],
    )
    parser.add_argument("--no_direct_param_T", action="store_true")

    parser.add_argument("--seed", type=int, default=6)
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw", "muon"])
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_eps", type=float, default=1e-8)
    parser.add_argument("--muon_momentum", type=float, default=0.95)
    parser.add_argument("--no_muon_nesterov", action="store_true")
    parser.add_argument("--muon_ns_steps", type=int, default=5)
    parser.add_argument("--muon_eps", type=float, default=1e-8)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--ttq_psnr_target", type=float, default=30.0)

    args = parser.parse_args()
    args.input_projection, args.fourier_scale = normalize_input_projection_name(
        args.input_projection, args.fourier_scale
    )
    device = _resolve_training_device(args.device)
    args.resolved_device = device

    if args.input_projection == "hashgrid_tcnn" and device.type != "cuda":
        raise RuntimeError("--input_projection hashgrid_tcnn requires CUDA (tinycudann).")
    if args.model == "mlp_tcnn" and args.optimizer.lower() == "muon":
        raise ValueError("--model mlp_tcnn is not supported with --optimizer muon.")
    _ip = str(args.input_projection).lower()
    if _ip in {"hashgrid", "hash", "ngp_hash", "hashgrid_tcnn", "hash_tcnn", "ngp_hash_tcnn"}:
        if str(args.optimizer).lower() == "muon":
            raise ValueError("Hash grid encoding is not supported with --optimizer muon.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.dataset == "satburst_synth":
        args.root_satburst_synth = (
            f"data/{args.sample_id}/scale_{args.df}_shift_{args.lr_shift:.1f}px_aug_{args.aug}"
        )
    elif args.dataset == "burst_synth":
        args.root_burst_synth = "SyntheticBurstVal"
        try:
            args.sample_id = int(args.sample_id)
        except ValueError:
            args.sample_id = 0

    train_data = get_dataset(args=args, name=args.dataset)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)

    input_projection = get_input_projection(
        args.input_projection,
        2,
        args.projection_dim,
        device,
        args.fourier_scale,
        hash_n_levels=args.hash_n_levels,
        hash_n_features_per_level=args.hash_n_features_per_level,
        hash_log2_hashmap_size=args.hash_log2_hashmap_size,
        hash_base_resolution=args.hash_base_resolution,
        hash_max_resolution=args.hash_max_resolution,
        hash_encoding_output_dtype=args.hash_encoding_dtype,
        hash_encoding_preset=args.hash_encoding_preset,
        hash_grid_type=args.hash_grid_type,
    )
    decoder_input_dim = (
        2
        if args.input_projection == "none"
        else getattr(input_projection, "projection_output_dim", args.projection_dim)
    )
    decoder = get_decoder(
        args.model,
        args.network_depth,
        decoder_input_dim,
        args.network_hidden_dim,
        output_dim=3,
        tcnn_mlp_dtype=args.tcnn_mlp_dtype,
        mlp_init=args.mlp_init,
        device=device,
    ).to(device)
    model = get_inr(input_projection, decoder, args.num_samples, use_gnll=False).to(device)

    args._optimizer_model = model
    optimizer = build_optimizer(model.parameters(), args)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.iters, eta_min=1e-6)

    print(f"Using device: {device}")
    print(f"Training for {args.iters} iterations…")

    iteration = 0
    progress_bar = tqdm(total=args.iters, desc="Training")
    it_list: list[int] = []
    psnr_list: list[float] = []
    recon_list: list[float] = []
    trans_list: list[float] = []
    total_list: list[float] = []
    train_wall_start = time.time()
    ttq_psnr_seconds = None
    ttq_psnr_iteration = None

    while iteration < args.iters:
        for batch in train_loader:
            if iteration >= args.iters:
                break
            losses = train_one_iteration(
                model, optimizer, batch, device, iteration=iteration, total_iterations=args.iters
            )
            scheduler.step()
            iteration += 1
            progress_bar.update(1)
            progress_bar.set_postfix(
                {
                    "recon": f"{losses['recon_loss']:.4f}",
                    "trans": f"{losses['trans_loss']:.4f}",
                    "valid": f"{losses.get('valid_fraction', 1.0):.2f}",
                }
            )

            if iteration % 100 == 0:
                test_loss, test_psnr = test_one_epoch(model, train_data, device)
                print(
                    f"\nIter {iteration}: train={losses['total_loss']:.6f} test={test_loss:.6f} psnr={test_psnr:.2f}dB"
                )
                it_list.append(iteration)
                psnr_list.append(test_psnr)
                recon_list.append(losses["recon_loss"])
                trans_list.append(losses["trans_loss"])
                total_list.append(losses["total_loss"])
                if ttq_psnr_seconds is None and test_psnr >= args.ttq_psnr_target:
                    ttq_psnr_seconds = time.time() - train_wall_start
                    ttq_psnr_iteration = iteration

    progress_bar.close()

    final_test_loss, final_test_psnr = test_one_epoch(model, train_data, device)
    print(f"Final: loss={final_test_loss:.6f} psnr={final_test_psnr:.2f}dB")
    if ttq_psnr_seconds is not None:
        print(
            f"TTQ {args.ttq_psnr_target}dB: iter={ttq_psnr_iteration} time={ttq_psnr_seconds:.1f}s"
        )

    out_root = Path("results")
    out_root.mkdir(exist_ok=True)
    sample_dir = out_root / f"{args.dataset}_{args.sample_id}"
    sample_dir.mkdir(exist_ok=True)

    model.eval()
    with torch.no_grad():
        hr_coords = train_data.get_hr_coordinates().unsqueeze(0).to(device)
        hr_image = train_data.get_original_hr().unsqueeze(0).to(device)
        sid0 = torch.tensor([0], device=device)

        if isinstance(model, INR):
            pred, _ = model(hr_coords, sid0, scale_factor=None, training=False)
        elif isinstance(model, NIR):
            pred, _ = model(hr_coords, sid0, scale_factor=1, training=False, lr_frames=hr_image)
            pred = pred.reshape(hr_image.shape[1], hr_image.shape[2], 3).unsqueeze(0)
        else:
            pred, _ = model(hr_coords, sid0, scale_factor=1, training=False, lr_frames=hr_image)
            pred = pred.reshape(hr_image.shape[1], hr_image.shape[2], 3).unsqueeze(0)

        pred = pred * train_data.get_lr_std(0).to(device) + train_data.get_lr_mean(0).to(device)
        pred_np = np.clip(pred.squeeze(0).cpu().numpy(), 0.0, 1.0)
        gt_np = np.clip(hr_image.squeeze(0).cpu().numpy(), 0.0, 1.0)

        if hasattr(train_data, "get_lr_sample_hwc"):
            lr_std_hwc = train_data.get_lr_sample_hwc(0).cpu().numpy()
            lr_std = train_data.get_lr_std(0).cpu().numpy().reshape(1, 1, -1)
            lr_mean = train_data.get_lr_mean(0).cpu().numpy().reshape(1, 1, -1)
            lr_np = lr_std_hwc * lr_std + lr_mean
        else:
            lr_any = train_data.get_lr_sample(0).cpu().numpy()
            lr_np = (
                lr_any.transpose(1, 2, 0) if (lr_any.ndim == 3 and lr_any.shape[0] == 3) else lr_any
            )

        lr_np = np.clip(lr_np[..., :3], 0.0, 1.0)
        hr_h, hr_w = gt_np.shape[:2]
        bilinear_np = np.clip(
            cv2.resize(lr_np, (hr_w, hr_h), interpolation=cv2.INTER_LINEAR), 0.0, 1.0
        )

        use_raw_b432 = bool(getattr(train_data, "use_raw_b432", False))
        stretch = (
            make_reflectance_display_stretch([gt_np, lr_np, bilinear_np, pred_np])
            if use_raw_b432
            else None
        )

        lr_disp = display_hwc(lr_np, use_raw_b432, stretch)
        bilinear_disp = display_hwc(bilinear_np, use_raw_b432, stretch)
        pred_disp = display_hwc(pred_np, use_raw_b432, stretch)
        gt_disp = display_hwc(gt_np, use_raw_b432, stretch)

        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes[0, 0].imshow(lr_disp)
        axes[0, 0].set_title("LR", fontsize=14, fontweight="bold")
        axes[0, 0].axis("off")
        axes[0, 1].imshow(bilinear_disp)
        axes[0, 1].set_title("Bilinear", fontsize=14, fontweight="bold")
        axes[0, 1].axis("off")
        axes[1, 0].imshow(pred_disp)
        axes[1, 0].set_title("Pred", fontsize=14, fontweight="bold")
        axes[1, 0].axis("off")
        axes[1, 1].imshow(gt_disp)
        axes[1, 1].set_title("GT", fontsize=14, fontweight="bold")
        axes[1, 1].axis("off")
        plt.tight_layout(pad=2.0)
        plt.savefig(sample_dir / "comparison.png", bbox_inches="tight", pad_inches=0.1, dpi=300)
        plt.close(fig)

        save_lr_reprojection_diagnostic(
            pred_hr_hwc=pred_np,
            lr_target_hwc=lr_np,
            df=args.df,
            out_path=sample_dir / "lr_reprojection_diagnostic.png",
            use_raw_b432=use_raw_b432,
        )

    if it_list:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        ax1.plot(it_list, psnr_list, color="blue", linewidth=2)
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("PSNR (dB)")
        ax1.grid(True, alpha=0.3)
        ax2.plot(it_list, recon_list, color="red", linewidth=2, label="recon")
        ax2.plot(it_list, trans_list, color="green", linewidth=2, label="trans")
        ax2.plot(it_list, total_list, color="purple", linewidth=2, label="total")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Loss")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        plt.tight_layout()
        plt.savefig(
            sample_dir / "training_metrics.png", bbox_inches="tight", pad_inches=0.1, dpi=300
        )
        plt.close(fig)

    # LR reprojection metric evaluation removed (was extra/slow/confusing).


def test_one_epoch(model, test_loader, device, eval_autocast_dtype=None):
    model.eval()

    with torch.no_grad():
        hr_coords = test_loader.get_hr_coordinates().unsqueeze(0).to(device)
        hr_image = test_loader.get_original_hr().unsqueeze(0).to(device)
        sample_id = torch.tensor([0]).to(device)

        if eval_autocast_dtype is not None:
            with torch.autocast(device_type="cuda", dtype=eval_autocast_dtype):
                if isinstance(model, INR):
                    output, _ = model(hr_coords, sample_id, scale_factor=1, training=False)
                elif isinstance(model, NIR):
                    output, _ = model(
                        hr_coords, sample_id, scale_factor=1, training=False, lr_frames=hr_image
                    )
                    output = output.reshape(hr_image.shape[1], hr_image.shape[2], 3).unsqueeze(0)
        else:
            if isinstance(model, INR):
                output, _ = model(hr_coords, sample_id, scale_factor=1, training=False)
            elif isinstance(model, NIR):
                output, _ = model(
                    hr_coords, sample_id, scale_factor=1, training=False, lr_frames=hr_image
                )
                output = output.reshape(hr_image.shape[1], hr_image.shape[2], 3).unsqueeze(0)

        # Unstandardize the output
        output = output * test_loader.get_lr_std(0).to(device) + test_loader.get_lr_mean(0).to(
            device
        )

        loss = F.mse_loss(output, hr_image)

        # Calculate PSNR
        psnr = -10 * torch.log10(loss)

    return loss.item(), psnr.item()


def evaluate_lr_reprojection(
    model,
    dataset,
    device,
):
    # Deprecated: LR reprojection metric evaluation removed.
    _ = (model, dataset, device)
    return None


def _variance_summary_title_for_path(path: Path) -> str:
    parts = path.stem.split("_")
    if len(parts) >= 2 and parts[0] == "sample":
        try:
            return f"Sample {int(parts[1])}"
        except ValueError:
            pass
    return path.stem


def _load_std_hw_map_from_npy(path: Path) -> np.ndarray | None:
    try:
        arr = np.load(path)
    except Exception:
        return None
    a = np.asarray(arr, dtype=np.float64)
    if a.size == 0:
        return None
    if a.ndim > 3:
        a = np.squeeze(a)
    if a.ndim == 3:
        return a.mean(axis=-1)
    if a.ndim == 2:
        return a
    return None


def create_variance_summary(train_data, variance_dir: Path, device) -> None:
    """Montage all per-sample std maps found under ``variance_dir`` with one colorbar.

    Prefers ``sample_*_std.npy``; falls back to ``sample_*_variance.npy`` (sqrt to std).
    ``train_data`` and ``device`` are kept for call-site compatibility.
    """
    _ = train_data
    _ = device
    variance_dir = Path(variance_dir)
    if not variance_dir.is_dir():
        return

    maps: list[tuple[Path, np.ndarray]] = []
    for path in sorted(variance_dir.glob("sample_*_std.npy")):
        m = _load_std_hw_map_from_npy(path)
        if m is not None and m.ndim == 2:
            maps.append((path, m))

    if not maps:
        for path in sorted(variance_dir.glob("sample_*_variance.npy")):
            try:
                v = np.maximum(np.asarray(np.load(path), dtype=np.float64), 0.0)
            except Exception:
                continue
            a = np.sqrt(v)
            if a.ndim > 3:
                a = np.squeeze(a)
            if a.ndim == 3:
                m = a.mean(axis=-1)
            elif a.ndim == 2:
                m = a
            else:
                m = None
            if m is not None and m.ndim == 2:
                maps.append((path, m))

    if not maps:
        return

    vmin = max(0.0, min(float(m.min()) for _, m in maps))
    vmax = max(float(m.max()) for _, m in maps)
    if not np.isfinite(vmin):
        vmin = 0.0
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1e-6

    n = len(maps)
    ncols = min(8, n)
    nrows = int(np.ceil(n / ncols))
    fig_w = max(10.0, ncols * 2.0)
    fig_h = max(3.0, nrows * 2.4)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)

    last_im = None
    for idx, (path, m) in enumerate(maps):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        last_im = ax.imshow(m, cmap="Blues", vmin=vmin, vmax=vmax)
        ax.set_title(_variance_summary_title_for_path(path), fontsize=9)
        ax.axis("off")

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].axis("off")

    if last_im is not None:
        fig.colorbar(
            last_im,
            ax=axes.ravel().tolist(),
            fraction=0.035,
            pad=0.02,
            label="Standard deviation",
        )
    plt.tight_layout()
    plt.savefig(variance_dir / "variance_summary.png", bbox_inches="tight", pad_inches=0.1, dpi=200)
    plt.close(fig)


def create_lr_variance_grid(lr_list, std_list, vmin, vmax, out_dir: Path) -> None:
    """2×8 grid: top row LR thumbnails, bottom row matching std maps (Blues, shared vmin/vmax)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ncols = 8
    fig, axes = plt.subplots(2, ncols, figsize=(ncols * 2.0, 5.0), squeeze=False)

    std_slice = list(std_list[:ncols]) if std_list is not None else []
    lr_slice = list(lr_list[:ncols]) if lr_list is not None else []

    if vmin is None or vmax is None:
        mats = []
        for s in std_slice:
            if s is None:
                continue
            a = np.asarray(s, dtype=np.float64)
            if a.ndim == 3:
                a = a.mean(axis=-1)
            if a.ndim == 2 and a.size:
                mats.append(a)
        if mats:
            vmin = max(0.0, min(float(m.min()) for m in mats))
            vmax = max(float(m.max()) for m in mats)
        else:
            vmin, vmax = 0.0, 1.0
    if not np.isfinite(vmin):
        vmin = 0.0
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1e-6

    im_for_cbar = None
    for j in range(ncols):
        ax_top = axes[0, j]
        ax_bot = axes[1, j]

        lr = lr_slice[j] if j < len(lr_slice) else None
        if lr is not None:
            lr_img = np.asarray(lr, dtype=np.float64)
            if lr_img.ndim == 2:
                lr_img = np.repeat(lr_img[..., None], 3, axis=-1)
            elif lr_img.shape[-1] == 1:
                lr_img = np.repeat(lr_img, 3, axis=-1)
            lr_img = np.clip(lr_img, 0.0, 1.0)
            ax_top.imshow(lr_img)
        else:
            ax_top.set_facecolor((0.9, 0.9, 0.9))
            ax_top.text(
                0.5,
                0.5,
                "No LR",
                ha="center",
                va="center",
                transform=ax_top.transAxes,
                fontsize=10,
                color="0.35",
            )
        ax_top.axis("off")

        std_j = std_slice[j] if j < len(std_slice) else None
        if std_j is not None:
            s = np.asarray(std_j, dtype=np.float64)
            if s.ndim == 3:
                s = s.mean(axis=-1)
            if s.ndim == 2 and s.size:
                im_for_cbar = ax_bot.imshow(s, cmap="Blues", vmin=vmin, vmax=vmax)
            else:
                ax_bot.set_facecolor((0.92, 0.92, 0.92))
        else:
            ax_bot.set_facecolor((0.92, 0.92, 0.92))
            ax_bot.text(
                0.5,
                0.5,
                "—",
                ha="center",
                va="center",
                transform=ax_bot.transAxes,
                fontsize=14,
                color="0.5",
            )
        ax_bot.axis("off")

    if im_for_cbar is not None:
        fig.colorbar(
            im_for_cbar,
            ax=axes[1, :].ravel().tolist(),
            fraction=0.035,
            pad=0.03,
            label="Standard deviation",
        )
    plt.tight_layout()
    plt.savefig(out_dir / "lr_variance_grid.png", bbox_inches="tight", pad_inches=0.1, dpi=250)
    plt.close(fig)


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
            axes[0, 0].set_title("Ground Truth HR", fontsize=12, fontweight="bold")
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
                    "Standard Deviation Statistics:",
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


def _metrics_row_from_sample_result(
    r: dict,
) -> tuple[float, float, float, float, float, float, float]:
    """Unpack one per-sample metrics dict (nested like metrics.json or flat keys)."""
    if "psnr" in r and isinstance(r["psnr"], dict):
        p = r["psnr"]
        s = r.get("ssim") or {}
        lp = r.get("lpips") or {}
        t = r.get("training") or {}
        return (
            float(p["model"]),
            float(p["bilinear"]),
            float(s["model"]),
            float(s["bilinear"]),
            float(lp["model"]),
            float(lp["bilinear"]),
            float(t.get("final_trans_loss", 0.0)),
        )
    return (
        float(r["model_psnr"]),
        float(r["bilinear_psnr"]),
        float(r["model_ssim"]),
        float(r["bilinear_ssim"]),
        float(r.get("model_lpips", r.get("pred_lpips", 0.0))),
        float(r["bilinear_lpips"]),
        float(r.get("final_trans_loss", r.get("trans_loss_final", 0.0))),
    )


def create_summary_visualization(all_results: list, output_dir: str | Path) -> None:
    """Aggregate metrics from multi-sample runs and write summary plots + JSON/text reports."""
    if not all_results:
        print("create_summary_visualization: no results; skipping summary.")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [_metrics_row_from_sample_result(r) for r in all_results]
    model_psnr = [row[0] for row in rows]
    bilinear_psnr = [row[1] for row in rows]
    model_ssim = [row[2] for row in rows]
    bilinear_ssim = [row[3] for row in rows]
    model_lpips = [row[4] for row in rows]
    bilinear_lpips = [row[5] for row in rows]
    trans_loss_values = [row[6] for row in rows]

    sample_indices = list(range(len(all_results)))
    psnr_improvement = [m - b for m, b in zip(model_psnr, bilinear_psnr)]
    lpips_improvement = [b - m for m, b in zip(model_lpips, bilinear_lpips)]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

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

    ssim_improvement = [m - b for m, b in zip(model_ssim, bilinear_ssim)]

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


def optimize_and_evaluate_sample(
    model,
    train_data,
    device: torch.device,
    sample_idx: int,
    args,
    output_dir: str | Path,
) -> dict:
    """
    Train one INR for a single dataset sample, save artifacts under output_dir, and return
    a metrics dict compatible with _metrics_row_from_sample_result / create_summary_visualization.
    """
    output_dir = Path(output_dir)
    sample_dir = output_dir / f"sample_{sample_idx:03d}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)
    args._optimizer_model = model
    optimizer = build_optimizer(model.parameters(), args)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.iters, eta_min=1e-6)

    print(f"Starting training for {args.iters} iterations (sample {sample_idx})...")
    train_wall_start = time.time()
    ttq_psnr_seconds = None
    ttq_psnr_iteration = None

    iteration = 0
    progress_bar = tqdm(total=args.iters, desc=f"Train sample {sample_idx}")
    psnr_list: list[float] = []
    recon_loss_list: list[float] = []
    trans_loss_list: list[float] = []
    total_loss_list: list[float] = []
    iteration_list: list[int] = []

    while iteration < args.iters:
        for train_sample in train_dataloader:
            if iteration >= args.iters:
                break

            train_losses = train_one_iteration(
                model,
                optimizer,
                train_sample,
                device,
                iteration=iteration,
                total_iterations=args.iters,
            )

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
            progress_bar.update(1)
            progress_bar.set_postfix(
                {
                    "recon": f"{train_losses['recon_loss']:.4f}",
                    "trans": f"{train_losses['trans_loss']:.4f}",
                    "valid": f"{train_losses.get('valid_fraction', 1.0):.2f}",
                }
            )

            if iteration % 100 == 0:
                eval_autocast_dtype = get_eval_autocast_dtype(args.eval_mixed_precision, device)
                test_loss, test_psnr = test_one_epoch(
                    model, train_data, device, eval_autocast_dtype
                )
                print(
                    f"\nIter {iteration}: Train Loss: {train_losses['total_loss']:.6f}, "
                    f"Test Loss: {test_loss:.6f}, Test PSNR: {test_psnr:.2f} dB"
                )
                if ttq_psnr_seconds is None and test_psnr >= args.ttq_psnr_target:
                    ttq_psnr_seconds = time.time() - train_wall_start
                    ttq_psnr_iteration = iteration
                    print(
                        f"TTQ reached: PSNR {test_psnr:.2f} dB >= {args.ttq_psnr_target:.2f} dB "
                        f"at iter {iteration} ({ttq_psnr_seconds:.1f}s)"
                    )

                iteration_list.append(iteration)
                psnr_list.append(test_psnr)
                recon_loss_list.append(train_losses["recon_loss"])
                trans_loss_list.append(train_losses["trans_loss"])
                total_loss_list.append(train_losses["total_loss"])

    progress_bar.close()

    model.eval()
    eval_autocast_dtype = get_eval_autocast_dtype(args.eval_mixed_precision, device)
    with torch.no_grad():
        hr_coords = train_data.get_hr_coordinates().unsqueeze(0).to(device)
        hr_image = train_data.get_original_hr().unsqueeze(0).to(device)
        sample_id_tensor = torch.tensor([0]).to(device)

        if eval_autocast_dtype is not None:
            with torch.autocast(device_type="cuda", dtype=eval_autocast_dtype):
                output, _ = model(hr_coords, sample_id_tensor, scale_factor=1, training=False)
        else:
            output, _ = model(hr_coords, sample_id_tensor, scale_factor=1, training=False)

        output = output * train_data.get_lr_std(0).to(device) + train_data.get_lr_mean(0).to(device)

        final_test_loss = F.mse_loss(output, hr_image).item()
        final_psnr = -10 * torch.log10(torch.tensor(final_test_loss)).item()

        pred_np = output.squeeze().cpu().numpy()
        gt_np = hr_image.squeeze().cpu().numpy()

        if hasattr(train_data, "get_lr_sample_hwc"):
            lr_original = train_data.get_lr_sample_hwc(0).cpu().numpy()
            lr_std = train_data.get_lr_std(0).cpu().numpy()
            lr_mean = train_data.get_lr_mean(0).cpu().numpy()
            if lr_std.ndim == 1:
                lr_std = lr_std.reshape(1, 1, -1)
            if lr_mean.ndim == 1:
                lr_mean = lr_mean.reshape(1, 1, -1)
            lr_original = lr_original * lr_std + lr_mean
        else:
            lr_original = train_data.get_lr_sample(0).cpu().numpy()
            if lr_original.ndim == 3:
                if lr_original.shape[0] in (1, 3, 4):
                    lr_original = lr_original.transpose(1, 2, 0)
                    if lr_original.shape[2] > 3:
                        H, W, C = lr_original.shape
                        if C % 3 == 0:
                            T = C // 3
                            lr_original = lr_original.reshape(H, W, T, 3)
                            lr_original = lr_original[:, :, 0, :]
                        else:
                            lr_original = lr_original[:, :, :3]

        hr_h, hr_w = gt_np.shape[:2]
        lr_bilinear = cv2.resize(lr_original, (hr_w, hr_h), interpolation=cv2.INTER_LINEAR)
        pred_np = np.clip(pred_np, 0, 1)
        gt_np = np.clip(gt_np, 0, 1)
        lr_original = np.clip(lr_original, 0, 1)
        lr_bilinear = np.clip(lr_bilinear, 0, 1)

        pred_tensor = torch.from_numpy(pred_np).unsqueeze(0).permute(0, 3, 1, 2).to(device)
        gt_tensor = torch.from_numpy(gt_np).unsqueeze(0).permute(0, 3, 1, 2).to(device)
        bilinear_tensor = torch.from_numpy(lr_bilinear).unsqueeze(0).permute(0, 3, 1, 2).to(device)

        pred_eval = pred_tensor
        bilinear_eval = bilinear_tensor

        model_psnr = peak_signal_noise_ratio(
            pred_eval.cpu(), gt_tensor.cpu(), data_range=1.0
        ).item()
        bilinear_psnr = peak_signal_noise_ratio(
            bilinear_eval.cpu(), gt_tensor.cpu(), data_range=1.0
        ).item()

        model_ssim = ssim(pred_eval.cpu(), gt_tensor.cpu(), data_range=1.0).item()
        bilinear_ssim = ssim(bilinear_eval.cpu(), gt_tensor.cpu(), data_range=1.0).item()

        lpips_fn = lpips.LPIPS(net="vgg").to(device)
        pred_lpips = lpips_fn((pred_eval * 2 - 1).to(device), (gt_tensor * 2 - 1).to(device)).item()
        bilinear_lpips = lpips_fn(
            (bilinear_eval * 2 - 1).to(device), (gt_tensor * 2 - 1).to(device)
        ).item()

        pred_eval_np = pred_eval.squeeze(0).permute(1, 2, 0).cpu().numpy()
        bilinear_eval_np = bilinear_eval.squeeze(0).permute(1, 2, 0).cpu().numpy()

        pred_eval_np = np.clip(pred_eval_np, 0, 1)
        bilinear_eval_np = np.clip(bilinear_eval_np, 0, 1)

        use_raw_b432 = bool(getattr(train_data, "use_raw_b432", False))
        reflectance_stretch = (
            make_reflectance_display_stretch([gt_np, lr_original, bilinear_eval_np, pred_eval_np])
            if use_raw_b432
            else None
        )
        pred_display_np = display_hwc(pred_eval_np, use_raw_b432, reflectance_stretch)
        bilinear_display_np = display_hwc(bilinear_eval_np, use_raw_b432, reflectance_stretch)
        gt_display_np = display_hwc(gt_np, use_raw_b432, reflectance_stretch)
        lr_display_np = display_hwc(lr_original, use_raw_b432, reflectance_stretch)

        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes[0, 0].imshow(lr_display_np)
        axes[0, 0].set_title("Original LR Image", fontsize=14, fontweight="bold")
        axes[0, 0].axis("off")
        axes[0, 1].imshow(bilinear_display_np)
        axes[0, 1].set_title(
            f"Bilinear Upsampling\nPSNR: {bilinear_psnr:.2f} dB",
            fontsize=14,
            fontweight="bold",
        )
        axes[0, 1].axis("off")
        axes[1, 0].imshow(pred_display_np)
        axes[1, 0].set_title(
            f"Model Output\nPSNR: {model_psnr:.2f} dB", fontsize=14, fontweight="bold"
        )
        axes[1, 0].axis("off")
        axes[1, 1].imshow(gt_display_np)
        axes[1, 1].set_title("Ground Truth HR", fontsize=14, fontweight="bold")
        axes[1, 1].axis("off")
        plt.tight_layout(pad=2.0)
        plt.savefig(sample_dir / "comparison.png", bbox_inches="tight", pad_inches=0.1, dpi=300)
        plt.close()

        save_lr_reprojection_diagnostic(
            pred_hr_hwc=pred_eval_np,
            lr_target_hwc=lr_original,
            df=args.df,
            out_path=sample_dir / "lr_reprojection_diagnostic.png",
            use_raw_b432=use_raw_b432,
        )

        for arr, fname in (
            (pred_display_np, "model_output.png"),
            (gt_display_np, "ground_truth.png"),
            (bilinear_display_np, "bilinear_baseline.png"),
            (lr_display_np, "lr_original.png"),
        ):
            plt.figure(figsize=(8, 8))
            plt.imshow(arr)
            plt.axis("off")
            plt.tight_layout(pad=0)
            plt.savefig(sample_dir / fname, bbox_inches="tight", pad_inches=0, dpi=300)
            plt.close()

    if len(psnr_list) > 0:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        ax1.plot(iteration_list, psnr_list, color="blue", linewidth=2, label="PSNR (Test)")
        ax1.set_xlabel("Iteration", fontsize=12)
        ax1.set_ylabel("PSNR (dB)", fontsize=12)
        ax1.set_title("Training PSNR Evolution", fontsize=14, fontweight="bold")
        ax1.grid(True, alpha=0.3)
        ax1.legend()
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
        plt.savefig(
            sample_dir / "training_metrics.png", bbox_inches="tight", pad_inches=0.1, dpi=300
        )
        plt.close()

    metrics_dict = {
        "dataset": args.dataset,
        "sample_id": str(args.sample_id),
        "multi_sample_index": sample_idx,
        "downsampling_factor": args.df,
        "model": args.model,
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
            "ttq_psnr_target": args.ttq_psnr_target,
            "ttq_psnr_seconds": ttq_psnr_seconds,
            "ttq_psnr_iteration": ttq_psnr_iteration,
            "artifact_dir": str(sample_dir),
        },
    }

    with open(sample_dir / "metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=2)

    print(
        f"Sample {sample_idx} ({args.sample_id}): PSNR model={model_psnr:.2f} dB, "
        f"bilinear={bilinear_psnr:.2f} dB — saved under {sample_dir}"
    )

    return metrics_dict


def main_legacy():
    parser = argparse.ArgumentParser(description="Legacy/unused entrypoint (do not use)")

    # Essential parameters only
    parser.add_argument(
        "--dataset",
        type=str,
        default="satburst_synth",
        choices=[
            "satburst_synth",
            "worldstrat",
            "burst_synth",
            "worldstrat_test",
            "worldstrat_sweet",
            "worldstrat_bitter",
        ],
    )
    parser.add_argument("--sample_id", default="Landcover-743192_rgb")
    parser.add_argument(
        "--df", type=int, default=4, help="Downsampling factor, or upsampling factor for the data"
    )
    parser.add_argument(
        "--scale_factor", type=float, default=4, help="scale factor for the input training grid"
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

    parser.add_argument("--lr_shift", type=float, default=1.0)
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument(
        "--aug", type=str, default="none", choices=["none", "light", "medium", "heavy"]
    )
    parser.add_argument(
        "--use_raw_b432",
        action="store_true",
        help=(
            "For satburst_synth exports with sample_XX_raw.npz, train/evaluate on raw "
            "Sentinel-2 B4/B3/B2 reflectance instead of display RGB PNGs."
        ),
    )

    # Model parameters
    parser.add_argument("--model", type=str, default="mlp", choices=["mlp", "mlp_tcnn", "nir"])
    parser.add_argument("--network_depth", type=int, default=4)
    parser.add_argument("--network_hidden_dim", type=int, default=256)
    parser.add_argument("--mlp_init", type=str, default="kaiming", choices=["kaiming", "xavier"])
    parser.add_argument(
        "--tcnn_mlp_dtype",
        type=str,
        default="fp16",
        choices=["fp16", "fp32"],
        help="Compute dtype for --model mlp_tcnn.",
    )
    parser.add_argument("--projection_dim", type=int, default=256)
    parser.add_argument(
        "--input_projection",
        type=str,
        default="fourier_10",
        choices=[
            "fourier_10",
            "fourier_5",
            "fourier_20",
            "fourier_40",
            "fourier",
            "hashgrid",
            "hash",
            "ngp_hash",
            "hashgrid_tcnn",
            "hash_tcnn",
            "ngp_hash_tcnn",
            "none",
            "None",
        ],
    )
    parser.add_argument("--fourier_scale", type=float, default=10.0)
    parser.add_argument("--hash_n_levels", type=int, default=16)
    parser.add_argument("--hash_n_features_per_level", type=int, default=2)
    parser.add_argument("--hash_log2_hashmap_size", type=int, default=19)
    parser.add_argument("--hash_base_resolution", type=int, default=16)
    parser.add_argument(
        "--hash_max_resolution",
        type=int,
        default=2048,
        help=(
            "Target finest grid resolution for multires hash: sets per_level_scale from "
            "base_resolution and n_levels (HashGrid preset). For --hash_encoding_preset "
            "smoothstep_grid this uses the same --hash_base_resolution and --hash_n_levels; "
            "try 4096 or 8192 if detail is lacking."
        ),
    )
    parser.add_argument(
        "--hash_encoding_dtype",
        type=str,
        default="fp32",
        choices=["fp16", "fp32"],
        help="Output dtype of the tcnn hash encoding (decoder still uses --model / --tcnn_mlp_dtype).",
    )
    parser.add_argument(
        "--hash_encoding_preset",
        type=str,
        default="hashgrid",
        choices=["hashgrid", "smoothstep_grid"],
        help=(
            "hashgrid: HashGrid (Linear) with --hash_* hyperparameters. "
            "smoothstep_grid: Grid+Hash with Smoothstep using the same --hash_* hyperparameters. "
            "Use e.g. --network_depth 3 --network_hidden_dim 64 with --model mlp_tcnn."
        ),
    )
    parser.add_argument(
        "--hash_grid_type",
        type=str,
        default="Hash",
        choices=["Hash", "Dense", "Tiled", "hash", "dense", "tiled"],
        help=(
            "Grid backing type for --hash_encoding_preset smoothstep_grid. "
            "Hash uses log2_hashmap_size; Dense avoids collisions but can use much more memory; "
            "Tiled repeats a dense grid periodically."
        ),
    )
    parser.add_argument(
        "--no_direct_param_T",
        action="store_true",
        help="Disable direct parameter T (default: use_direct_param_T=True)",
    )

    # Training parameters
    parser.add_argument("--seed", type=int, default=6)
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Training batch size for DataLoader."
    )
    parser.add_argument("--learning_rate", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw", "muon"])
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_eps", type=float, default=1e-8)
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
        "--ttq_psnr_target",
        type=float,
        default=30.0,
        help="Target PSNR (dB) for time-to-quality logging.",
    )

    args = parser.parse_args()

    # Accept common CUDA device spellings like "cuda:7" in addition to "7".
    if isinstance(args.device, str) and args.device.lower().startswith("cuda:"):
        args.device = args.device.split(":", 1)[1]

    if args.model == "mlp_tcnn" and args.optimizer.lower() == "muon":
        raise ValueError("Model 'mlp_tcnn' is currently supported only with --optimizer adamw.")

    _ip = str(args.input_projection).lower()
    if _ip in {"hashgrid", "hash", "ngp_hash", "hashgrid_tcnn", "hash_tcnn", "ngp_hash_tcnn"}:
        if str(args.optimizer).lower() == "muon":
            raise ValueError(
                "Hash grid encoding uses non-matrix tcnn parameters; use --optimizer adam or adamw, not muon."
            )

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

        usable_idx = _first_usable_cuda_device(requested_idx)
        if usable_idx is None:
            print("Warning: no usable CUDA device found (all busy/unavailable). Using CPU.")
            device = torch.device("cpu")
        else:
            if usable_idx != requested_idx:
                print(
                    f"Warning: requested CUDA device {requested_idx} is busy/unavailable. "
                    f"Falling back to cuda:{usable_idx}."
                )
            device = torch.device(f"cuda:{usable_idx}")
    else:
        print(f"Warning: CUDA device {args.device} requested but CUDA not available. Using CPU.")
        device = torch.device("cpu")

    print(f"Using device: {device}")
    args.resolved_device = device
    # tinycudann (hash grid) allocates on the *current* CUDA device during module __init__;
    # set it explicitly so --device N matches tcnn, not default cuda:0.
    if device.type == "cuda":
        try:
            torch.cuda.set_device(device.index)
        except RuntimeError as e:
            print(
                f"Warning: failed to set CUDA device cuda:{device.index} ({e}). "
                "Searching for another usable device..."
            )
            fallback_idx = _first_usable_cuda_device(0)
            if fallback_idx is None:
                print("Warning: no usable CUDA device found. Falling back to CPU.")
                device = torch.device("cpu")
                args.resolved_device = device
            else:
                device = torch.device(f"cuda:{fallback_idx}")
                args.resolved_device = device
                torch.cuda.set_device(device.index)
                print(f"Using device: {device}")
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

    args.input_projection, args.fourier_scale = normalize_input_projection_name(
        args.input_projection, args.fourier_scale
    )
    if args.input_projection == "hashgrid_tcnn" and device.type == "cpu":
        raise RuntimeError(
            "--input_projection hashgrid_tcnn requires CUDA (tinycudann); got CPU device."
        )

    # Setup dataset
    if args.dataset == "satburst_synth":
        args.root_satburst_synth = (
            f"data/{args.sample_id}/scale_{args.df}_shift_{args.lr_shift:.1f}px_aug_{args.aug}"
        )
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
        if args.dataset in ["worldstrat_test", "worldstrat_sweet", "worldstrat_bitter"]:
            # For worldstrat_test, we need to get all sample IDs
            if args.dataset == "worldstrat_test":
                data_root = "worldstrat_test_data"
            elif args.dataset == "worldstrat_sweet":
                data_root = "worldstrat_datasets/worldstrat_sweet"
            else:
                data_root = "worldstrat_datasets/worldstrat_bitter"
            # Hint to downstream loaders which root to use (if supported)
            os.environ["WORLDSTRAT_TEST_ROOT"] = str(data_root)
            sample_dirs = [d for d in Path(data_root).iterdir() if d.is_dir()]
            sample_ids = [d.name for d in sample_dirs]
            print(f"Found {len(sample_ids)} samples: {sample_ids[:5]}...")
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
        elif args.dataset == "satburst_synth":
            # For satburst_synth, each sample is a directory inside data/
            data_root = Path("data")
            if not data_root.exists():
                print(f"Error: data directory {data_root} not found!")
                return
            sample_dirs = [
                d for d in data_root.iterdir() if d.is_dir() and not d.name.startswith(".")
            ]
            sample_ids = [d.name for d in sample_dirs]
            sample_ids.sort()
            print(f"Found {len(sample_ids)} samples: {sample_ids[:5]}...")
        else:
            print(f"Error: Unsupported dataset for multi-sample: {args.dataset}")
            return

        output_dim = 3
        # Setup model components (needed for all samples)
        input_projection = get_input_projection(
            args.input_projection,
            2,
            args.projection_dim,
            device,
            args.fourier_scale,
            hash_n_levels=args.hash_n_levels,
            hash_n_features_per_level=args.hash_n_features_per_level,
            hash_log2_hashmap_size=args.hash_log2_hashmap_size,
            hash_base_resolution=args.hash_base_resolution,
            hash_max_resolution=args.hash_max_resolution,
            hash_encoding_output_dtype=args.hash_encoding_dtype,
            hash_encoding_preset=args.hash_encoding_preset,
            hash_grid_type=args.hash_grid_type,
        )
        decoder_input_dim = (
            2
            if args.input_projection == "none"
            else getattr(input_projection, "projection_output_dim", args.projection_dim)
        )
        decoder = get_decoder(
            args.model,
            args.network_depth,
            decoder_input_dim,
            args.network_hidden_dim,
            output_dim=output_dim,
            tcnn_mlp_dtype=args.tcnn_mlp_dtype,
            mlp_init=args.mlp_init,
            device=device,
        )

        # Run optimization for each sample
        all_results = []
        for sample_idx, sample_id in enumerate(sample_ids):
            print(f"\n{'='*60}")
            print(f"Processing sample {sample_idx + 1}/{len(sample_ids)}: {sample_id}")
            print(f"{'='*60}")

            # Create a FRESH model for each sample (this is the key fix!)
            print(
                f"🔄 Creating fresh model for sample {sample_id} (sample {sample_idx + 1}/{len(sample_ids)})"
            )
            model = get_inr(
                input_projection,
                decoder,
                args.num_samples,
                use_gnll=False,
            ).to(device)
            print("✅ Fresh model created and initialized")

            # Set the sample_id for this iteration
            args.sample_id = sample_id
            # Recompute dataset-specific roots per sample when needed
            if args.dataset == "satburst_synth":
                args.root_satburst_synth = f"data/{args.sample_id}/scale_{args.df}_shift_{args.lr_shift:.1f}px_aug_{args.aug}"

            # Get dataset for this specific sample
            # Treat worldstrat_sweet/bitter like worldstrat_test for loader name
            dataset_name_for_loader = args.dataset
            if args.dataset in ["worldstrat_sweet", "worldstrat_bitter"]:
                dataset_name_for_loader = "worldstrat_test"
            train_data = get_dataset(args=args, name=dataset_name_for_loader)

            # Run optimization for this sample with the fresh model
            result = optimize_and_evaluate_sample(
                model, train_data, device, sample_idx, args, output_dir
            )
            all_results.append(result)

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

    input_projection = get_input_projection(
        args.input_projection,
        2,
        args.projection_dim,
        device,
        args.fourier_scale,
        hash_n_levels=args.hash_n_levels,
        hash_n_features_per_level=args.hash_n_features_per_level,
        hash_log2_hashmap_size=args.hash_log2_hashmap_size,
        hash_base_resolution=args.hash_base_resolution,
        hash_max_resolution=args.hash_max_resolution,
        hash_encoding_output_dtype=args.hash_encoding_dtype,
        hash_encoding_preset=args.hash_encoding_preset,
        hash_grid_type=args.hash_grid_type,
    )
    decoder_input_dim = (
        2
        if args.input_projection == "none"
        else getattr(input_projection, "projection_output_dim", args.projection_dim)
    )
    output_dim = 3
    decoder = get_decoder(
        args.model,
        args.network_depth,
        decoder_input_dim,
        args.network_hidden_dim,
        output_dim=output_dim,
        tcnn_mlp_dtype=args.tcnn_mlp_dtype,
        mlp_init=args.mlp_init,
        device=device,
    )
    model = get_inr(
        input_projection,
        decoder,
        args.num_samples,
        use_gnll=False,
    ).to(device)
    # model = NIR(input_projection, decoder, args.num_samples, use_gnll=False).to(device)

    # Setup optimizer
    args._optimizer_model = model
    optimizer = build_optimizer(model.parameters(), args)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.iters, eta_min=1e-6)

    print(f"Starting training for {args.iters} iterations...")
    train_wall_start = time.time()
    ttq_psnr_seconds = None
    ttq_psnr_iteration = None

    # Training loop
    iteration = 0
    progress_bar = tqdm(total=args.iters, desc="Training")

    # Lists to store PSNR and losses for plotting
    psnr_list = []
    recon_loss_list = []
    trans_loss_list = []
    total_loss_list = []
    iteration_list = []
    affine_motion_rows = []
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
                iteration=iteration,
                total_iterations=args.iters,
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
                "valid": f"{train_losses.get('valid_fraction', 1.0):.2f}",
            }
            progress_bar.set_postfix(postfix_dict)

            # Periodic evaluation
            if iteration % 100 == 0:
                eval_autocast_dtype = get_eval_autocast_dtype(args.eval_mixed_precision, device)
                test_loss, test_psnr = test_one_epoch(
                    model, train_data, device, eval_autocast_dtype
                )
                print(
                    f"\nIter {iteration}: Train Loss: {train_losses['total_loss']:.6f}, "
                    f"Test Loss: {test_loss:.6f}, Test PSNR: {test_psnr:.2f} dB"
                )
                if ttq_psnr_seconds is None and test_psnr >= args.ttq_psnr_target:
                    ttq_psnr_seconds = time.time() - train_wall_start
                    ttq_psnr_iteration = iteration
                    print(
                        f"TTQ reached: PSNR {test_psnr:.2f} dB >= {args.ttq_psnr_target:.2f} dB "
                        f"at iter {iteration} ({ttq_psnr_seconds:.1f}s)"
                    )

                # Append to lists for plotting
                iteration_list.append(iteration)
                psnr_list.append(test_psnr)
                recon_loss_list.append(train_losses["recon_loss"])
                trans_loss_list.append(train_losses["trans_loss"])
                total_loss_list.append(train_losses["total_loss"])
                affine_motion_rows.extend(
                    capture_affine_motion_rows(
                        model=model,
                        device=device,
                        num_samples=args.num_samples,
                        iteration=iteration,
                    )
                )
    progress_bar.close()
    if iteration > 0 and (len(iteration_list) == 0 or iteration_list[-1] != iteration):
        affine_motion_rows.extend(
            capture_affine_motion_rows(
                model=model,
                device=device,
                num_samples=args.num_samples,
                iteration=iteration,
            )
        )
    # Final evaluation and save output
    model.eval()
    eval_autocast_dtype = get_eval_autocast_dtype(args.eval_mixed_precision, device)
    with torch.no_grad():
        hr_coords = train_data.get_hr_coordinates().unsqueeze(0).to(device)
        hr_image = train_data.get_original_hr().unsqueeze(0).to(device)
        sample_id = torch.tensor([0]).to(device)

        if eval_autocast_dtype is not None:
            with torch.autocast(device_type="cuda", dtype=eval_autocast_dtype):
                output, _ = model(hr_coords, sample_id, scale_factor=1, training=False)
        else:
            output, _ = model(hr_coords, sample_id, scale_factor=1, training=False)

        # Unstandardize the output
        output = output * train_data.get_lr_std(0).to(device) + train_data.get_lr_mean(0).to(device)

        final_test_loss = F.mse_loss(output, hr_image).item()
        final_psnr = -10 * torch.log10(torch.tensor(final_test_loss)).item()

        # Convert tensors to numpy for saving as images
        pred_np = output.squeeze().cpu().numpy()
        gt_np = hr_image.squeeze().cpu().numpy()

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
        hr_h, hr_w = gt_np.shape[:2]
        lr_bilinear = cv2.resize(lr_original, (hr_w, hr_h), interpolation=cv2.INTER_LINEAR)
        pred_np = np.clip(pred_np, 0, 1)
        gt_np = np.clip(gt_np, 0, 1)
        lr_original = np.clip(lr_original, 0, 1)
        lr_bilinear = np.clip(lr_bilinear, 0, 1)

        # Convert numpy arrays to torch tensors for base-frame metrics.
        pred_tensor = (
            torch.from_numpy(pred_np).unsqueeze(0).permute(0, 3, 1, 2).to(device)
        )  # [1, C, H, W]
        gt_tensor = (
            torch.from_numpy(gt_np).unsqueeze(0).permute(0, 3, 1, 2).to(device)
        )  # [1, C, H, W]
        bilinear_tensor = (
            torch.from_numpy(lr_bilinear).unsqueeze(0).permute(0, 3, 1, 2).to(device)
        )  # [1, C, H, W]

        pred_eval = pred_tensor
        bilinear_eval = bilinear_tensor

        # PSNR in the base-frame HR coordinate system.
        model_psnr = peak_signal_noise_ratio(
            pred_eval.cpu(), gt_tensor.cpu(), data_range=1.0
        ).item()
        bilinear_psnr = peak_signal_noise_ratio(
            bilinear_eval.cpu(), gt_tensor.cpu(), data_range=1.0
        ).item()

        model_ssim = ssim(pred_eval.cpu(), gt_tensor.cpu(), data_range=1.0).item()
        bilinear_ssim = ssim(bilinear_eval.cpu(), gt_tensor.cpu(), data_range=1.0).item()

        # LPIPS expects [-1,1] range.
        lpips_fn = lpips.LPIPS(net="vgg").to(device)
        pred_lpips = lpips_fn((pred_eval * 2 - 1).to(device), (gt_tensor * 2 - 1).to(device)).item()
        bilinear_lpips = lpips_fn(
            (bilinear_eval * 2 - 1).to(device), (gt_tensor * 2 - 1).to(device)
        ).item()

        # Convert tensors back to numpy for visualization
        pred_eval_np = pred_eval.squeeze(0).permute(1, 2, 0).cpu().numpy()
        bilinear_eval_np = bilinear_eval.squeeze(0).permute(1, 2, 0).cpu().numpy()

        # Ensure images are in valid range
        pred_eval_np = np.clip(pred_eval_np, 0, 1)
        bilinear_eval_np = np.clip(bilinear_eval_np, 0, 1)

        use_raw_b432 = bool(getattr(train_data, "use_raw_b432", False))
        reflectance_stretch = (
            make_reflectance_display_stretch([gt_np, lr_original, bilinear_eval_np, pred_eval_np])
            if use_raw_b432
            else None
        )
        pred_display_np = display_hwc(pred_eval_np, use_raw_b432, reflectance_stretch)
        bilinear_display_np = display_hwc(bilinear_eval_np, use_raw_b432, reflectance_stretch)
        gt_display_np = display_hwc(gt_np, use_raw_b432, reflectance_stretch)
        lr_display_np = display_hwc(lr_original, use_raw_b432, reflectance_stretch)

        # Create structured output directory for single sample results
        output_base_dir = Path("single_samples")
        dataset_dir = output_base_dir / args.dataset
        sample_dir = dataset_dir / str(args.sample_id)
        sample_dir.mkdir(parents=True, exist_ok=True)

        # Save comparison figure with LR, bilinear upsampling, model output, and ground truth.
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))

        # Original LR image
        axes[0, 0].imshow(lr_display_np)
        axes[0, 0].set_title("Original LR Image", fontsize=14, fontweight="bold")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(bilinear_display_np)
        axes[0, 1].set_title(
            f"Bilinear Upsampling\nPSNR: {bilinear_psnr:.2f} dB",
            fontsize=14,
            fontweight="bold",
        )
        axes[0, 1].axis("off")

        axes[1, 0].imshow(pred_display_np)
        axes[1, 0].set_title(
            f"Model Output\nPSNR: {model_psnr:.2f} dB", fontsize=14, fontweight="bold"
        )
        axes[1, 0].axis("off")

        # Ground truth
        axes[1, 1].imshow(gt_display_np)
        axes[1, 1].set_title("Ground Truth HR", fontsize=14, fontweight="bold")
        axes[1, 1].axis("off")

        plt.tight_layout(pad=2.0)
        comparison_path = sample_dir / "comparison.png"
        plt.savefig(comparison_path, bbox_inches="tight", pad_inches=0.1, dpi=300)
        plt.close()

        diagnostic_path = sample_dir / "lr_reprojection_diagnostic.png"
        save_lr_reprojection_diagnostic(
            pred_hr_hwc=pred_eval_np,
            lr_target_hwc=lr_original,
            df=args.df,
            out_path=diagnostic_path,
            use_raw_b432=use_raw_b432,
        )

        # Save individual images for reference.
        plt.figure(figsize=(8, 8))
        plt.imshow(pred_display_np)
        plt.axis("off")
        plt.tight_layout(pad=0)
        pred_path = sample_dir / "model_output.png"
        plt.savefig(pred_path, bbox_inches="tight", pad_inches=0, dpi=300)
        plt.close()

        plt.figure(figsize=(8, 8))
        plt.imshow(gt_display_np)
        plt.axis("off")
        plt.tight_layout(pad=0)
        gt_path = sample_dir / "ground_truth.png"
        plt.savefig(gt_path, bbox_inches="tight", pad_inches=0, dpi=300)
        plt.close()

        # Save bilinear baseline for reference.
        plt.figure(figsize=(8, 8))
        plt.imshow(bilinear_display_np)
        plt.axis("off")
        plt.tight_layout(pad=0)
        bilinear_path = sample_dir / "bilinear_baseline.png"
        plt.savefig(bilinear_path, bbox_inches="tight", pad_inches=0, dpi=300)
        plt.close()

        # Save LR original for reference
        plt.figure(figsize=(8, 8))
        plt.imshow(lr_display_np)
        plt.axis("off")
        plt.tight_layout(pad=0)
        lr_path = sample_dir / "lr_original.png"
        plt.savefig(lr_path, bbox_inches="tight", pad_inches=0, dpi=300)
        plt.close()

        output_path = comparison_path

    print("\nFinal Results:")
    print(f"Test Loss: {final_test_loss:.6f}")
    print(f"Test PSNR: {final_psnr:.2f} dB")
    print(f"Model PSNR: {model_psnr:.2f} dB")
    print(f"Bilinear PSNR: {bilinear_psnr:.2f} dB")
    print(f"PSNR Improvement: {model_psnr - bilinear_psnr:.2f} dB")
    # LR reprojection metric evaluation removed.
    print(f"Model output saved to {output_path}")
    print(f"LR reprojection diagnostic saved to {diagnostic_path}")

    # Create structured output directory for single sample results
    output_base_dir = Path("single_samples")
    dataset_dir = output_base_dir / args.dataset
    sample_dir = dataset_dir / str(args.sample_id)
    sample_dir.mkdir(parents=True, exist_ok=True)
    affine_csv_path, affine_plot_path, affine_spatial_plot_path = save_affine_motion_artifacts(
        rows=affine_motion_rows,
        sample_dir=sample_dir,
        num_samples=args.num_samples,
        title_prefix=f"Sample {args.sample_id}",
    )

    # Save PSNR results to a text file in the structured directory
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

    Training Results:
    - Final Test Loss: {final_test_loss:.6f}
    - Final Test PSNR: {final_psnr:.2f} dB
    - Final Reconstruction Loss: {recon_loss_list[-1] if recon_loss_list else 0:.6f}
    - Final Transformation Loss: {trans_loss_list[-1] if trans_loss_list else 0:.6f}
    - Final Total Loss: {total_loss_list[-1] if total_loss_list else 0:.6f}
    - LR Reprojection PSNR: (removed)
    - LR Valid Fraction: (removed)

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
        if ttq_psnr_seconds is not None:
            results_text += (
                f"- TTQ@PSNR>={args.ttq_psnr_target:.2f} dB: "
                f"{ttq_psnr_seconds:.2f}s at iter {ttq_psnr_iteration}\n"
            )
        else:
            results_text += f"- TTQ@PSNR>={args.ttq_psnr_target:.2f} dB: not reached\n"
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
        "downsampling_factor": args.df,
        "model": args.model,
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
            "ttq_psnr_target": args.ttq_psnr_target,
            "ttq_psnr_seconds": ttq_psnr_seconds,
            "ttq_psnr_iteration": ttq_psnr_iteration,
            "affine_motion_rows": len(affine_motion_rows),
            "affine_motion_csv": str(affine_csv_path) if affine_csv_path else None,
            "affine_motion_plot": str(affine_plot_path) if affine_plot_path else None,
            "affine_motion_spatial_plot": (
                str(affine_spatial_plot_path) if affine_spatial_plot_path else None
            ),
        },
    }

    with open(sample_dir / "metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=2)

    print(f"Results saved to: {sample_dir}")
    print("PSNR results also saved to psnr_results.txt (current directory)")

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


if __name__ == "__main__":
    main()
