#!/usr/bin/env python3
"""Controlled synthetic resolution experiment for ScaleF.

The generator creates phase-diverse, duplicated, and integer-LR-aligned burst
controls from one exact HR truth.  Frames are translated in canonical HR
coordinates, degraded with ScaleF's band-wise meter PSF, then area pooled.

Examples
--------
python scripts/synthetic_resolution_experiment.py generate \
  --analytic slanted_edges --out experiments/synthetic_resolution --frames 8

python scripts/synthetic_resolution_experiment.py generate \
  --hr-source /path/to/nib.tif --crop 1000 2000 512 512 \
  --out experiments/synthetic_resolution --optimize-mode dry-run

python scripts/synthetic_resolution_experiment.py analyze \
  --experiment experiments/synthetic_resolution \
  --reconstruction phase_diverse=/path/to/sr_pred.tif \
  --reconstruction duplicated=/path/to/sr_pred.tif
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.s2_psf_forward import (  # noqa: E402
    DEFAULT_S2_PSF_SIGMA_M_BY_BAND,
    S2_RGB_BAND_ORDER,
    degrade_hr_bchw,
)

CONTROL_NAMES = ("phase_diverse", "duplicated", "integer_aligned")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_dump(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def analytic_target(size: int, kind: str = "slanted_edges") -> np.ndarray:
    """Return a deterministic HWC RGB target in [0, 1]."""
    if size < 32:
        raise ValueError("analytic target size must be at least 32")
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    xn = (x + 0.5) / size
    yn = (y + 0.5) / size

    if kind == "slanted_edges":
        edge_a = (xn + 0.23 * yn > 0.43).astype(np.float32)
        edge_b = (yn - 0.17 * xn > 0.55).astype(np.float32)
        rings = 0.5 + 0.5 * np.cos(2 * np.pi * (3.0 + 18.0 * xn) * xn)
        checker = ((np.floor(x / 3) + np.floor(y / 3)) % 2).astype(np.float32)
        rgb = np.stack(
            [
                0.08 + 0.72 * edge_a + 0.12 * rings,
                0.10 + 0.65 * edge_b + 0.16 * rings,
                0.10 + 0.55 * edge_a * edge_b + 0.22 * checker,
            ],
            axis=-1,
        )
    elif kind == "chirp":
        radial = np.sqrt((xn - 0.5) ** 2 + (yn - 0.5) ** 2)
        rgb = np.stack(
            [
                0.5 + 0.42 * np.sin(2 * np.pi * (2 + 28 * xn) * xn),
                0.5 + 0.42 * np.sin(2 * np.pi * (2 + 28 * yn) * yn),
                0.5 + 0.42 * np.sin(2 * np.pi * (3 + 36 * radial) * radial),
            ],
            axis=-1,
        )
    else:
        raise ValueError(f"unknown analytic target {kind!r}")
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def _normalize_image(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    source_dtype = array.dtype
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    if array.ndim != 3:
        raise ValueError(f"expected a 2D/3D image, got shape {array.shape}")
    if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.moveaxis(array, 0, -1)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    array = array[..., :3].astype(np.float32)
    if np.issubdtype(source_dtype, np.integer):
        peak = float(np.max(array)) if array.size else 1.0
        divisor = (
            255.0
            if peak <= 255.0
            else 10000.0
            if peak <= 20000.0
            else float(np.iinfo(source_dtype).max)
        )
        array /= divisor
    else:
        finite = array[np.isfinite(array)]
        peak = float(np.percentile(finite, 99.9)) if finite.size else 1.0
        if peak > 1.5:
            array /= 10000.0 if peak <= 20000 else peak
    return np.clip(np.nan_to_num(array), 0.0, 1.0).astype(np.float32)


def load_hr_source(
    path: Path,
    *,
    crop: tuple[int, int, int, int] | None = None,
    size: int | None = None,
) -> tuple[np.ndarray, dict]:
    """Load an HR image; crop is x, y, width, height."""
    suffix = path.suffix.lower()
    source_meta: dict = {"path": str(path.resolve()), "sha256": sha256_file(path)}
    if suffix == ".npy":
        raw = np.load(path)
    elif suffix == ".npz":
        archive = np.load(path)
        key = "hr" if "hr" in archive else archive.files[0]
        raw = archive[key]
        source_meta["npz_key"] = key
    elif suffix in {".tif", ".tiff"}:
        import rasterio

        with rasterio.open(path) as src:
            raw = src.read()
            source_meta.update(
                {
                    "crs": str(src.crs),
                    "transform": list(src.transform)[:6],
                    "source_shape": [src.height, src.width],
                }
            )
    else:
        from PIL import Image

        raw = np.asarray(Image.open(path).convert("RGB"))

    image = _normalize_image(raw)
    h, w = image.shape[:2]
    if crop is not None:
        x, y, cw, ch = crop
        if min(x, y) < 0 or cw <= 0 or ch <= 0 or x + cw > w or y + ch > h:
            raise ValueError(f"crop {crop} lies outside source shape {(h, w)}")
    else:
        side = min(h, w) if not size else min(int(size), h, w)
        x, y, cw, ch = (w - side) // 2, (h - side) // 2, side, side
    image = image[y : y + ch, x : x + cw]
    source_meta["crop_xywh"] = [x, y, cw, ch]
    return image, source_meta


def warp_hr_translation(
    truth_hwc: np.ndarray | torch.Tensor,
    *,
    dx_hr_px: float,
    dy_hr_px: float,
) -> torch.Tensor:
    """Sample truth at ``(x + dx, y + dy)`` using ScaleF affine semantics.

    Returns BCHW.  The corresponding ScaleF 2x3 coordinate affine has
    translations ``dx / W`` and ``dy / H`` because ScaleF coordinates span
    [0, 1].
    """
    tensor = torch.as_tensor(truth_hwc, dtype=torch.float32)
    if tensor.ndim == 3 and tensor.shape[-1] in (1, 3, 4):
        tensor = tensor.permute(2, 0, 1)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4:
        raise ValueError(f"truth must be HWC or BCHW, got {tuple(tensor.shape)}")
    _, _, height, width = tensor.shape
    theta = torch.tensor(
        [[[1.0, 0.0, 2.0 * dx_hr_px / width], [0.0, 1.0, 2.0 * dy_hr_px / height]]],
        dtype=tensor.dtype,
        device=tensor.device,
    )
    grid = F.affine_grid(theta, tensor.shape, align_corners=False)
    return F.grid_sample(
        tensor,
        grid,
        mode="bilinear",
        padding_mode="reflection",
        align_corners=False,
    )


def generate_control_shifts(
    control: str,
    frames: int,
    *,
    seed: int,
    max_subpixel_lr: float,
    max_integer_lr: int,
) -> list[tuple[float, float]]:
    """Generate deterministic (dx, dy) shifts measured in LR pixels."""
    if frames < 1:
        raise ValueError("frames must be positive")
    if control not in CONTROL_NAMES:
        raise ValueError(f"unknown control {control!r}")
    shifts = [(0.0, 0.0)]
    rng = np.random.default_rng(seed)
    if control == "phase_diverse":
        # Stratify each axis so phases cover the LR pixel instead of clustering.
        phases_x = (np.arange(frames - 1) + rng.random(frames - 1)) / max(1, frames - 1)
        phases_y = (np.arange(frames - 1) + rng.random(frames - 1)) / max(1, frames - 1)
        rng.shuffle(phases_y)
        for px, py in zip(phases_x, phases_y):
            shifts.append(
                (
                    float((2.0 * px - 1.0) * max_subpixel_lr),
                    float((2.0 * py - 1.0) * max_subpixel_lr),
                )
            )
    elif control == "duplicated":
        shifts.extend([(0.0, 0.0)] * (frames - 1))
    else:
        choices = [
            (float(dx), float(dy))
            for dy in range(-max_integer_lr, max_integer_lr + 1)
            for dx in range(-max_integer_lr, max_integer_lr + 1)
            if dx != 0 or dy != 0
        ]
        if not choices and frames > 1:
            raise ValueError("max_integer_lr must be >= 1 for multiple integer-aligned frames")
        order = rng.permutation(len(choices)) if choices else []
        for index in range(frames - 1):
            shifts.append(choices[int(order[index % len(order)])])
    return shifts


def degrade_shifted_truth(
    truth_hwc: np.ndarray,
    shift_lr_px: tuple[float, float],
    *,
    scale: int,
    native_gsd_m: float,
    sigma_m_by_band: dict[str, float],
    truncate: float = 4.0,
) -> np.ndarray:
    dx_lr, dy_lr = shift_lr_px
    shifted = warp_hr_translation(
        truth_hwc,
        dx_hr_px=dx_lr * scale,
        dy_hr_px=dy_lr * scale,
    )
    lr = degrade_hr_bchw(
        shifted,
        scale,
        "s2_psf_m",
        truncate=truncate,
        band_order=S2_RGB_BAND_ORDER,
        native_gsd_m=native_gsd_m,
        sigma_m_by_band=sigma_m_by_band,
    )
    return lr[0].permute(1, 2, 0).numpy()


def _write_geotiff(path: Path, image_hwc: np.ndarray, *, gsd_m: float) -> None:
    import rasterio
    from rasterio.transform import from_origin

    height, width, channels = image_hwc.shape
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": channels,
        "dtype": "float32",
        "crs": "EPSG:32632",
        "transform": from_origin(500000.0, 7000000.0, gsd_m, gsd_m),
        "compress": "deflate",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.moveaxis(image_hwc.astype(np.float32), -1, 0))


def _git_provenance() -> dict:
    def run(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


def build_optimize_command(
    dataset_dir: Path,
    truth_path: Path,
    *,
    control: str,
    scale: int,
    frames: int,
    seed: int,
    device: str,
    native_gsd_m: float = 10.0,
    sigma_m_by_band: dict[str, float] | None = None,
    psf_truncate: float = 4.0,
    extra: Iterable[str] = (),
) -> list[str]:
    sigma = sigma_m_by_band or DEFAULT_S2_PSF_SIGMA_M_BY_BAND
    return [
        sys.executable,
        str(ROOT / "optimize.py"),
        "--dataset",
        f"synthetic_{control}",
        "--s2-dir",
        str(dataset_dir.resolve()),
        "--hr-path",
        str(truth_path.resolve()),
        "--df",
        str(scale),
        "--scale_factor",
        str(scale),
        "--num_samples",
        str(frames),
        "--seed",
        str(seed),
        "--device",
        str(device),
        "--lr_degradation",
        "s2_psf_m",
        "--s2-native-gsd-m",
        str(native_gsd_m),
        "--s2-psf-truncate",
        str(psf_truncate),
        "--s2-psf-sigma-b02-m",
        str(sigma["B02"]),
        "--s2-psf-sigma-b03-m",
        str(sigma["B03"]),
        "--s2-psf-sigma-b04-m",
        str(sigma["B04"]),
        "--no_hr_harmonize",
        "--no_hr_spatial_align",
        "--run_name",
        dataset_dir.parent.name,
        *extra,
    ]


def generate_experiment(args: argparse.Namespace) -> dict:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.hr_source:
        truth, source = load_hr_source(
            args.hr_source,
            crop=tuple(args.crop) if args.crop else None,
            size=args.hr_size,
        )
    else:
        truth = analytic_target(args.hr_size, args.analytic)
        source = {"analytic": args.analytic, "size": args.hr_size}
    height, width = truth.shape[:2]
    if height % args.scale or width % args.scale:
        raise ValueError(
            f"HR shape {(height, width)} must be divisible by scale {args.scale}"
        )

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    truth_path = out / "hr_ground_truth.tif"
    _write_geotiff(truth_path, truth, gsd_m=args.native_gsd_m / args.scale)

    sigma = {
        "B02": args.sigma_b02_m,
        "B03": args.sigma_b03_m,
        "B04": args.sigma_b04_m,
    }
    controls = args.controls or list(CONTROL_NAMES)
    commands: dict[str, list[str]] = {}
    control_manifests: dict[str, dict] = {}
    start_date = datetime(2020, 1, 1, tzinfo=timezone.utc)

    for control_index, control in enumerate(controls):
        dataset_dir = out / control
        dataset_dir.mkdir(parents=True, exist_ok=True)
        shifts = generate_control_shifts(
            control,
            args.frames,
            seed=args.seed + 1009 * control_index,
            max_subpixel_lr=args.max_subpixel_lr,
            max_integer_lr=args.max_integer_lr,
        )
        frames_meta = []
        transforms = []
        duplicated_frame: np.ndarray | None = None
        for frame_index, (dx_lr, dy_lr) in enumerate(shifts):
            if control == "duplicated" and duplicated_frame is not None:
                lr = duplicated_frame.copy()
            else:
                lr = degrade_shifted_truth(
                    truth,
                    (dx_lr, dy_lr),
                    scale=args.scale,
                    native_gsd_m=args.native_gsd_m,
                    sigma_m_by_band=sigma,
                    truncate=args.psf_truncate,
                )
                if args.noise_std > 0:
                    noise_rng = np.random.default_rng(
                        args.seed + 100000 * control_index + frame_index
                    )
                    lr = np.clip(
                        lr + noise_rng.normal(0.0, args.noise_std, lr.shape),
                        0.0,
                        1.0,
                    ).astype(np.float32)
                if control == "duplicated":
                    duplicated_frame = lr.copy()
            dt = start_date + timedelta(days=frame_index)
            frame_name = f"{frame_index + 1:03d}_{dt.strftime('%Y%m%d')}.tif"
            frame_path = dataset_dir / frame_name
            _write_geotiff(frame_path, lr, gsd_m=args.native_gsd_m)
            frames_meta.append(
                {
                    "index": frame_index + 1,
                    "path": frame_name,
                    "datetime": dt.isoformat(),
                    "source": "controlled_synthetic_resolution",
                    "sha256": sha256_file(frame_path),
                }
            )
            transforms.append(
                {
                    "frame_index": frame_index,
                    "shift_lr_px": {"dx": dx_lr, "dy": dy_lr},
                    "shift_hr_px": {
                        "dx": dx_lr * args.scale,
                        "dy": dy_lr * args.scale,
                    },
                    "canonical_to_frame_affine_2x3": [
                        [1.0, 0.0, dx_lr * args.scale / width],
                        [0.0, 1.0, dy_lr * args.scale / height],
                    ],
                    "sampling_convention": "frame(x,y)=truth(x+dx,y+dy); pixel centers; align_corners=False",
                }
            )

        lr_h, lr_w = height // args.scale, width // args.scale
        meta = {
            "schema": "scalef_s2_revisit_v1",
            "synthetic": True,
            "control": control,
            "resolution_m": args.native_gsd_m,
            "width": lr_w,
            "height": lr_h,
            "crs": "EPSG:32632",
            "transform": [args.native_gsd_m, 0.0, 500000.0, 0.0, -args.native_gsd_m, 7000000.0],
            "aoi_window": {"col_off": 0, "row_off": 0, "width": lr_w, "height": lr_h},
            "center_date": "2020-01-01",
            "nib_acquisition_date": "2020-01-01",
            "frames": frames_meta,
        }
        _json_dump(dataset_dir / "meta.json", meta)
        transform_payload = {
            "schema": "scalef_exact_transforms_v1",
            "control": control,
            "reference_frame": 0,
            "scale": args.scale,
            "transforms": transforms,
        }
        _json_dump(dataset_dir / "exact_transforms.json", transform_payload)
        control_manifests[control] = {
            "dataset_dir": str(dataset_dir),
            "meta_sha256": sha256_file(dataset_dir / "meta.json"),
            "transform_sha256": sha256_file(dataset_dir / "exact_transforms.json"),
            "shifts_lr_px": [[dx, dy] for dx, dy in shifts],
        }
        commands[control] = build_optimize_command(
            dataset_dir,
            truth_path,
            control=control,
            scale=args.scale,
            frames=args.frames,
            seed=args.seed,
            device=args.device,
            native_gsd_m=args.native_gsd_m,
            sigma_m_by_band=sigma,
            psf_truncate=args.psf_truncate,
            extra=args.optimize_arg,
        )

    manifest = {
        "schema": "scalef_synthetic_resolution_experiment_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "seed": args.seed,
        "source": source,
        "truth": {
            "path": str(truth_path),
            "sha256": sha256_file(truth_path),
            "shape_hwc": list(truth.shape),
        },
        "degradation": {
            "operator": "models.s2_psf_forward.degrade_hr_bchw",
            "mode": "s2_psf_m",
            "scale": args.scale,
            "native_gsd_m": args.native_gsd_m,
            "sigma_m_by_band": sigma,
            "band_order": list(S2_RGB_BAND_ORDER),
            "truncate": args.psf_truncate,
            "area_downsampling": True,
            "noise_std": args.noise_std,
        },
        "controls": control_manifests,
        "optimize_commands": commands,
        "provenance": {
            "git": _git_provenance(),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "script_sha256": sha256_file(Path(__file__)),
        },
    }
    _json_dump(out / "manifest.json", manifest)

    for control in controls:
        command = commands[control]
        print(f"[{control}] {shlex.join(command)}")
        if args.optimize_mode == "run":
            subprocess.run(command, cwd=ROOT, check=True)
    print(f"Wrote controlled experiment to {out}")
    return manifest


def _center_match(pred: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h = min(pred.shape[0], truth.shape[0])
    w = min(pred.shape[1], truth.shape[1])

    def crop(image: np.ndarray) -> np.ndarray:
        y = (image.shape[0] - h) // 2
        x = (image.shape[1] - w) // 2
        return image[y : y + h, x : x + w, :3]

    return crop(pred), crop(truth)


def image_metrics(pred: np.ndarray, truth: np.ndarray, *, use_lpips: bool = True) -> dict:
    """CPU full-reference metrics in [0,1], with optional LPIPS."""
    pred, truth = _center_match(
        np.clip(pred.astype(np.float32), 0, 1),
        np.clip(truth.astype(np.float32), 0, 1),
    )
    mse = float(np.mean((pred - truth) ** 2))
    psnr = float("inf") if mse == 0 else float(-10.0 * math.log10(mse))
    try:
        from skimage.metrics import structural_similarity

        ssim = float(structural_similarity(truth, pred, data_range=1.0, channel_axis=-1))
    except ImportError:
        # Global SSIM fallback, deterministic and dependency-free.
        mux, muy = float(pred.mean()), float(truth.mean())
        vx, vy = float(pred.var()), float(truth.var())
        covariance = float(np.mean((pred - mux) * (truth - muy)))
        ssim = ((2 * mux * muy + 0.01**2) * (2 * covariance + 0.03**2)) / (
            (mux * mux + muy * muy + 0.01**2) * (vx + vy + 0.03**2)
        )
    result = {"mse": mse, "psnr_db": psnr, "ssim": ssim, "lpips": None}
    if use_lpips:
        try:
            import lpips

            model = lpips.LPIPS(net="vgg").cpu().eval()
            pred_t = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0) * 2 - 1
            truth_t = torch.from_numpy(truth).permute(2, 0, 1).unsqueeze(0) * 2 - 1
            with torch.no_grad():
                result["lpips"] = float(model(pred_t, truth_t).item())
        except Exception as exc:  # LPIPS may be installed while its weights are unavailable.
            result["lpips_unavailable"] = str(exc)
    return result


def radial_fourier_recovery(
    pred: np.ndarray,
    truth: np.ndarray,
    *,
    bins: int = 32,
) -> dict:
    """Radially bin exact-truth Fourier amplitude recovery on CPU."""
    pred, truth = _center_match(pred, truth)
    pred_gray = pred.mean(axis=-1).astype(np.float64)
    truth_gray = truth.mean(axis=-1).astype(np.float64)
    # Hann window limits crop-boundary leakage.
    window = np.outer(np.hanning(pred.shape[0]), np.hanning(pred.shape[1]))
    pred_fft = np.abs(np.fft.fftshift(np.fft.fft2((pred_gray - pred_gray.mean()) * window)))
    truth_fft = np.abs(np.fft.fftshift(np.fft.fft2((truth_gray - truth_gray.mean()) * window)))
    fy = np.fft.fftshift(np.fft.fftfreq(pred.shape[0]))
    fx = np.fft.fftshift(np.fft.fftfreq(pred.shape[1]))
    radius = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    edges = np.linspace(0.0, 0.5, bins + 1)
    centers, recovery, truth_amp, pred_amp = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (radius >= lo) & (radius < hi)
        ta = float(truth_fft[mask].mean()) if np.any(mask) else 0.0
        pa = float(pred_fft[mask].mean()) if np.any(mask) else 0.0
        centers.append(float((lo + hi) / 2))
        truth_amp.append(ta)
        pred_amp.append(pa)
        recovery.append(pa / ta if ta > 1e-12 else None)
    valid = [
        (frequency, value)
        for frequency, value in zip(centers, recovery)
        if value is not None and frequency >= 0.25
    ]
    weighted_high = float(np.mean([min(value, 2.0) for _, value in valid])) if valid else None
    return {
        "frequency_cycles_per_hr_pixel": centers,
        "truth_radial_amplitude": truth_amp,
        "reconstruction_radial_amplitude": pred_amp,
        "amplitude_recovery": recovery,
        "high_frequency_recovery_0p25_to_0p5": weighted_high,
    }


def analyze_reconstructions(args: argparse.Namespace) -> dict:
    experiment = args.experiment.resolve()
    manifest = json.loads((experiment / "manifest.json").read_text())
    truth = _read_image(Path(manifest["truth"]["path"]))
    reconstructions: dict[str, Path] = {}
    for item in args.reconstruction:
        if "=" not in item:
            raise ValueError("--reconstruction must be CONTROL=PATH")
        control, path = item.split("=", 1)
        reconstructions[control] = Path(path)
    if args.auto_discover:
        for control in manifest["controls"]:
            expected = (
                ROOT
                / "single_samples"
                / f"synthetic_{control}"
                / "sample"
                / experiment.name
                / "qgis"
                / "sr_pred.tif"
            )
            if expected.is_file():
                reconstructions.setdefault(control, expected)
    if not reconstructions:
        raise ValueError("no reconstructions supplied or discovered")

    rows = {}
    for control, path in reconstructions.items():
        pred = _read_image(path)
        rows[control] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "metrics": image_metrics(pred, truth, use_lpips=not args.no_lpips),
            "fourier_radial": radial_fourier_recovery(pred, truth, bins=args.radial_bins),
        }
    comparisons = {}
    if "phase_diverse" in rows:
        reference = rows["phase_diverse"]
        for control, row in rows.items():
            if control == "phase_diverse":
                continue
            metric_delta = {}
            for key in ("psnr_db", "ssim", "lpips"):
                phase_value = reference["metrics"].get(key)
                control_value = row["metrics"].get(key)
                metric_delta[key] = (
                    None
                    if phase_value is None or control_value is None
                    else float(phase_value - control_value)
                )
            phase_fourier = reference["fourier_radial"].get(
                "high_frequency_recovery_0p25_to_0p5"
            )
            control_fourier = row["fourier_radial"].get(
                "high_frequency_recovery_0p25_to_0p5"
            )
            comparisons[control] = {
                "phase_diverse_minus_control": {
                    **metric_delta,
                    "high_frequency_recovery": (
                        None
                        if phase_fourier is None or control_fourier is None
                        else float(phase_fourier - control_fourier)
                    ),
                }
            }
    summary = {
        "schema": "scalef_synthetic_resolution_analysis_v1",
        "experiment_manifest_sha256": sha256_file(experiment / "manifest.json"),
        "truth_sha256": manifest["truth"]["sha256"],
        "controls": rows,
        "comparisons_to_phase_diverse": comparisons,
    }
    output = args.output or experiment / "analysis.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    _json_dump(output, summary)
    print(json.dumps({name: row["metrics"] for name, row in rows.items()}, indent=2))
    print(f"Wrote analysis to {output}")
    return summary


def _read_image(path: Path) -> np.ndarray:
    if path.suffix.lower() in {".tif", ".tiff"}:
        import rasterio

        with rasterio.open(path) as src:
            raw = src.read()
    elif path.suffix.lower() == ".npy":
        raw = np.load(path)
    else:
        from PIL import Image

        raw = np.asarray(Image.open(path).convert("RGB"))
    return _normalize_image(raw)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate", help="generate datasets and optional ScaleF commands")
    source = generate.add_mutually_exclusive_group()
    source.add_argument("--hr-source", type=Path)
    source.add_argument("--analytic", choices=["slanted_edges", "chirp"], default="slanted_edges")
    generate.add_argument("--crop", type=int, nargs=4, metavar=("X", "Y", "W", "H"))
    generate.add_argument("--hr-size", type=int, default=256)
    generate.add_argument("--out", type=Path, required=True)
    generate.add_argument("--frames", type=int, default=8)
    generate.add_argument("--scale", type=int, default=4)
    generate.add_argument("--seed", type=int, default=6)
    generate.add_argument("--controls", nargs="+", choices=CONTROL_NAMES)
    generate.add_argument("--max-subpixel-lr", type=float, default=0.49)
    generate.add_argument("--max-integer-lr", type=int, default=2)
    generate.add_argument("--native-gsd-m", type=float, default=10.0)
    generate.add_argument("--sigma-b02-m", type=float, default=DEFAULT_S2_PSF_SIGMA_M_BY_BAND["B02"])
    generate.add_argument("--sigma-b03-m", type=float, default=DEFAULT_S2_PSF_SIGMA_M_BY_BAND["B03"])
    generate.add_argument("--sigma-b04-m", type=float, default=DEFAULT_S2_PSF_SIGMA_M_BY_BAND["B04"])
    generate.add_argument("--psf-truncate", type=float, default=4.0)
    generate.add_argument("--noise-std", type=float, default=0.0)
    generate.add_argument("--optimize-mode", choices=["none", "dry-run", "run"], default="none")
    generate.add_argument("--device", default="0")
    generate.add_argument(
        "--optimize-arg",
        action="append",
        default=[],
        help="one extra optimize.py token; repeat for flags and values",
    )
    generate.set_defaults(func=generate_experiment)

    analyze = sub.add_parser("analyze", help="compare reconstructions with exact truth")
    analyze.add_argument("--experiment", type=Path, required=True)
    analyze.add_argument("--reconstruction", action="append", default=[])
    analyze.add_argument("--auto-discover", action=argparse.BooleanOptionalAction, default=True)
    analyze.add_argument("--radial-bins", type=int, default=32)
    analyze.add_argument("--no-lpips", action="store_true")
    analyze.add_argument("--output", type=Path)
    analyze.set_defaults(func=analyze_reconstructions)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
