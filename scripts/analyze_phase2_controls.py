#!/usr/bin/env python3
"""Summarize learned alignment, subpixel phase coverage, and run metrics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def _metric(metrics: dict, *path: str) -> float | None:
    value = metrics
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


def _phase_coverage(dx: np.ndarray, dy: np.ndarray, bins: int = 4) -> dict:
    px = np.mod(dx, 1.0)
    py = np.mod(dy, 1.0)
    occupied = np.zeros((bins, bins), dtype=bool)
    occupied[np.minimum((py * bins).astype(int), bins - 1),
             np.minimum((px * bins).astype(int), bins - 1)] = True
    resultant_x = abs(np.mean(np.exp(2j * np.pi * px)))
    resultant_y = abs(np.mean(np.exp(2j * np.pi * py)))
    return {
        "bins_per_axis": bins,
        "occupied_bins": int(occupied.sum()),
        "total_bins": bins * bins,
        "occupancy_fraction": float(occupied.mean()),
        "circular_spread_x": float(1.0 - resultant_x),
        "circular_spread_y": float(1.0 - resultant_y),
        "phases_xy": [[float(x), float(y)] for x, y in zip(px, py)],
    }


def analyze_run(run_dir: Path, *, phase_bins: int = 4) -> dict:
    run_dir = Path(run_dir)
    affine_path = run_dir / "affines.json"
    metrics_path = run_dir / "metrics.json"
    if not affine_path.is_file():
        raise FileNotFoundError(f"{affine_path} does not exist")
    affine = json.loads(affine_path.read_text())
    metrics = json.loads(metrics_path.read_text()) if metrics_path.is_file() else {}
    frames = list(affine.get("frames") or [])
    width = float(affine.get("lr_width") or 1)
    height = float(affine.get("lr_height") or 1)

    dx, dy, linear_error = [], [], []
    for frame in frames:
        matrix = np.asarray(frame["matrix_2x3"], dtype=np.float64)
        u = matrix[0, 0] * 0.5 + matrix[0, 1] * 0.5 + matrix[0, 2]
        v = matrix[1, 0] * 0.5 + matrix[1, 1] * 0.5 + matrix[1, 2]
        dx.append((u - 0.5) * width)
        dy.append((v - 0.5) * height)
        linear_error.append(float(np.linalg.norm(matrix[:, :2] - np.eye(2))))
    dx_arr = np.asarray(dx, dtype=np.float64)
    dy_arr = np.asarray(dy, dtype=np.float64)
    radial = np.hypot(dx_arr, dy_arr)
    if len(frames):
        centered = np.hypot(dx_arr - np.median(dx_arr), dy_arr - np.median(dy_arr))
    else:
        centered = np.asarray([], dtype=np.float64)

    def stats(values: np.ndarray) -> dict:
        if values.size == 0:
            return {"mean": None, "std": None, "median": None, "max_abs": None}
        return {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "median": float(np.median(values)),
            "max_abs": float(np.max(np.abs(values))),
        }

    return {
        "run_dir": str(run_dir),
        "frame_count": len(frames),
        "frozen_frame_count": sum(bool(frame.get("frozen")) for frame in frames),
        "phase_coverage": _phase_coverage(dx_arr, dy_arr, phase_bins) if len(frames) else None,
        "shift_distribution_lr_px": {
            "dx": stats(dx_arr),
            "dy": stats(dy_arr),
            "magnitude": stats(radial),
        },
        "residual_alignment_proxies": {
            "translation_about_median_rms_lr_px": (
                float(np.sqrt(np.mean(centered**2))) if centered.size else None
            ),
            "linear_identity_frobenius_rms": (
                float(np.sqrt(np.mean(np.square(linear_error)))) if linear_error else None
            ),
            "final_training_transform_loss": _metric(metrics, "training", "final_trans_loss"),
        },
        "metrics": {
            "psnr": _metric(metrics, "psnr", "model"),
            "psnr_improvement": _metric(metrics, "psnr", "improvement"),
            "ssim": _metric(metrics, "ssim", "model"),
            "ssim_improvement": _metric(metrics, "ssim", "improvement"),
            "lpips": _metric(metrics, "lpips", "model"),
            "lpips_improvement": _metric(metrics, "lpips", "improvement"),
            "training_time_seconds": _metric(metrics, "training_time_seconds"),
        },
    }


def _correlations(runs: list[dict]) -> dict:
    features = {
        "frame_count": lambda r: r["frame_count"],
        "phase_occupancy": lambda r: (r["phase_coverage"] or {}).get("occupancy_fraction"),
        "phase_spread_x": lambda r: (r["phase_coverage"] or {}).get("circular_spread_x"),
        "phase_spread_y": lambda r: (r["phase_coverage"] or {}).get("circular_spread_y"),
        "shift_std": lambda r: r["shift_distribution_lr_px"]["magnitude"]["std"],
        "translation_residual": lambda r: r["residual_alignment_proxies"][
            "translation_about_median_rms_lr_px"
        ],
    }
    outcomes = ("psnr", "psnr_improvement", "ssim", "lpips", "lpips_improvement")
    result = {}
    for feature, getter in features.items():
        for outcome in outcomes:
            pairs = [
                (getter(run), run["metrics"].get(outcome))
                for run in runs
            ]
            pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
            key = f"{feature}__vs__{outcome}"
            if len(pairs) < 2:
                result[key] = {"n": len(pairs), "pearson_r": None}
                continue
            x, y = np.asarray(pairs, dtype=np.float64).T
            if np.std(x) == 0 or np.std(y) == 0:
                result[key] = {"n": len(pairs), "pearson_r": None}
            else:
                result[key] = {"n": len(pairs), "pearson_r": float(np.corrcoef(x, y)[0, 1])}
    return result


def build_report(run_dirs: list[Path], *, phase_bins: int = 4) -> dict:
    runs = [analyze_run(path, phase_bins=phase_bins) for path in run_dirs]
    return {
        "runs": runs,
        "correlations": _correlations(runs) if len(runs) > 1 else {},
        "notes": {
            "phase": "Learned affine center shifts modulo one LR pixel.",
            "residual_proxy": "Dispersion around the median translation and non-identity linear terms; not image registration error.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="Run directories containing affines.json")
    parser.add_argument("--phase-bins", type=int, default=4)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = build_report(args.runs, phase_bins=args.phase_bins)
    text = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"Wrote {args.out}")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
