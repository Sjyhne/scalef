#!/usr/bin/env python3
"""Regenerate SR / fixed-spot visualizations from saved eval outputs.

Examples::

    # Multi-sample run folder
    python scripts/visualize_eval_results.py multi_sample_results/

    # Single run under single_samples/
    python scripts/visualize_eval_results.py \\
        single_samples/satburst_synth/UNHCR-YEMs035290_rgb/bench_lr224_hashmax_auto

    # LR-size benchmark summary (spot vs LR plot)
    python scripts/visualize_eval_results.py \\
        single_samples/satburst_synth/UNHCR-YEMs035290_rgb/benchmark_psf_ablation_*/summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2
import numpy as np

from eval.visualize import (
    create_spot_summary_visualization,
    create_sr_sample_grid,
    save_eval_visualizations,
    visualize_benchmark_runs,
)


def _load_sample_results(run_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for sample_dir in sorted(run_dir.glob("sample_*")):
        metrics_path = sample_dir / "metrics.json"
        if not metrics_path.is_file():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        idx = int(sample_dir.name.split("_")[-1])
        rows.append(
            {
                "sample_idx": idx,
                "sample_info": {"sample_id": metrics.get("sample_id", sample_dir.name)},
                "image_metrics": {
                    "model_psnr": metrics.get("model_psnr"),
                    "bilinear_psnr": metrics.get("bilinear_psnr"),
                    "model_ssim": (metrics.get("ssim") or {}).get("model"),
                    "bilinear_ssim": (metrics.get("ssim") or {}).get("bilinear"),
                    "model_lpips": (metrics.get("lpips") or {}).get("model"),
                    "bilinear_lpips": (metrics.get("lpips") or {}).get("bilinear"),
                    "fixed_spot": metrics.get("fixed_spot"),
                },
            }
        )
    return rows


def _regenerate_sample_dir(sample_dir: Path) -> None:
    metrics_path = sample_dir / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Missing {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    fixed_spot = metrics.get("fixed_spot")

    def _read(name: str) -> np.ndarray:
        path = sample_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}")
        img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        return img.astype(np.float32) / 255.0

    save_eval_visualizations(
        sample_dir,
        lr_hwc=_read("lr_original.png"),
        bilinear_hwc=_read("bilinear_baseline.png"),
        pred_hwc=_read("model_output_aligned.png"),
        gt_hwc=_read("ground_truth.png"),
        image_metrics={
            "model_psnr": metrics.get("model_psnr", 0.0),
            "bilinear_psnr": metrics.get("bilinear_psnr", 0.0),
            "model_ssim": (metrics.get("ssim") or {}).get("model", 0.0),
            "bilinear_ssim": (metrics.get("ssim") or {}).get("bilinear", 0.0),
            "model_lpips": (metrics.get("lpips") or {}).get("model", 0.0),
            "bilinear_lpips": (metrics.get("lpips") or {}).get("bilinear", 0.0),
        },
        fixed_spot=fixed_spot,
        sample_label=str(metrics.get("sample_id") or sample_dir.name),
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", type=Path, help="Run dir, sample dir, or benchmark summary.json")
    p.add_argument(
        "--benchmark-plot",
        action="store_true",
        help="Force benchmark spot-vs-LR plot (path must be summary.json).",
    )
    args = p.parse_args()
    path = Path(args.path)

    if path.is_file() and path.name == "summary.json":
        out = visualize_benchmark_runs(path)
        print(f"Wrote {out}")
        return 0

    if path.is_file():
        raise SystemExit(f"Expected directory or summary.json, got file: {path}")

    if (path / "summary.json").is_file() and args.benchmark_plot:
        out = visualize_benchmark_runs(path / "summary.json")
        print(f"Wrote {out}")
        return 0

    if path.name.startswith("sample_") and (path / "metrics.json").is_file():
        _regenerate_sample_dir(path)
        print(f"Updated visualizations in {path}")
        return 0

    sample_dirs = sorted(path.glob("sample_*"))
    if sample_dirs:
        for sample_dir in sample_dirs:
            if (sample_dir / "metrics.json").is_file() and (sample_dir / "model_output_aligned.png").is_file():
                _regenerate_sample_dir(sample_dir)
                print(f"Updated {sample_dir}")
        all_results = _load_sample_results(path)
        if all_results:
            create_spot_summary_visualization(all_results, path)
            create_sr_sample_grid(path, all_results)
            print(f"Wrote spot_summary_metrics.png and sr_spot_grid.png in {path}")
        return 0

    if (path / "metrics.json").is_file() and (path / "model_output_aligned.png").is_file():
        _regenerate_sample_dir(path)
        print(f"Updated visualizations in {path}")
        return 0

    raise SystemExit(
        f"Could not interpret {path}. Expected multi-sample dir, single run dir, or summary.json."
    )


if __name__ == "__main__":
    raise SystemExit(main())
