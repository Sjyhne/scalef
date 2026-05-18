from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Experiment:
    name: str
    args: list[str]


def _run(cmd: list[str], *, cuda_expandable_segments: bool = False) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    if cuda_expandable_segments and "PYTORCH_CUDA_ALLOC_CONF" not in env:
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    subprocess.run(cmd, env=env, check=True)


def _read_metrics(metrics_path: Path) -> dict:
    with metrics_path.open("r") as f:
        return json.load(f)


def _max_lr_side_from_scene(scene_dir: Path, scale_leaf: str) -> int | None:
    """Read reference LR ``H,W`` from ``transform_log.json`` (first sorted key)."""
    tl = scene_dir / scale_leaf / "transform_log.json"
    if not tl.is_file():
        return None
    with tl.open("r") as f:
        log = json.load(f)
    for k in sorted(log.keys()):
        sh = log[k].get("shape")
        if isinstance(sh, (list, tuple)) and len(sh) >= 2:
            return max(int(sh[0]), int(sh[1]))
    return None


def _min_lr_side_from_scene(scene_dir: Path, scale_leaf: str) -> int | None:
    """Smallest LR side across all entries in ``transform_log.json`` for a scene."""
    tl = scene_dir / scale_leaf / "transform_log.json"
    if not tl.is_file():
        return None
    with tl.open("r") as f:
        log = json.load(f)
    sides: list[int] = []
    for v in log.values():
        sh = v.get("shape") if isinstance(v, dict) else None
        if isinstance(sh, (list, tuple)) and len(sh) >= 2:
            sides.append(min(int(sh[0]), int(sh[1])))
    return min(sides) if sides else None


def _discover_satburst_samples(
    repo_root: Path, data_root: str, df: int, lr_shift: float, aug: str, *, max_lr_side: int = 0
) -> list[str]:
    data_dir = repo_root / data_root
    if not data_dir.exists():
        raise FileNotFoundError(f"Expected data directory at {data_dir}")
    scale_leaf = f"scale_{df}_shift_{lr_shift:.1f}px_aug_{aug}"
    candidates = [
        p
        for p in data_dir.iterdir()
        if p.is_dir()
        and not p.name.startswith(".")
        and p.name != "__pycache__"
        and (p / scale_leaf).is_dir()
    ]
    sample_ids: list[str] = []
    skipped = 0
    for p in sorted(candidates, key=lambda x: x.name):
        if max_lr_side and max_lr_side > 0:
            side = _max_lr_side_from_scene(p, scale_leaf)
            if side is not None and side > max_lr_side:
                skipped += 1
                continue
        sample_ids.append(p.name)
    if max_lr_side and max_lr_side > 0 and skipped:
        print(f"Skipped {skipped} scene(s) with LR max side > {max_lr_side} (--max_lr_side).", flush=True)
    return sample_ids


def _write_rows_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_rows_json(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(rows, f, indent=2)


def _plan_hashgrid_args() -> list[str]:
    """HashGrid hyperparameters for the suite (decoder input = n_levels * n_features)."""
    return [
        "--input_projection",
        "hashgrid_tcnn",
        "--hash_encoding_preset",
        "smoothstep_grid",
        "--hash_grid_type",
        "Hash",
        "--hash_n_levels",
        "16",
        "--hash_n_features_per_level",
        "2",
        "--hash_log2_hashmap_size",
        "17",
        "--hash_base_resolution",
        "16",
        "--hash_max_resolution",
        "64",
        "--hash_encoding_dtype",
        "fp32",
    ]


def _suite_mlp_args() -> list[str]:
    return ["--model", "mlp", "--network_depth", "4", "--network_hidden_dim", "256"]


def _fourier_baseline_mlp() -> list[str]:
    return ["--input_projection", "fourier_10", *_suite_mlp_args()]


def build_experiment_suite() -> list[Experiment]:
    """SuperF Fourier features vs SuperF HashGrid (same MLP decoder)."""
    hg = _plan_hashgrid_args()
    hm = _suite_mlp_args()
    fb = _fourier_baseline_mlp()
    return [
        Experiment(
            name="m0_superf_fourier",
            args=[*fb, "--run_name", "m0_superf_fourier"],
        ),
        Experiment(
            name="m1_hashgrid_superf",
            args=[*hg, *hm, "--run_name", "m1_hashgrid_superf"],
        ),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Fourier vs HashGrid SuperF on satburst samples.")
    parser.add_argument(
        "--dataset",
        default="satburst_synth",
        choices=["satburst_synth", "satburst_real"],
        help="satburst_synth: scenes under data/. satburst_real: scenes under data_real/ (default root).",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--df", type=int, default=4)
    parser.add_argument("--lr_shift", type=float, default=1.0)
    parser.add_argument("--aug", default="none")
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--optimizer", default="adamw")
    parser.add_argument("--learning_rate", default="2e-3")
    parser.add_argument("--weight_decay", default="0.05")
    parser.add_argument(
        "--seed",
        type=int,
        default=6,
        help="Random seed passed through to optimize.py (torch/numpy/python).",
    )
    parser.add_argument(
        "--satburst_data_root",
        type=str,
        default=None,
        help=(
            "Override scene parent folder. Default: ``data`` for satburst_synth, ``data_real`` for "
            "satburst_real. Scenes without a matching ``scale_<df>_shift_<lr_shift>px_aug_<aug>/`` "
            "subfolder are skipped."
        ),
    )
    parser.add_argument(
        "--limit_samples",
        type=int,
        default=0,
        help="If >0, only run the first N discovered samples (smoke tests).",
    )
    parser.add_argument(
        "--max_lr_side",
        type=int,
        default=0,
        help="If >0, skip scenes whose reference LR max(H,W) from transform_log exceeds this (optional sweep filter).",
    )
    parser.add_argument(
        "--eval_crop_lr_size",
        type=str,
        default="auto",
        help=(
            "Pass-through to optimize.py --eval_crop_lr_size so per-scene SR/GT/bilinear are "
            "center-cropped to the same physical area before metrics+viz. "
            "Integer (e.g. 64) forces an explicit crop; ``auto`` (default) auto-detects the "
            "smallest LR side across discovered scenes (e.g. picks 64 across {64,128,256,512}); "
            "``0``/``off``/``none`` disables cropping."
        ),
    )
    parser.add_argument(
        "--cuda_expandable_segments",
        action="store_true",
        help=(
            "Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True for each optimize subprocess "
            "(can reduce fragmentation; off by default—some drivers/stacks error on first CUDA alloc)."
        ),
    )
    args = parser.parse_args()

    dataset = str(args.dataset)
    device = str(args.device)

    if args.satburst_data_root:
        data_root = str(args.satburst_data_root).strip().rstrip("/")
    elif dataset == "satburst_real":
        data_root = "data_real"
    else:
        data_root = "data"

    common = [
        "--dataset",
        dataset,
        "--satburst_data_root",
        data_root,
        "--df",
        str(args.df),
        "--lr_shift",
        str(args.lr_shift),
        "--aug",
        str(args.aug),
        "--num_samples",
        str(args.num_samples),
        "--iters",
        str(args.iters),
        "--batch_size",
        str(args.batch_size),
        "--optimizer",
        str(args.optimizer),
        "--learning_rate",
        str(args.learning_rate),
        "--weight_decay",
        str(args.weight_decay),
        "--seed",
        str(args.seed),
        "--device",
        device,
    ]

    suite = build_experiment_suite()
    repo_root = Path(__file__).resolve().parents[1]
    sample_ids = _discover_satburst_samples(
        repo_root,
        data_root,
        int(args.df),
        float(args.lr_shift),
        str(args.aug),
        max_lr_side=int(args.max_lr_side),
    )
    if args.limit_samples and args.limit_samples > 0:
        sample_ids = sample_ids[: int(args.limit_samples)]
    if not sample_ids:
        raise RuntimeError(
            f"No samples found under {repo_root / data_root} with scale folder "
            f"scale_{args.df}_shift_{float(args.lr_shift):.1f}px_aug_{args.aug}"
        )

    if dataset == "satburst_real":
        print(
            "Note (data_real): scenes can use different LR/HR sizes; metrics are per-scene. "
            "See data_real/README.txt.",
            flush=True,
        )

    # Resolve --eval_crop_lr_size: int / auto / 0|off|none. ``auto`` picks the smallest LR side
    # across discovered scenes so a 64/128/256/512 sweep is anchored on the 64 patch area.
    eval_crop_raw = str(args.eval_crop_lr_size).strip().lower()
    scale_leaf = f"scale_{int(args.df)}_shift_{float(args.lr_shift):.1f}px_aug_{args.aug}"
    if eval_crop_raw in {"0", "off", "none", "no", "false", ""}:
        eval_crop_lr_size = 0
    elif eval_crop_raw == "auto":
        sides: list[int] = []
        for sid in sample_ids:
            s = _min_lr_side_from_scene(repo_root / data_root / sid, scale_leaf)
            if s is not None and s > 0:
                sides.append(int(s))
        eval_crop_lr_size = min(sides) if sides else 0
        if eval_crop_lr_size > 0:
            print(
                f"Auto-detected --eval_crop_lr_size={eval_crop_lr_size} (smallest LR side across "
                f"{len(sample_ids)} scene(s)).",
                flush=True,
            )
        else:
            print(
                "Auto-detect for --eval_crop_lr_size found no usable LR sides; cropping disabled.",
                flush=True,
            )
    else:
        try:
            eval_crop_lr_size = max(0, int(eval_crop_raw))
        except ValueError as exc:
            raise SystemExit(
                f"Invalid --eval_crop_lr_size {args.eval_crop_lr_size!r}: expected int, 'auto', or 'off'."
            ) from exc

    print(f"Suite: {len(suite)} experiments/sample (Fourier + HashGrid), data_root={data_root!r}", flush=True)

    all_rows: list[dict] = []

    for sample_id in sample_ids:
        print(f"\n=== Sample: {sample_id} ({dataset}) ===", flush=True)
        sample_rows: list[dict] = []

        for exp in suite:
            exp_args = [*common, "--sample_id", sample_id, *exp.args]
            if eval_crop_lr_size > 0:
                exp_args = [*exp_args, "--eval_crop_lr_size", str(int(eval_crop_lr_size))]
            _run(
                [sys.executable, str(repo_root / "optimize.py"), *exp_args],
                cuda_expandable_segments=bool(args.cuda_expandable_segments),
            )
            metrics_path = (
                repo_root / "single_samples" / dataset / sample_id / exp.name / "metrics.json"
            )
            if not metrics_path.exists():
                raise FileNotFoundError(f"Expected metrics at {metrics_path}")
            m = _read_metrics(metrics_path)

            ec = m.get("eval_crop", {}) or {}
            row = {
                "experiment": exp.name,
                "seed": int(args.seed),
                "dataset": m.get("dataset"),
                "sample_id": m.get("sample_id"),
                "df": m.get("downsampling_factor"),
                "model": m.get("model"),
                "input_projection": m.get("input_projection"),
                "hash_base_resolution": m.get("hash_base_resolution"),
                "hash_max_resolution": m.get("hash_max_resolution"),
                "iters": m.get("iterations"),
                "psnr_model": m.get("psnr", {}).get("model"),
                "psnr_bilinear": m.get("psnr", {}).get("bilinear"),
                "psnr_improvement": m.get("psnr", {}).get("improvement"),
                "ssim_model": m.get("ssim", {}).get("model"),
                "ssim_bilinear": m.get("ssim", {}).get("bilinear"),
                "ssim_improvement": m.get("ssim", {}).get("improvement"),
                "lpips_model": m.get("lpips", {}).get("model"),
                "lpips_bilinear": m.get("lpips", {}).get("bilinear"),
                "lpips_improvement": m.get("lpips", {}).get("improvement"),
                "eval_crop_applied": bool(ec.get("applied", False)),
                "eval_crop_lr": ec.get("lr_crop"),
                "eval_crop_hr": ec.get("hr_crop"),
                "total_runtime_seconds": m.get("training", {}).get("total_runtime_seconds"),
                "ttq_psnr_seconds": m.get("training", {}).get("ttq_psnr_seconds"),
                "ttq_psnr_iteration": m.get("training", {}).get("ttq_psnr_iteration"),
                "artifact_dir": str(metrics_path.parent),
            }
            sample_rows.append(row)
            all_rows.append(row)

        sample_out_dir = repo_root / "single_samples" / dataset / sample_id
        _write_rows_csv(sample_out_dir / "experiment_results.csv", sample_rows)
        _write_rows_json(sample_out_dir / "experiment_results.json", sample_rows)

    agg_dir = repo_root / "single_samples" / dataset
    _write_rows_csv(agg_dir / "experiment_results_all_samples.csv", all_rows)
    _write_rows_json(agg_dir / "experiment_results_all_samples.json", all_rows)

    print(f"\nWrote {agg_dir / 'experiment_results_all_samples.csv'}")
    print(f"Wrote {agg_dir / 'experiment_results_all_samples.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
