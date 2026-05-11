from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Experiment:
    name: str
    args: list[str]


def _run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _read_metrics(metrics_path: Path) -> dict:
    with metrics_path.open("r") as f:
        return json.load(f)


def _discover_satburst_samples(repo_root: Path) -> list[str]:
    data_dir = repo_root / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"Expected data directory at {data_dir}")
    sample_ids = [
        p.name
        for p in data_dir.iterdir()
        if p.is_dir() and not p.name.startswith(".") and p.name != "__pycache__"
    ]
    sample_ids.sort()
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
    """HashGrid hyperparameters for the TTO suite (tuned for satburst_synth; decoder input = n_levels * n_features)."""
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


def _superf_baseline_mlp_args() -> list[str]:
    """Fourier SuperF baseline (M0): classic wide MLP."""
    return ["--model", "mlp", "--network_depth", "4", "--network_hidden_dim", "256"]


def _hashgrid_mlp_args() -> list[str]:
    """HashGrid runs (M1+): smaller MLP than the Fourier baseline."""
    return ["--model", "mlp", "--network_depth", "3", "--network_hidden_dim", "64"]


def _fourier_baseline_mlp() -> list[str]:
    return ["--input_projection", "fourier_10", *_superf_baseline_mlp_args()]


def build_experiment_suite(*, include_scale_gates: bool) -> list[Experiment]:
    """
    TTO-only SuperF ablations (``tto_only_superf_improvement_plan.md``): M0–M4 core + optional M6 gates.
    """
    hg = _plan_hashgrid_args()
    hm = _hashgrid_mlp_args()
    fb = _fourier_baseline_mlp()

    suite: list[Experiment] = [
        Experiment(
            name="m0_superf_fourier",
            args=[*fb, "--run_name", "m0_superf_fourier"],
        ),
        Experiment(
            name="m1_hashgrid_superf",
            args=[*hg, *hm, "--hash_scale_attention", "none", "--run_name", "m1_hashgrid_superf"],
        ),
        Experiment(
            name="m2_progressive_hashgrid",
            args=[
                *hg,
                *hm,
                "--progressive_hashgrid",
                "--hash_scale_attention",
                "none",
                "--run_name",
                "m2_progressive_hashgrid",
            ],
        ),
        Experiment(
            name="m3_progressive_radiometric",
            args=[
                *hg,
                *hm,
                "--progressive_hashgrid",
                "--radiometric",
                "gain_offset",
                "--hash_scale_attention",
                "none",
                "--run_name",
                "m3_progressive_radiometric",
            ],
        ),
        Experiment(
            name="m4_progressive_robust_charbonnier",
            args=[
                *hg,
                *hm,
                "--progressive_hashgrid",
                "--radiometric",
                "gain_offset",
                "--recon_loss",
                "charbonnier",
                "--hash_scale_attention",
                "none",
                "--run_name",
                "m4_progressive_robust_charbonnier",
            ],
        ),
    ]

    if include_scale_gates:
        base_m4 = [
            *hg,
            *hm,
            "--progressive_hashgrid",
            "--radiometric",
            "gain_offset",
            "--recon_loss",
            "charbonnier",
            "--hash_scale_warmup_iterations",
            "300",
        ]
        suite.extend(
            [
                Experiment(
                    name="m6_scalegate_softmax",
                    args=[
                        *base_m4,
                        "--hash_scale_attention",
                        "softmax",
                        "--run_name",
                        "m6_scalegate_softmax",
                    ],
                ),
                Experiment(
                    name="m6_scalegate_sigmoid",
                    args=[
                        *base_m4,
                        "--hash_scale_attention",
                        "sigmoid",
                        "--run_name",
                        "m6_scalegate_sigmoid",
                    ],
                ),
                Experiment(
                    name="m6_scalegate_softmax_coord",
                    args=[
                        *base_m4,
                        "--hash_scale_attention",
                        "softmax",
                        "--hash_scale_coord_gate",
                        "--run_name",
                        "m6_scalegate_softmax_coord",
                    ],
                ),
            ]
        )

    return suite


def main() -> int:
    parser = argparse.ArgumentParser(description="Run experiment suite across satburst samples.")
    parser.add_argument("--dataset", default="satburst_synth")
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
        "--no_scale_gates",
        action="store_true",
        help="Skip optional M6 scale-gate experiments (M0–M4 core only).",
    )
    parser.add_argument(
        "--limit_samples",
        type=int,
        default=0,
        help="If >0, only run the first N discovered samples (smoke tests).",
    )
    args = parser.parse_args()

    dataset = str(args.dataset)
    device = str(args.device)

    common = [
        "--dataset",
        dataset,
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

    suite = build_experiment_suite(include_scale_gates=not bool(args.no_scale_gates))
    repo_root = Path(__file__).resolve().parents[1]
    sample_ids = _discover_satburst_samples(repo_root)
    if args.limit_samples and args.limit_samples > 0:
        sample_ids = sample_ids[: int(args.limit_samples)]
    if not sample_ids:
        raise RuntimeError(f"No samples found under {repo_root / 'data'}")

    print(
        f"Suite: {len(suite)} experiments/sample, scale_gates={'off' if args.no_scale_gates else 'on'}",
        flush=True,
    )

    all_rows: list[dict] = []

    for sample_id in sample_ids:
        print(f"\n=== Sample: {sample_id} ({dataset}) ===", flush=True)
        sample_rows: list[dict] = []

        for exp in suite:
            exp_args = [*common, "--sample_id", sample_id, *exp.args]
            _run([sys.executable, str(repo_root / "optimize.py"), *exp_args])
            metrics_path = (
                repo_root / "single_samples" / dataset / sample_id / exp.name / "metrics.json"
            )
            if not metrics_path.exists():
                raise FileNotFoundError(f"Expected metrics at {metrics_path}")
            m = _read_metrics(metrics_path)
            stats = m.get("hash_scale_attn_stats") or {}

            row = {
                "experiment": exp.name,
                "seed": int(args.seed),
                "dataset": m.get("dataset"),
                "sample_id": m.get("sample_id"),
                "df": m.get("downsampling_factor"),
                "model": m.get("model"),
                "progressive_hashgrid": m.get("progressive_hashgrid"),
                "radiometric": m.get("radiometric"),
                "recon_loss": m.get("recon_loss"),
                "hash_scale_attention": m.get("hash_scale_attention"),
                "hash_scale_coord_gate": m.get("hash_scale_coord_gate"),
                "hash_scale_coord_lowfreq_pe": m.get("hash_scale_coord_lowfreq_pe"),
                "hash_scale_gate_uses_coordinates": m.get("hash_scale_gate_uses_coordinates"),
                "hash_scale_mean_entropy": stats.get("mean_entropy"),
                "hash_scale_coarse_mass": stats.get("coarse_mass_mean"),
                "hash_scale_mid_mass": stats.get("mid_mass_mean"),
                "hash_scale_fine_mass": stats.get("fine_mass_mean"),
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
