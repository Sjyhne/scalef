from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from collections.abc import Callable
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


WORLDSTRAT_DATASETS = ("worldstrat_test", "worldstrat_sweet", "worldstrat_bitter")


def _default_worldstrat_data_root(dataset: str) -> str:
    if dataset == "worldstrat_sweet":
        return "worldstrat_datasets/worldstrat_sweet"
    if dataset == "worldstrat_bitter":
        return "worldstrat_datasets/worldstrat_bitter"
    return "worldstrat_test_data"


def _worldstrat_data_missing_message(repo_root: Path, data_root: str) -> str:
    data_path = repo_root / data_root
    lines = [
        f"WorldStrat data directory not found: {data_path}",
        "",
        "Expected layout (see docs/WORLDSTRAT_EXPERIMENTS.md):",
        f"  {data_root}/<area_name>/hr/*.png",
        f"  {data_root}/<area_name>/lr/*.png",
        "",
        "Fix options:",
        f"  1. Download or copy areas into {data_path}",
        f"  2. Point to an existing tree: WORLDSTRAT_DATA_ROOT=/path/to/parent ./scripts/run_attention_experiment_suite.sh",
        "     or: python scripts/run_experiment_suite.py --dataset worldstrat_sweet --worldstrat_data_root /path/to/parent ...",
    ]
    if data_path.is_symlink():
        try:
            target = data_path.readlink()
        except OSError:
            target = "?"
        resolved = data_path.resolve(strict=False)
        lines.append(f"  Note: {data_path} is a symlink -> {target} (resolved: {resolved})")
    elif not data_path.exists() and (repo_root / "worldstrat_datasets").is_symlink():
        ws = repo_root / "worldstrat_datasets"
        lines.append(
            f"  Note: {ws} symlink may be broken (target {ws.readlink()}). "
            "Recreate it to your data checkout."
        )
    return "\n".join(lines)


def _discover_worldstrat_samples(repo_root: Path, data_root: str) -> list[str]:
    data_dir = repo_root / data_root
    if not data_dir.is_dir():
        raise FileNotFoundError(_worldstrat_data_missing_message(repo_root, data_root))
    sample_ids: list[str] = []
    for p in sorted(data_dir.iterdir(), key=lambda x: x.name):
        if not p.is_dir() or p.name.startswith("."):
            continue
        if (p / "hr").is_dir() and (p / "lr").is_dir():
            sample_ids.append(p.name)
    return sample_ids


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
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_rows_json(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(rows, f, indent=2)


def _adaptive_hash_resolution_args() -> list[str]:
    """Per-scene coarsest/finest grid from LR size (see ``optimize.resolve_hash_grid_resolutions``)."""
    return [
        "--hash_base_resolution",
        "0",
        "--hash_max_resolution",
        "0",
    ]


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
        "21",
        "--hash_encoding_dtype",
        "fp32",
        *_adaptive_hash_resolution_args(),
    ]


def _suite_mlp_tcnn_args(*, network_depth: str = "4", network_hidden_dim: str = "256") -> list[str]:
    """Plain MLP decoder via tiny-cuda-nn (same role as PyTorch ``mlp``, faster on GPU)."""
    return [
        "--model",
        "mlp_tcnn",
        "--tcnn_mlp_dtype",
        "fp16",
        "--network_depth",
        str(network_depth),
        "--network_hidden_dim",
        str(network_hidden_dim),
    ]


def _fourier_baseline_mlp_tcnn() -> list[str]:
    return ["--input_projection", "fourier_10", *_suite_mlp_tcnn_args()]


def build_experiment_suite() -> list[Experiment]:
    """SuperF Fourier features vs SuperF HashGrid (both use decoder 4×256)."""
    hg = _plan_hashgrid_args()
    hm = _suite_mlp_tcnn_args()
    fb = _fourier_baseline_mlp_tcnn()
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


def _hash_attn_exam_hashgrid_args() -> list[str]:
    """HashGrid settings for the exam ablation (see hashgrid_attention_agent_brief.md)."""
    return [
        "--input_projection",
        "hashgrid",
        "--hash_n_levels",
        "16",
        "--hash_n_features_per_level",
        "2",
        "--hash_log2_hashmap_size",
        "21",
        "--hash_encoding_dtype",
        "fp32",
        "--hash_encoding_preset",
        "smoothstep_grid",
        "--hash_grid_type",
        "Hash",
        *_adaptive_hash_resolution_args(),
        "--network_depth",
        "4",
        "--network_hidden_dim",
        "256",
    ]


def build_hash_attn_experiment_suite() -> list[Experiment]:
    """HashGrid + plain MLP vs HashGrid + coordinate-conditioned level attention."""
    hg = _hash_attn_exam_hashgrid_args()
    return [
        Experiment(
            name="hashgrid_mlp",
            args=[*hg, "--model", "mlp_tcnn", "--tcnn_mlp_dtype", "fp16", "--run_name", "hashgrid_mlp"],
        ),
        Experiment(
            name="hashgrid_level_attention",
            args=[
                *hg,
                "--model",
                "hash_attn",
                "--hash_attn_token_dim",
                "32",
                "--run_name",
                "hashgrid_level_attention",
            ],
        ),
    ]


def build_all_experiment_suite() -> list[Experiment]:
    """Fourier + HashGrid SuperF, then HashGrid MLP vs level-attention (full comparison)."""
    return [*build_experiment_suite(), *build_hash_attn_experiment_suite()]


def _attention_exam_decoder_args() -> list[str]:
    """Matched MLP decoder for all attention-suite methods (A–D)."""
    return [
        "--network_depth",
        "4",
        "--network_hidden_dim",
        "256",
        "--weight_decay",
        "0",
    ]


def _attention_exam_hashgrid_args() -> list[str]:
    """HashGrid encoding for methods B and C (decoder 4×256)."""
    return [
        "--input_projection",
        "hashgrid",
        "--hash_n_levels",
        "16",
        "--hash_n_features_per_level",
        "2",
        "--hash_log2_hashmap_size",
        "21",
        *_adaptive_hash_resolution_args(),
        "--hash_encoding_dtype",
        "fp32",
        "--hash_encoding_preset",
        "smoothstep_grid",
        "--hash_grid_type",
        "Hash",
        *_attention_exam_decoder_args(),
    ]


def _attention_exam_fourier_args() -> list[str]:
    """Fourier encoding for methods A and D (decoder 4×256)."""
    return [
        "--input_projection",
        "fourier",
        "--projection_dim",
        "256",
        "--fourier_scale",
        "10",
        *_attention_exam_decoder_args(),
    ]


def _attention_run_name(base: str, lr_degradation: str, lr_degradations: tuple[str, ...]) -> str:
    """Legacy unsuffixed names when only ``area``; suffix ``_<mode>`` when multiple degradations."""
    if len(lr_degradations) == 1 and lr_degradations[0] == "area":
        return base
    return f"{base}_{lr_degradation}"


def _attention_experiment_definitions(
    *, include_fourier_band_attn: bool
) -> list[tuple[str, list[str]]]:
    """(base run name, method-specific CLI args without ``--run_name`` / ``--lr_degradation``)."""
    hg = _attention_exam_hashgrid_args()
    fo = _attention_exam_fourier_args()
    defs: list[tuple[str, list[str]]] = [
        (
            "fourier_mlp_baseline",
            [
                *fo,
                "--model",
                "mlp_tcnn",
                "--tcnn_mlp_dtype",
                "fp16",
            ],
        ),
        (
            "hashgrid_mlp_baseline",
            [
                *hg,
                "--model",
                "mlp_tcnn",
                "--tcnn_mlp_dtype",
                "fp16",
            ],
        ),
        (
            "hashgrid_level_attention",
            [
                *hg,
                "--model",
                "hash_attn",
                "--hash_attn_token_dim",
                "32",
                "--log_attention",
            ],
        ),
    ]
    if include_fourier_band_attn:
        defs.append(
            (
                "fourier_band_attention",
                [
                    *fo,
                    "--model",
                    "fourier_band_attn",
                    "--attn_token_dim",
                    "32",
                    "--fourier_num_bands",
                    "8",
                    "--log_attention",
                ],
            )
        )
    return defs


def build_attention_experiment_suite(
    *,
    include_fourier_band_attn: bool = False,
    lr_degradations: tuple[str, ...] = ("area",),
) -> list[Experiment]:
    """Fair exam ablation: Fourier vs HashGrid baselines vs attention decoders.

    All methods use decoder 4×256 (weight_decay 0). HashGrid uses 16 levels / log2
    hashmap 21 with per-scene base/max from LR size, etc.
    Attention runs enable ``--log_attention`` for ``attention_log.json``.

    Args:
        include_fourier_band_attn: If True, add method D (``fourier_band_attention``).
        lr_degradations: HR→LR operators passed as ``--lr_degradation`` (e.g. ``area``, ``s2_psf``).
    """
    exps: list[Experiment] = []
    for base_name, method_args in _attention_experiment_definitions(
        include_fourier_band_attn=include_fourier_band_attn
    ):
        for deg in lr_degradations:
            run_name = _attention_run_name(base_name, deg, lr_degradations)
            exps.append(
                Experiment(
                    name=run_name,
                    args=[
                        *method_args,
                        "--lr_degradation",
                        deg,
                        "--run_name",
                        run_name,
                    ],
                )
            )
    return exps


def build_attention_full_experiment_suite() -> list[Experiment]:
    """Methods A–D with both ``area`` and ``s2_psf`` HR→LR degradation."""
    return build_attention_experiment_suite(
        include_fourier_band_attn=True,
        lr_degradations=("area", "s2_psf"),
    )


def _suite_builders() -> dict[str, Callable[[], list[Experiment]]]:
    return {
        "superf": build_experiment_suite,
        "hash_attn": build_hash_attn_experiment_suite,
        "all": build_all_experiment_suite,
        "attention": build_attention_experiment_suite,
        "attention_full": build_attention_full_experiment_suite,
    }


def _metrics_row(m: dict, *, experiment: str, seed: int, metrics_path: Path) -> dict:
    ec = m.get("eval_crop", {}) or {}
    attn = m.get("attention", {}) or {}
    row = {
        "experiment": experiment,
        "seed": int(seed),
        "dataset": m.get("dataset"),
        "sample_id": m.get("sample_id"),
        "df": m.get("downsampling_factor"),
        "lr_degradation": m.get("lr_degradation"),
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
        "attn_entropy_mean": attn.get("entropy_mean"),
        "attn_fine_mass_last3": attn.get("fine_mass_last3"),
        "training_time_seconds": m.get("training", {}).get("training_time_seconds"),
        "time_per_iteration_seconds": m.get("training", {}).get("time_per_iteration_seconds"),
        "total_runtime_seconds": m.get("training", {}).get("total_runtime_seconds"),
        "final_recon_loss": m.get("training", {}).get("final_recon_loss"),
        "ttq_psnr_seconds": m.get("training", {}).get("ttq_psnr_seconds"),
        "ttq_psnr_iteration": m.get("training", {}).get("ttq_psnr_iteration"),
        "artifact_dir": str(metrics_path.parent),
        "command": m.get("command"),
    }
    if attn.get("mean_by_level") is not None:
        row["attn_mean_by_level"] = json.dumps(attn["mean_by_level"])
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Fourier vs HashGrid SuperF on satburst samples.")
    parser.add_argument(
        "--dataset",
        default="satburst_synth",
        choices=[
            "satburst_synth",
            "satburst_real",
            "worldstrat_test",
            "worldstrat_sweet",
            "worldstrat_bitter",
        ],
        help=(
            "satburst_*: scenes under data/ or data_real/. "
            "worldstrat_*: areas under worldstrat_datasets/ or worldstrat_test_data/ (hr/ + lr/)."
        ),
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
        "--worldstrat_data_root",
        type=str,
        default=None,
        help=(
            "Override WorldStrat area parent folder. Default: ``worldstrat_datasets/worldstrat_sweet`` "
            "or ``worldstrat_bitter``, or ``worldstrat_test_data`` for worldstrat_test."
        ),
    )
    parser.add_argument(
        "--limit_samples",
        type=int,
        default=0,
        help="If >0, only run the first N discovered samples (smoke tests).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip optimize.py when metrics.json already exists for that experiment.",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Only rebuild experiment_results CSV/JSON from existing metrics (no training).",
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
            "Pass-through to optimize.py --eval_crop_lr_size: top-left crop for metrics and "
            "comparison.png only (training stays on the full patch). "
            "Integer (e.g. 64) forces an explicit crop; ``auto`` (default) auto-detects the "
            "smallest LR side across discovered scenes (e.g. picks 64 across {64,128,256,512}); "
            "``0``/``off``/``none`` disables cropping."
        ),
    )
    parser.add_argument(
        "--train_crop_lr_size",
        type=int,
        default=0,
        help=(
            "Optional pass-through to optimize.py --train_crop_lr_size (OOM workaround only). "
            "Default 0 = train on the full sample. Do not set this to match --eval_crop_lr_size; "
            "eval crop is for cross-resolution metrics/viz only."
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
    parser.add_argument(
        "--suite",
        default="superf",
        choices=["superf", "hash_attn", "all", "attention", "attention_full"],
        help=(
            "Experiment set per scene: ``superf`` = Fourier + HashGrid MLP; "
            "``hash_attn`` = HashGrid MLP vs level-attention; "
            "``all`` = superf + hash_attn (legacy 4-run mix); "
            "``attention`` = exam ablation A/B/C (area HR→LR only); "
            "``attention_full`` = A/B/C/D with both ``area`` and ``s2_psf`` HR→LR (8 runs/scene)."
        ),
    )
    args = parser.parse_args()

    dataset = str(args.dataset)
    device = str(args.device)

    is_worldstrat = dataset in WORLDSTRAT_DATASETS
    if is_worldstrat:
        if args.worldstrat_data_root:
            data_root = str(args.worldstrat_data_root).strip().rstrip("/")
        else:
            data_root = _default_worldstrat_data_root(dataset)
    elif args.satburst_data_root:
        data_root = str(args.satburst_data_root).strip().rstrip("/")
    elif dataset == "satburst_real":
        data_root = "data_real"
    else:
        data_root = "data"

    common = [
        "--dataset",
        dataset,
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
    if is_worldstrat:
        common.extend(["--worldstrat_data_root", data_root, "--df", str(args.df)])
    else:
        common.extend(
            [
                "--satburst_data_root",
                data_root,
                "--df",
                str(args.df),
                "--lr_shift",
                str(args.lr_shift),
                "--aug",
                str(args.aug),
            ]
        )

    suite_name = str(args.suite)
    suite = _suite_builders()[suite_name]()
    repo_root = Path(__file__).resolve().parents[1]
    if is_worldstrat:
        sample_ids = _discover_worldstrat_samples(repo_root, data_root)
        if not sample_ids:
            raise RuntimeError(
                f"No WorldStrat areas found under {repo_root / data_root} "
                "(expected <area>/hr/ and <area>/lr/)."
            )
    else:
        sample_ids = _discover_satburst_samples(
            repo_root,
            data_root,
            int(args.df),
            float(args.lr_shift),
            str(args.aug),
            max_lr_side=int(args.max_lr_side),
        )
        if not sample_ids:
            raise RuntimeError(
                f"No samples found under {repo_root / data_root} with scale folder "
                f"scale_{args.df}_shift_{float(args.lr_shift):.1f}px_aug_{args.aug}"
            )
    if args.limit_samples and args.limit_samples > 0:
        sample_ids = sample_ids[: int(args.limit_samples)]

    if dataset == "satburst_real":
        print(
            "Note (data_real): scenes can use different LR/HR sizes; metrics are per-scene. "
            "See data_real/README.txt.",
            flush=True,
        )
    elif is_worldstrat:
        print(
            f"WorldStrat: {len(sample_ids)} area(s) under {repo_root / data_root} "
            f"(hr/ + lr/ per area). See docs/WORLDSTRAT_EXPERIMENTS.md.",
            flush=True,
        )

    # Resolve --eval_crop_lr_size: int / auto / 0|off|none. ``auto`` picks the smallest LR side
    # across discovered scenes so a 64/128/256/512 sweep is anchored on the 64 patch area.
    eval_crop_raw = str(args.eval_crop_lr_size).strip().lower()
    scale_leaf = f"scale_{int(args.df)}_shift_{float(args.lr_shift):.1f}px_aug_{args.aug}"
    if eval_crop_raw in {"0", "off", "none", "no", "false", ""}:
        eval_crop_lr_size = 0
    elif eval_crop_raw == "auto":
        if is_worldstrat:
            eval_crop_lr_size = 64
            print(
                "WorldStrat: using --eval_crop_lr_size=64 (center-cropped LR patches).",
                flush=True,
            )
        else:
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

    if eval_crop_lr_size > 0 and int(args.train_crop_lr_size) <= 0:
        print(
            f"Eval crop {eval_crop_lr_size}px (metrics + comparison.png); "
            "training on full patch → comparison_full.png when larger.",
            flush=True,
        )
    elif int(args.train_crop_lr_size) > 0:
        print(
            f"Training crop {int(args.train_crop_lr_size)}px (--train_crop_lr_size OOM workaround).",
            flush=True,
        )

    suite_labels = {
        "superf": "Fourier + HashGrid MLP",
        "hash_attn": "HashGrid MLP vs level-attention",
        "all": "superf + hash_attn (legacy full comparison)",
        "attention": "exam ablation A/B/C (area HR→LR)",
        "attention_full": "exam ablation A/B/C/D (area + s2_psf HR→LR)",
    }
    print(
        f"Suite {suite_name!r} ({suite_labels.get(suite_name, suite_name)}): "
        f"{len(suite)} experiment(s)/scene, data_root={data_root!r}",
        flush=True,
    )

    all_rows: list[dict] = []

    for sample_id in sample_ids:
        print(f"\n=== Sample: {sample_id} ({dataset}) ===", flush=True)
        sample_rows: list[dict] = []

        for exp in suite:
            metrics_path = (
                repo_root / "single_samples" / dataset / sample_id / exp.name / "metrics.json"
            )
            if args.aggregate_only:
                if not metrics_path.exists():
                    print(f"  Skip {exp.name}: no metrics at {metrics_path}", flush=True)
                    continue
            elif args.skip_existing and metrics_path.exists():
                print(f"  Skip {exp.name}: reusing {metrics_path}", flush=True)
            else:
                exp_args = [*common, "--sample_id", sample_id, *exp.args]
                if eval_crop_lr_size > 0:
                    exp_args = [
                        *exp_args,
                        "--eval_crop_lr_size",
                        str(int(eval_crop_lr_size)),
                        "--eval_crop_anchor",
                        "topleft",
                    ]
                if int(args.train_crop_lr_size) > 0:
                    exp_args = [
                        *exp_args,
                        "--train_crop_lr_size",
                        str(int(args.train_crop_lr_size)),
                    ]
                cmd = [sys.executable, str(repo_root / "optimize.py"), *exp_args]
                _run(cmd, cuda_expandable_segments=bool(args.cuda_expandable_segments))
                metrics_path.parent.mkdir(parents=True, exist_ok=True)
                (metrics_path.parent / "run_command.txt").write_text(
                    "$ " + " ".join(cmd) + "\n", encoding="utf-8"
                )
                if not metrics_path.exists():
                    raise FileNotFoundError(f"Expected metrics at {metrics_path}")
            if not metrics_path.exists():
                raise FileNotFoundError(f"Expected metrics at {metrics_path}")
            m = _read_metrics(metrics_path)
            row = _metrics_row(
                m, experiment=exp.name, seed=int(args.seed), metrics_path=metrics_path
            )
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
