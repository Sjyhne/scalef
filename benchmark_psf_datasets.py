#!/usr/bin/env python3
"""Benchmark data PSF × train PSF × hash_max on synth bursts.

Full factorial (default ``--mode quick``):

- **Data construction** (2): DSen2 ``satburstsynth_data`` vs meter PSF ``satsynthburst_data_s2psfm``
- **Train LR degradation** (2): ``s2_psf`` vs ``s2_psf_m``
- **hash_max** (2): auto (0 → LR side) vs 2×LR

→ 8 runs at LR 224. Use ``--matched-only`` for the old 4-run matched-pair grid.

Examples:

    # Full 2×2×2 grid (8 runs)
    python benchmark_psf_datasets.py --mode quick --dry-run

    # Matched pairs only (4 runs): dsen2+s2_psf, psfm+s2_psf_m
    python benchmark_psf_datasets.py --mode quick --matched-only --dry-run

    # Ablation: factorial × lr_relative hash grid × all LR folder sizes
    python benchmark_psf_datasets.py --mode ablation --all-lr-sizes --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from argparse import Namespace
from copy import copy
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
OPTIMIZE_PY = REPO_ROOT / "optimize.py"

LR_FOLDER_RE = re.compile(
    r"^scale_(?P<df>\d+)_lr(?P<lr>\d+)_shift_(?P<shift>[\d.]+)px_aug_(?P<aug>\w+)$"
)

# (tag, satburst_data_root) — tag labels how LR bursts were synthesized.
DATA_SOURCES: tuple[tuple[str, str], ...] = (
    ("dsen2", "satburstsynth_data"),
    ("psfm", "satsynthburst_data_s2psfm"),
)

TRAIN_DEGRADATIONS: tuple[str, ...] = ("s2_psf", "s2_psf_m")

# Legacy matched pairs: (data_tag, data_root, train_degradation)
MATCHED_PSF_DATASETS: tuple[tuple[str, str, str], ...] = (
    ("dsen2", "satburstsynth_data", "s2_psf"),
    ("psfm", "satsynthburst_data_s2psfm", "s2_psf_m"),
)


def satburst_scene_dir(args: Namespace) -> str:
    lr_size = int(getattr(args, "lr_size", 0) or 0)
    if lr_size > 0:
        sub = (
            f"scale_{int(args.df)}_lr{lr_size}_shift_{float(args.lr_shift):.1f}px_aug_{args.aug}"
        )
    else:
        sub = f"scale_{int(args.df)}_shift_{float(args.lr_shift):.1f}px_aug_{args.aug}"
    return f"{args.satburst_data_root}/{args.sample_id}/{sub}"


def single_sample_output_dir(args: Namespace) -> Path:
    p = Path("single_samples") / args.dataset / str(args.sample_id)
    if getattr(args, "run_name", None):
        return p / str(args.run_name)
    return p


def default_hash_max_values(lr_side: int) -> list[int]:
    if lr_side <= 0:
        return [0, 32, 48, 64, 96, 128, 192, 256]
    vals = [0]
    for div in (8, 4, 2, 1):
        vals.append(max(16, lr_side // div))
    vals.append(lr_side * 2)
    return sorted(set(vals))


def quick_hash_max_values(lr_side: int) -> list[int]:
    """Auto + 2×S (best quality point from prior ablation)."""
    return [0, int(lr_side) * 2]


def discover_satburst_lr_sizes(
    data_root: Path | str,
    sample_id: str,
    *,
    df: int,
    lr_shift: float,
    aug: str,
) -> list[int]:
    scene_root = Path(data_root) / str(sample_id)
    if not scene_root.is_dir():
        return []
    shift_s = f"{float(lr_shift):.1f}"
    sizes: set[int] = set()
    for p in scene_root.iterdir():
        if not p.is_dir():
            continue
        m = LR_FOLDER_RE.match(p.name)
        if m is None:
            continue
        if int(m.group("df")) != int(df):
            continue
        if m.group("shift") != shift_s:
            continue
        if m.group("aug") != str(aug):
            continue
        if (p / "transform_log.json").is_file():
            sizes.add(int(m.group("lr")))
    return sorted(sizes)


def read_scene_degradation(scene_dir: Path) -> str | None:
    meta = scene_dir / "synth_export_meta.json"
    if not meta.is_file():
        return None
    with meta.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return str(data.get("degradation")) if isinstance(data, dict) else None


def _train_degradation_tag(train_degradation: str) -> str:
    return "s2psfm" if str(train_degradation) == "s2_psf_m" else "s2psf"


def _run_name(
    hash_max: int,
    prefix: str,
    *,
    lr_size: int | None = None,
    data_tag: str | None = None,
    train_degradation: str | None = None,
) -> str:
    parts = [prefix]
    if data_tag:
        parts.append(f"data_{data_tag}")
    if train_degradation:
        parts.append(f"train_{_train_degradation_tag(train_degradation)}")
    if lr_size is not None:
        parts.append(f"lr{int(lr_size)}")
    base = "_".join(parts)
    tag = "auto" if int(hash_max) <= 0 else f"{int(hash_max):04d}"
    return f"{base}_hashmax_{tag}"


def build_optimize_command(args: Namespace, hash_max: int) -> list[str]:
    lr_size = int(getattr(args, "lr_size", 0) or 0)
    run_name = _run_name(
        hash_max,
        args.run_prefix,
        lr_size=lr_size if getattr(args, "_multi_lr", False) else None,
        data_tag=str(getattr(args, "data_tag", "")) or None,
        train_degradation=str(getattr(args, "lr_degradation", "")) or None,
    )
    cmd = [
        sys.executable,
        str(OPTIMIZE_PY),
        "--dataset",
        args.dataset,
        "--sample_id",
        str(args.sample_id),
        "--df",
        str(args.df),
        "--lr_shift",
        str(args.lr_shift),
        "--aug",
        args.aug,
        "--num_samples",
        str(args.num_samples),
        "--iters",
        str(args.iters),
        "--device",
        str(args.device),
        "--seed",
        str(args.seed),
        "--input_projection",
        "hashgrid",
        "--model",
        args.model,
        "--supervision_channels",
        str(args.supervision_channels),
        "--lr_degradation",
        str(args.lr_degradation),
        "--hash_max_resolution",
        str(int(hash_max)),
        "--run_name",
        run_name,
    ]
    if lr_size > 0:
        cmd.extend(["--lr_size", str(lr_size)])
    if args.satburst_data_root:
        cmd.extend(["--satburst_data_root", str(args.satburst_data_root)])
    if args.no_multiband_diagnostics:
        cmd.append("--no_multiband_diagnostics")
    if args.skip_artifacts:
        cmd.append("--skip_artifacts")
    if args.skip_eval:
        cmd.append("--skip_eval")
    if int(getattr(args, "eval_every", 500)) != 100:
        cmd.extend(["--eval_every", str(int(args.eval_every))])
    if args.optimize_extra:
        cmd.extend(args.optimize_extra)
    return cmd


def _resolve_lr_sizes(args: Namespace) -> list[int]:
    if args.lr_sizes:
        return sorted({max(1, int(s)) for s in args.lr_sizes})
    if args.all_lr_sizes:
        found = discover_satburst_lr_sizes(
            args.satburst_data_root,
            str(args.sample_id),
            df=int(args.df),
            lr_shift=float(args.lr_shift),
            aug=str(args.aug),
        )
        if not found:
            raise FileNotFoundError(
                f"No LR folders under {args.satburst_data_root}/{args.sample_id} "
                f"for df={args.df} shift={args.lr_shift} aug={args.aug}"
            )
        return found
    return [max(1, int(args.lr_size))]


def _resolve_hash_max_list(args: Namespace, lr_side: int) -> list[int]:
    if args.hash_max:
        return [max(0, int(v)) for v in args.hash_max]
    if str(args.mode) == "quick":
        return quick_hash_max_values(lr_side)
    if str(args.hash_max_preset) == "lr_relative":
        return default_hash_max_values(lr_side)
    preset = str(args.hash_max_preset)
    if preset == "small":
        return [0, 32, 48, 64]
    if preset == "medium":
        return [0, 64, 128, 192, 256]
    if preset == "large":
        return [0, 128, 256, 384, 512, 768, 1024]
    raise ValueError(f"Unknown preset: {preset!r}")


def build_plan_for_cell(
    base_args: Namespace,
    *,
    data_tag: str,
    data_root: str,
    train_degradation: str,
) -> list[dict]:
    args = copy(base_args)
    args.satburst_data_root = str(data_root)
    args.lr_degradation = str(train_degradation)
    args.data_tag = str(data_tag)

    lr_sizes = _resolve_lr_sizes(args)
    multi_lr = len(lr_sizes) > 1
    plan: list[dict] = []

    for lr_size in lr_sizes:
        run_args = copy(args)
        run_args.lr_size = int(lr_size)
        run_args._multi_lr = multi_lr or str(base_args.mode) == "ablation"
        scene = Path(satburst_scene_dir(run_args))
        if not scene.is_dir():
            raise FileNotFoundError(f"Scene directory not found: {scene}")

        lr_side = int(lr_size)
        hash_max_values = _resolve_hash_max_list(run_args, lr_side)
        data_deg = read_scene_degradation(scene)

        for hm in hash_max_values:
            run_name = _run_name(
                hm,
                run_args.run_prefix,
                lr_size=lr_size if run_args._multi_lr else None,
                data_tag=data_tag,
                train_degradation=train_degradation,
            )
            entry_args = copy(run_args)
            entry_args.run_name = run_name
            plan.append(
                {
                    "data_tag": str(data_tag),
                    "satburst_data_root": str(data_root),
                    "data_degradation": data_deg,
                    "train_lr_degradation": str(train_degradation),
                    "lr_size": int(lr_size),
                    "lr_side": int(lr_side),
                    "scene": str(scene),
                    "hash_max_requested": int(hm),
                    "run_name": run_name,
                    "run_args": entry_args,
                    "command": build_optimize_command(entry_args, hm),
                }
            )
    return plan


def build_factorial_plan(base_args: Namespace) -> list[dict]:
    data_sources = list(DATA_SOURCES)
    train_degs = list(TRAIN_DEGRADATIONS)

    if base_args.data_sources:
        allowed = set(base_args.data_sources)
        data_sources = [(t, r) for t, r in data_sources if t in allowed]
    if base_args.train_degradations:
        allowed = set(base_args.train_degradations)
        train_degs = [d for d in train_degs if d in allowed]

    plan: list[dict] = []
    for data_tag, data_root in data_sources:
        for train_deg in train_degs:
            plan.extend(
                build_plan_for_cell(
                    base_args,
                    data_tag=data_tag,
                    data_root=data_root,
                    train_degradation=train_deg,
                )
            )
    return plan


def build_matched_plan(base_args: Namespace) -> list[dict]:
    selected = set(base_args.data_sources) if base_args.data_sources else None
    plan: list[dict] = []
    for data_tag, data_root, train_deg in MATCHED_PSF_DATASETS:
        if selected and data_tag not in selected:
            continue
        plan.extend(
            build_plan_for_cell(
                base_args,
                data_tag=data_tag,
                data_root=data_root,
                train_degradation=train_deg,
            )
        )
    return plan


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--mode",
        choices=("quick", "ablation"),
        default="quick",
        help="quick = LR 224, hash_max auto+2S, full 2×2×2 factorial; ablation = lr_relative grid.",
    )
    p.add_argument("--hash-max", dest="hash_max", nargs="*", type=int)
    p.add_argument("--hash-max-preset", default="lr_relative", choices=("lr_relative", "small", "medium", "large"))
    p.add_argument("--run-prefix", default="bench")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--matched-only",
        action="store_true",
        help="Only matched data/train PSF pairs (4 runs in quick mode). Default is full factorial (8 runs).",
    )
    p.add_argument(
        "--data-sources",
        dest="data_sources",
        nargs="*",
        default=None,
        help="Subset of data construction tags: dsen2 psfm (default: both).",
    )
    p.add_argument(
        "--train-degradations",
        dest="train_degradations",
        nargs="*",
        default=None,
        choices=["s2_psf", "s2_psf_m"],
        help="Subset of train --lr_degradation values (default: both).",
    )
    p.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Deprecated alias for --data-sources (dsen2 psfm).",
    )

    lr = p.add_argument_group("scene")
    lr.add_argument("--lr-sizes", dest="lr_sizes", nargs="*", type=int)
    lr.add_argument("--all-lr-sizes", action="store_true")
    lr.add_argument("--lr-size", dest="lr_size", type=int, default=224)
    p.add_argument("--sample-id", dest="sample_id", default="UNHCR-YEMs035290_rgb")
    p.add_argument("--dataset", default="satburst_synth")
    p.add_argument("--df", type=int, default=4)
    p.add_argument("--lr-shift", dest="lr_shift", type=float, default=1.0)
    p.add_argument("--aug", default="none")
    p.add_argument("--num-samples", dest="num_samples", type=int, default=12)
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--device", default="0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", default="mlp_tcnn", choices=["mlp", "mlp_tcnn"])
    p.add_argument("--supervision-channels", dest="supervision_channels", type=int, default=3)
    p.add_argument("--eval-every", dest="eval_every", type=int, default=500)
    p.add_argument("--no-multiband-diagnostics", dest="no_multiband_diagnostics", action="store_true", default=True)
    p.add_argument("--multiband-diagnostics", dest="no_multiband_diagnostics", action="store_false")
    p.add_argument("--skip-artifacts", dest="skip_artifacts", action="store_true")
    p.add_argument("--skip-eval", dest="skip_eval", action="store_true")
    p.add_argument("optimize_extra", nargs=argparse.REMAINDER)
    return p


def _load_metrics(path: Path) -> dict:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _spot_from_metrics(metrics: dict) -> dict:
    spot = metrics.get("fixed_spot")
    return spot if isinstance(spot, dict) else {}


def _write_spot_lr_comparison(results: list[dict], out_dir: Path) -> Path | None:
    """Pivot fixed-spot metrics by LR size for runs that share the same config."""
    rows: list[dict] = []
    for row in results:
        spot = row.get("fixed_spot") or {}
        if not spot:
            continue
        rows.append(
            {
                "data_tag": row.get("data_tag"),
                "data_degradation": row.get("data_degradation"),
                "train_lr_degradation": row.get("train_lr_degradation"),
                "hash_max_requested": row.get("hash_max_requested"),
                "lr_size": row.get("lr_size"),
                "run_name": row.get("run_name"),
                "spot_hr_pixels": spot.get("hr_pixels"),
                "spot_model_psnr": spot.get("model_psnr"),
                "spot_bilinear_psnr": spot.get("bilinear_psnr"),
                "spot_psnr_improvement": spot.get("psnr_improvement"),
                "spot_model_ssim": spot.get("model_ssim"),
                "spot_bilinear_ssim": spot.get("bilinear_ssim"),
                "spot_model_lpips": spot.get("model_lpips"),
                "spot_bilinear_lpips": spot.get("bilinear_lpips"),
            }
        )
    if not rows:
        return None

    csv_path = out_dir / "spot_by_lr_size.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (str(r["data_tag"]), int(r["lr_size"]))))
    return csv_path


def main() -> None:
    if not OPTIMIZE_PY.is_file():
        raise SystemExit(f"Missing {OPTIMIZE_PY} — restore optimize.py before running benchmarks.")

    args = build_arg_parser().parse_args()
    if args.optimize_extra and args.optimize_extra[0] == "--":
        args.optimize_extra = args.optimize_extra[1:]

    if args.datasets and not args.data_sources:
        args.data_sources = list(args.datasets)
    # Legacy names from the first benchmark script.
    if args.data_sources:
        legacy = {"s2_psf": "dsen2", "s2_psf_m": "psfm"}
        args.data_sources = [legacy.get(t, t) for t in args.data_sources]

    if args.matched_only:
        plan = build_matched_plan(args)
    else:
        plan = build_factorial_plan(args)

    if not plan:
        raise SystemExit("No benchmark runs planned (check --data-sources / --train-degradations).")

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.output_dir or Path("single_samples") / args.dataset / str(args.sample_id) / f"benchmark_psf_{args.mode}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Mode: {args.mode}")
    print(f"Summary dir: {out_dir}")
    print(f"Total runs: {len(plan)}")
    for row in plan:
        print(
            f"  [data={row['data_tag']}] lr={row['lr_size']:>3} hash_max={row['hash_max_requested']:>4} "
            f"train={row['train_lr_degradation']} synth={row['data_degradation']} → {row['run_name']}"
        )

    plan_path = out_dir / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "mode": args.mode,
                "runs": [{k: v for k, v in r.items() if k not in ("command", "run_args")} for r in plan],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if args.dry_run:
        print("\nDry run — commands:")
        for row in plan:
            print(" ".join(row["command"]))
        print(f"\nWrote {plan_path}")
        return

    results: list[dict] = []
    for i, row in enumerate(plan, start=1):
        print(f"\n[{i}/{len(plan)}] {row['run_name']} …", flush=True)
        t0 = time.perf_counter()
        proc = subprocess.run(row["command"], cwd=REPO_ROOT, check=False)
        wall = time.perf_counter() - t0
        metrics_path = single_sample_output_dir(row["run_args"]) / "metrics.json"
        metrics = _load_metrics(metrics_path)
        spot = _spot_from_metrics(metrics)
        iters = metrics.get("completed_iters") or 0
        train_t = metrics.get("training_time_seconds")
        sec_per_iter = (float(train_t) / float(iters)) if train_t and iters else None
        results.append(
            {
                **{k: row[k] for k in row if k not in ("command", "run_args")},
                "metrics_path": str(metrics_path),
                "exit_code": int(proc.returncode),
                "wall_seconds": wall,
                "model_psnr": metrics.get("model_psnr"),
                "bilinear_psnr": metrics.get("bilinear_psnr"),
                "final_test_psnr": metrics.get("final_test_psnr"),
                "fixed_spot": spot or None,
                "spot_model_psnr": spot.get("model_psnr"),
                "spot_bilinear_psnr": spot.get("bilinear_psnr"),
                "spot_psnr_improvement": spot.get("psnr_improvement"),
                "spot_model_ssim": spot.get("model_ssim"),
                "spot_model_lpips": spot.get("model_lpips"),
                "training_time_seconds": train_t,
                "completed_iters": iters,
                "time_per_iteration_seconds": sec_per_iter,
            }
        )
        psnr = results[-1].get("spot_model_psnr") or results[-1]["model_psnr"]
        psnr_s = f"{psnr:.2f}" if isinstance(psnr, (int, float)) else "n/a"
        print(f"  exit={proc.returncode} psnr={psnr_s} ({wall:.1f}s)", flush=True)

    summary_json = out_dir / "summary.json"
    summary_json.write_text(json.dumps({"mode": args.mode, "results": results}, indent=2), encoding="utf-8")

    fieldnames = [
        "data_tag",
        "data_degradation",
        "train_lr_degradation",
        "satburst_data_root",
        "lr_size",
        "hash_max_requested",
        "run_name",
        "model_psnr",
        "bilinear_psnr",
        "spot_hr_pixels",
        "spot_model_psnr",
        "spot_bilinear_psnr",
        "spot_psnr_improvement",
        "spot_model_ssim",
        "spot_model_lpips",
        "training_time_seconds",
        "time_per_iteration_seconds",
        "wall_seconds",
        "exit_code",
        "metrics_path",
    ]
    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            spot = row.get("fixed_spot") or {}
            flat = {k: row.get(k) for k in fieldnames}
            flat["spot_hr_pixels"] = spot.get("hr_pixels")
            writer.writerow(flat)

    spot_csv = _write_spot_lr_comparison(results, out_dir)

    print(f"\nWrote {summary_json}")
    print(f"Wrote {csv_path}")
    if spot_csv is not None:
        print(f"Wrote {spot_csv}")
    failed = [r for r in results if r["exit_code"] != 0]
    if failed:
        raise SystemExit(f"{len(failed)} run(s) failed; see {summary_json}")


if __name__ == "__main__":
    main()
