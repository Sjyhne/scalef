#!/usr/bin/env python3
"""Ablation sweeps for hash-grid and training hyperparameters.

By default only ``hash_max`` is swept (``--sweep-axes max``). Use ``--sweep-axes``
to add more dimensions; each listed axis is factorial with the others.

Recommended **staged** workflow (avoids 100+ run grids):

1. ``--sweep-axes max`` — find coarsest/finest grid scale (blocking usually here)
2. ``--sweep-axes base --fixed-hash-max 256`` — coarse-level coverage at fixed max
3. ``--sweep-axes levels --fixed-hash-max 256`` — octave count (8/12/16/20)
4. ``--sweep-axes interp --fixed-hash-max 256`` — linear vs smoothstep
5. ``--sweep-axes lr --fixed-hash-max 256`` — training rate (artifact overfitting)
6. ``--sweep-axes sigma --fixed-hash-max 256`` — Zip-NeRF level downweighting
   (``--hash_level_sigma``; 0 = off, defaults derived from LR footprint)

Note: ``hash_base=0`` means auto ``max//4``, so sweeping max alone also moves base.

Examples:

    # Stage 1: hash_max only (default)
    python benchmark_hashgrid_resolution.py \\
      --sample-id "UNHCR-TURs005204" --mode ablation --dry-run

    # Stage 2–4: grid geometry at best max=256
    python benchmark_hashgrid_resolution.py \\
      --sample-id "UNHCR-TURs005204" \\
      --sweep-axes base --fixed-hash-max 256 --dry-run

    # Stage 5: learning rate at winner grid config
    python benchmark_hashgrid_resolution.py \\
      --sample-id "UNHCR-TURs005204" \\
      --sweep-axes lr --fixed-hash-max 256 --fixed-hash-base 64 \\
      --hash-n-levels 16 --hash-interpolation linear --dry-run

    # Full factorial (small grids only): max × base × levels × interp × lr
    python benchmark_hashgrid_resolution.py \\
      --sample-id "UNHCR-TURs005204" \\
      --sweep-axes max,base,levels,interp,lr \\
      --hash-max 128 256 --hash-base 0 64 --hash-n-levels 12 16 \\
      --hash-interpolation linear smoothstep --learning-rate 1e-3 2e-3 \\
      --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from argparse import Namespace
from copy import copy
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
OPTIMIZE_PY = REPO_ROOT / "optimize.py"
WORLDSTRAT_LIST = REPO_ROOT / "worldstrat_datasets.txt"


def _hr_side_for_dataset(dataset: str) -> int:
  """Typical HR side used to pick sensible hash_max presets."""
  if dataset in {"worldstrat_test", "worldstrat_sweet", "worldstrat_bitter"}:
    return 256
  return 0


def default_hash_max_values(hr_side: int) -> list[int]:
  if hr_side <= 0:
    return [0, 32, 48, 64, 96, 128, 192, 256]
  vals = {0, hr_side // 4, hr_side // 2, hr_side, hr_side * 3 // 2, hr_side * 2}
  return sorted(v for v in vals if v >= 0)


def quick_hash_max_values(hr_side: int) -> list[int]:
  if hr_side <= 0:
    return [0, 128, 256]
  return [0, hr_side // 2, hr_side]


def ablation_hash_max_values(hr_side: int) -> list[int]:
  if hr_side <= 0:
    return default_hash_max_values(0)
  vals = {0}
  for num in (8, 6, 4, 3, 2, 1):
    vals.add(max(16, hr_side * 2 // num))
  vals.add(hr_side * 2)
  return sorted(vals)


def default_hash_base_values(representative_max: int) -> list[int]:
  mx = max(16, int(representative_max) if int(representative_max) > 0 else 256)
  return sorted({0, max(8, mx // 8), max(8, mx // 4), max(8, mx // 2)})


def default_hash_n_levels_values() -> list[int]:
  return [8, 12, 16, 20]


def default_learning_rate_values() -> list[float]:
  return [1e-3, 2e-3, 5e-3]


def default_hash_level_sigma_values(hr_side: int, df: int, lr_size: int = 0) -> list[float]:
  """0 (off) plus footprint-derived sigmas: s = 1/(sqrt(12)*W_lr) for area degradation."""
  w_lr = int(lr_size) if int(lr_size or 0) > 0 else (int(hr_side) // max(1, int(df)) if int(hr_side or 0) > 0 else 64)
  s = 1.0 / (math.sqrt(12.0) * w_lr)
  return [0.0, round(0.5 * s, 6), round(s, 6), round(2.0 * s, 6)]


def parse_sweep_axes(spec: str) -> set[str]:
  allowed = {"max", "base", "levels", "interp", "lr", "sigma"}
  axes = {part.strip().lower() for part in str(spec).split(",") if part.strip()}
  unknown = axes - allowed
  if unknown:
    raise ValueError(f"Unknown sweep axis(es): {sorted(unknown)}; allowed: {sorted(allowed)}")
  return axes or {"max"}


def discover_worldstrat_set(name: str) -> list[str]:
  name = str(name).lower().strip()
  if name in {"sweet", "bitter"}:
    root = REPO_ROOT / "worldstrat_datasets" / f"worldstrat_{name}"
    if root.is_dir():
      return sorted(p.name for p in root.iterdir() if p.is_dir())
  if name == "test":
    root = REPO_ROOT / "worldstrat_test_data"
    if root.is_dir():
      return sorted(p.name for p in root.iterdir() if p.is_dir())

  # Fallback: parse worldstrat_datasets.txt
  if not WORLDSTRAT_LIST.is_file():
    return []
  section = None
  out: list[str] = []
  for line in WORLDSTRAT_LIST.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line:
      continue
    low = line.lower()
    if low.startswith("worldstrat sweet"):
      section = "sweet"
      continue
    if low.startswith("worldstrat bitter"):
      section = "bitter"
      continue
    if section == name and ". " in line:
      _, sid = line.split(". ", 1)
      out.append(sid.strip())
  return out


def resolve_sample_ids(args: Namespace) -> list[str]:
  if args.sample_ids:
    return list(args.sample_ids)
  if args.worldstrat_set:
    ids = discover_worldstrat_set(args.worldstrat_set)
    if not ids:
      raise SystemExit(f"No samples found for worldstrat set {args.worldstrat_set!r}")
    if args.max_samples and args.max_samples > 0:
      ids = ids[: int(args.max_samples)]
    return ids
  if args.sample_id:
    return [str(args.sample_id)]
  raise SystemExit("Provide --sample-id, --sample-ids, or --worldstrat-set")


def resolve_dataset_for_optimize(args: Namespace) -> str:
  ds = str(args.dataset).lower()
  if ds in {"worldstrat_sweet", "worldstrat_bitter"}:
    return "worldstrat_test"
  return ds


def single_sample_output_dir(args: Namespace) -> Path:
  p = Path("single_samples") / resolve_dataset_for_optimize(args) / str(args.sample_id)
  if getattr(args, "run_name", None):
    return p / str(args.run_name)
  return p


def _run_name(
  *,
  prefix: str,
  sample_id: str,
  hash_max: int,
  hash_base: int,
  hash_n_levels: int,
  hash_interpolation: str,
  learning_rate: float,
  hash_level_sigma: float = 0.0,
) -> str:
  hm = "auto" if int(hash_max) <= 0 else f"{int(hash_max):04d}"
  hb = "auto" if int(hash_base) <= 0 else f"{int(hash_base):04d}"
  interp = str(hash_interpolation).lower().strip()[:4]
  lr_tag = f"{float(learning_rate):.0e}".replace("+", "").replace("e-0", "e-")
  sid = sample_id.replace(" ", "_")
  name = f"{prefix}_{sid}_hmax_{hm}_hbase_{hb}_L{int(hash_n_levels)}_{interp}_lr{lr_tag}"
  if float(hash_level_sigma) > 0.0:
    sig_tag = f"{float(hash_level_sigma):.1e}".replace("+", "").replace("e-0", "e-")
    name += f"_sig{sig_tag}"
  return name


def _resolve_hash_max_list(args: Namespace, hr_side: int) -> list[int]:
  if args.hash_max:
    return sorted({max(0, int(v)) for v in args.hash_max})
  mode = str(args.mode).lower()
  if mode == "quick":
    return quick_hash_max_values(hr_side)
  if mode == "ablation":
    return ablation_hash_max_values(hr_side)
  raise ValueError(f"Unknown mode: {args.mode!r}")


def _resolve_hash_base_list(args: Namespace, *, sweeping: bool, representative_max: int) -> list[int]:
  if args.hash_base:
    return sorted({max(0, int(v)) for v in args.hash_base})
  if sweeping:
    return default_hash_base_values(representative_max)
  return [int(getattr(args, "fixed_hash_base", 0) or 0)]


def _resolve_hash_n_levels_list(args: Namespace, *, sweeping: bool) -> list[int]:
  if args.hash_n_levels:
    return sorted({max(1, int(v)) for v in args.hash_n_levels})
  if sweeping:
    return default_hash_n_levels_values()
  return [int(args.hash_n_levels_default)]


def _resolve_hash_interpolation_list(args: Namespace, *, sweeping: bool) -> list[str]:
  if args.hash_interpolation:
    return [str(v).lower().strip() for v in args.hash_interpolation]
  if sweeping:
    return ["smoothstep", "linear"]
  return [str(args.hash_interpolation_default).lower().strip()]


def _resolve_learning_rate_list(args: Namespace, *, sweeping: bool) -> list[float]:
  if args.learning_rate:
    return [float(v) for v in args.learning_rate]
  if sweeping:
    return default_learning_rate_values()
  return [float(args.learning_rate_default)]


def _resolve_hash_level_sigma_list(args: Namespace, *, sweeping: bool, hr_side: int) -> list[float]:
  if args.hash_level_sigma:
    return [max(0.0, float(v)) for v in args.hash_level_sigma]
  if sweeping:
    return default_hash_level_sigma_values(hr_side, int(args.df), int(args.lr_size or 0))
  return [max(0.0, float(args.hash_level_sigma_default))]


def build_optimize_command(
  args: Namespace,
  *,
  sample_id: str,
  hash_max: int,
  hash_base: int,
  hash_n_levels: int,
  hash_interpolation: str,
  learning_rate: float,
  hash_level_sigma: float = 0.0,
) -> list[str]:
  run_args = copy(args)
  run_args.sample_id = str(sample_id)
  run_name = _run_name(
    prefix=args.run_prefix,
    sample_id=sample_id,
    hash_max=hash_max,
    hash_base=hash_base,
    hash_n_levels=hash_n_levels,
    hash_interpolation=hash_interpolation,
    learning_rate=learning_rate,
    hash_level_sigma=hash_level_sigma,
  )
  run_args.run_name = run_name

  dataset = resolve_dataset_for_optimize(args)
  cmd = [
    sys.executable,
    str(OPTIMIZE_PY),
    "--dataset",
    dataset,
    "--sample_id",
    str(sample_id),
    "--df",
    str(args.df),
    "--scale_factor",
    str(args.scale_factor),
    "--num_samples",
    str(args.num_samples),
    "--iters",
    str(args.iters),
    "--device",
    str(args.device),
    "--seed",
    str(args.seed),
    "--learning_rate",
    str(float(learning_rate)),
    "--input_projection",
    str(args.input_projection),
    "--model",
    str(args.model),
    "--supervision_channels",
    str(args.supervision_channels),
    "--lr_degradation",
    str(args.lr_degradation),
    "--hash_max_resolution",
    str(int(hash_max)),
    "--hash_base_resolution",
    str(int(hash_base)),
    "--hash_n_levels",
    str(int(hash_n_levels)),
    "--hash_interpolation",
    str(hash_interpolation),
    "--hash_level_sigma",
    str(float(hash_level_sigma)),
    "--hash_n_features_per_level",
    str(int(args.hash_n_features_per_level)),
    "--hash_log2_hashmap_size",
    str(int(args.hash_log2_hashmap_size)),
    "--run_name",
    run_name,
    "--eval_every",
    str(int(args.eval_every)),
  ]
  if int(getattr(args, "lr_size", 0) or 0) > 0:
    cmd.extend(["--lr_size", str(int(args.lr_size))])
  if args.satburst_data_root:
    cmd.extend(["--satburst_data_root", str(args.satburst_data_root)])
  if args.lr_shift is not None:
    cmd.extend(["--lr_shift", str(float(args.lr_shift))])
  if args.aug:
    cmd.extend(["--aug", str(args.aug)])
  if args.skip_artifacts:
    cmd.append("--skip_artifacts")
  if args.skip_eval:
    cmd.append("--skip_eval")
  if args.no_multiband_diagnostics:
    cmd.append("--no_multiband_diagnostics")
  if args.optimize_extra:
    cmd.extend(args.optimize_extra)
  return cmd, run_args, run_name


def build_plan(args: Namespace) -> list[dict]:
  sample_ids = resolve_sample_ids(args)
  hr_side = int(args.hr_side) if int(args.hr_side or 0) > 0 else _hr_side_for_dataset(args.dataset)
  axes = parse_sweep_axes(args.sweep_axes)
  fixed_max = int(getattr(args, "fixed_hash_max", 0) or 0)

  hash_max_values = (
    _resolve_hash_max_list(args, hr_side)
    if "max" in axes
    else [fixed_max]
  )
  representative_max = max(hash_max_values) if hash_max_values else (fixed_max or hr_side or 256)
  hash_base_values = _resolve_hash_base_list(
    args, sweeping=("base" in axes), representative_max=representative_max
  )
  n_levels_values = _resolve_hash_n_levels_list(args, sweeping=("levels" in axes))
  interp_values = _resolve_hash_interpolation_list(args, sweeping=("interp" in axes))
  learning_rate_values = _resolve_learning_rate_list(args, sweeping=("lr" in axes))
  level_sigma_values = _resolve_hash_level_sigma_list(args, sweeping=("sigma" in axes), hr_side=hr_side)

  plan: list[dict] = []
  for sample_id in sample_ids:
    for hm in hash_max_values:
      for hb in hash_base_values:
        for nl in n_levels_values:
          for interp in interp_values:
            for lr in learning_rate_values:
              for sigma in level_sigma_values:
                cmd, run_args, run_name = build_optimize_command(
                  args,
                  sample_id=sample_id,
                  hash_max=hm,
                  hash_base=hb,
                  hash_n_levels=nl,
                  hash_interpolation=interp,
                  learning_rate=lr,
                  hash_level_sigma=sigma,
                )
                effective_max = hm if hm > 0 else (int(args.lr_size) if int(args.lr_size or 0) > 0 else 48)
                effective_base = hb if hb > 0 else max(8, effective_max // 4)
                plan.append(
                  {
                    "sample_id": str(sample_id),
                    "hash_max_requested": int(hm),
                    "hash_base_requested": int(hb),
                    "hash_n_levels": int(nl),
                    "hash_interpolation": str(interp),
                    "learning_rate": float(lr),
                    "hash_level_sigma": float(sigma),
                    "effective_hash_max": int(effective_max),
                    "effective_hash_base": int(effective_base),
                    "sweep_axes": sorted(axes),
                    "run_name": run_name,
                    "run_args": run_args,
                    "command": cmd,
                  }
                )
  return plan


def _load_metrics(path: Path) -> dict:
  if not path.is_file():
    return {}
  with path.open("r", encoding="utf-8") as f:
    return json.load(f)


def _history_extrema(metrics: dict) -> dict:
  hist = (metrics.get("training") or {}).get("history") or {}
  psnr = hist.get("psnr") or []
  ssim = hist.get("model_ssim") or []
  lpips = hist.get("model_lpips") or []
  out: dict = {}
  if psnr:
    out["peak_train_psnr"] = max(psnr)
    out["final_train_psnr"] = psnr[-1]
  if ssim:
    out["peak_train_ssim"] = max(ssim)
    out["final_train_ssim"] = ssim[-1]
    out["min_train_ssim"] = min(ssim)
  if lpips:
    out["min_train_lpips"] = min(lpips)
    out["final_train_lpips"] = lpips[-1]
    out["max_train_lpips"] = max(lpips)
  return out


def _metric_block(metrics: dict, key: str) -> dict:
  block = metrics.get(key)
  return block if isinstance(block, dict) else {}


def build_arg_parser() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--mode", choices=("quick", "ablation"), default="quick")
  p.add_argument("--run-prefix", default="hash_ablate")
  p.add_argument("--output-dir", type=Path, default=None)
  p.add_argument("--dry-run", action="store_true")

  samples = p.add_argument_group("samples")
  samples.add_argument("--dataset", default="worldstrat_test")
  samples.add_argument("--sample-id", dest="sample_id", default=None)
  samples.add_argument("--sample-ids", dest="sample_ids", nargs="*", default=None)
  samples.add_argument(
    "--worldstrat-set",
    choices=("sweet", "bitter", "test"),
    default=None,
    help="Run all samples listed under a WorldStrat split (uses --dataset worldstrat_test).",
  )
  samples.add_argument("--max-samples", type=int, default=0, help="Cap samples when using --worldstrat-set.")

  grid = p.add_argument_group("hash grid sweep")
  grid.add_argument(
    "--sweep-axes",
    default="max",
    help="Comma-separated axes to factorial: max, base, levels, interp, lr, sigma. "
    "Axes not listed use --fixed-* / *-default values.",
  )
  grid.add_argument("--fixed-hash-max", dest="fixed_hash_max", type=int, default=0)
  grid.add_argument("--fixed-hash-base", dest="fixed_hash_base", type=int, default=0)
  grid.add_argument("--hash-max", dest="hash_max", nargs="*", type=int)
  grid.add_argument("--hash-base", dest="hash_base", nargs="*", type=int, help="0 = auto (max//4).")
  grid.add_argument("--hash-n-levels", dest="hash_n_levels", nargs="*", type=int)
  grid.add_argument("--hash-n-levels-default", dest="hash_n_levels_default", type=int, default=16)
  grid.add_argument("--hash-interpolation", dest="hash_interpolation", nargs="*", choices=["smoothstep", "linear"])
  grid.add_argument("--hash-interpolation-default", dest="hash_interpolation_default", default="smoothstep")
  grid.add_argument(
    "--hash-level-sigma",
    dest="hash_level_sigma",
    nargs="*",
    type=float,
    help="Zip-NeRF level downweighting sigmas to sweep (0 = off). "
    "Default sweep values derive from the LR footprint: 1/(sqrt(12)*W_lr).",
  )
  grid.add_argument("--hash-level-sigma-default", dest="hash_level_sigma_default", type=float, default=0.0)
  grid.add_argument("--hash-n-features-per-level", dest="hash_n_features_per_level", type=int, default=2)
  grid.add_argument("--hash-log2-hashmap-size", dest="hash_log2_hashmap_size", type=int, default=19)
  grid.add_argument("--hr-side", dest="hr_side", type=int, default=0, help="HR side for preset grids (WorldStrat=256).")

  train = p.add_argument_group("training")
  train.add_argument("--df", type=int, default=4)
  train.add_argument("--scale-factor", dest="scale_factor", type=float, default=4.0)
  train.add_argument("--lr-size", dest="lr_size", type=int, default=0)
  train.add_argument("--lr-shift", dest="lr_shift", type=float, default=1.0)
  train.add_argument("--aug", default="none")
  train.add_argument("--num-samples", dest="num_samples", type=int, default=16)
  train.add_argument("--iters", type=int, default=2000)
  train.add_argument("--learning-rate", dest="learning_rate", nargs="*", type=float)
  train.add_argument("--learning-rate-default", dest="learning_rate_default", type=float, default=2e-3)
  train.add_argument("--device", default="1")
  train.add_argument("--seed", type=int, default=6)
  train.add_argument("--model", default="mlp_tcnn", choices=["mlp", "mlp_tcnn"])
  train.add_argument("--input-projection", dest="input_projection", default="hashgrid_tcnn")
  train.add_argument("--supervision-channels", dest="supervision_channels", type=int, default=3)
  train.add_argument("--lr-degradation", dest="lr_degradation", default="area")
  train.add_argument("--eval-every", dest="eval_every", type=int, default=100)
  train.add_argument("--satburst-data-root", dest="satburst_data_root", default=None)
  train.add_argument("--skip-artifacts", dest="skip_artifacts", action="store_true")
  train.add_argument("--skip-eval", dest="skip_eval", action="store_true")
  train.add_argument("--no-multiband-diagnostics", dest="no_multiband_diagnostics", action="store_true", default=True)
  train.add_argument("--multiband-diagnostics", dest="no_multiband_diagnostics", action="store_false")
  p.add_argument("optimize_extra", nargs=argparse.REMAINDER)
  return p


def main() -> None:
  if not OPTIMIZE_PY.is_file():
    raise SystemExit(f"Missing {OPTIMIZE_PY}")

  args = build_arg_parser().parse_args()
  if args.optimize_extra and args.optimize_extra[0] == "--":
    args.optimize_extra = args.optimize_extra[1:]

  if args.worldstrat_set:
    args.dataset = "worldstrat_test"

  plan = build_plan(args)
  if not plan:
    raise SystemExit("Empty ablation plan.")

  ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
  if args.output_dir is not None:
    out_dir = Path(args.output_dir)
  elif len({r["sample_id"] for r in plan}) == 1:
    sid = plan[0]["sample_id"]
    out_dir = (
      Path("single_samples")
      / resolve_dataset_for_optimize(args)
      / str(sid)
      / f"benchmark_hashgrid_{args.mode}_{ts}"
    )
  else:
    out_dir = Path("benchmark_results") / f"hashgrid_{args.mode}_{ts}"

  out_dir.mkdir(parents=True, exist_ok=True)

  print(f"Mode: {args.mode}")
  print(f"Sweep axes: {sorted(parse_sweep_axes(args.sweep_axes))}")
  print(f"Dataset: {args.dataset} (optimize loader: {resolve_dataset_for_optimize(args)})")
  print(f"Summary dir: {out_dir}")
  print(f"Total runs: {len(plan)}")
  for row in plan:
    print(
      f"  {row['sample_id']}: hmax={row['hash_max_requested']:>4} "
      f"hbase={row['hash_base_requested']:>4} L={row['hash_n_levels']} "
      f"{row['hash_interpolation']} lr={row['learning_rate']:.0e} "
      f"sigma={row['hash_level_sigma']:g} → {row['run_name']}"
    )

  plan_path = out_dir / "plan.json"
  plan_path.write_text(
    json.dumps(
      {"mode": args.mode, "runs": [{k: v for k, v in r.items() if k not in ("command", "run_args")} for r in plan]},
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
    psnr_b = _metric_block(metrics, "psnr")
    ssim_b = _metric_block(metrics, "ssim")
    lpips_b = _metric_block(metrics, "lpips")
    spot = metrics.get("fixed_spot") if isinstance(metrics.get("fixed_spot"), dict) else {}
    hist_x = _history_extrema(metrics)

    row_result = {
      **{k: row[k] for k in row if k not in ("command", "run_args")},
      "metrics_path": str(metrics_path),
      "exit_code": int(proc.returncode),
      "wall_seconds": wall,
      "model_psnr": metrics.get("model_psnr"),
      "bilinear_psnr": metrics.get("bilinear_psnr"),
      "psnr_improvement": psnr_b.get("improvement"),
      "model_ssim": ssim_b.get("model"),
      "bilinear_ssim": ssim_b.get("bilinear"),
      "ssim_improvement": ssim_b.get("improvement"),
      "model_lpips": lpips_b.get("model"),
      "bilinear_lpips": lpips_b.get("bilinear"),
      "lpips_improvement": lpips_b.get("improvement"),
      "final_test_psnr": metrics.get("final_test_psnr"),
      "training_time_seconds": metrics.get("training_time_seconds"),
      "completed_iters": metrics.get("completed_iters"),
      "spot_model_psnr": spot.get("model_psnr"),
      "spot_model_ssim": spot.get("model_ssim"),
      "spot_model_lpips": spot.get("model_lpips"),
      "spot_psnr_improvement": spot.get("psnr_improvement"),
      "spot_lpips_improvement": spot.get("lpips_improvement"),
      **hist_x,
    }
    results.append(row_result)

    psnr = row_result.get("model_psnr")
    ssim_v = row_result.get("model_ssim")
    lpips_v = row_result.get("model_lpips")
    psnr_s = f"{psnr:.2f}" if isinstance(psnr, (int, float)) else "n/a"
    ssim_s = f"{ssim_v:.4f}" if isinstance(ssim_v, (int, float)) else "n/a"
    lpips_s = f"{lpips_v:.4f}" if isinstance(lpips_v, (int, float)) else "n/a"
    print(
      f"  exit={proc.returncode} PSNR={psnr_s} SSIM={ssim_s} LPIPS={lpips_s} ({wall:.1f}s)",
      flush=True,
    )

  summary_json = out_dir / "summary.json"
  summary_json.write_text(json.dumps({"mode": args.mode, "results": results}, indent=2), encoding="utf-8")

  fieldnames = [
    "sample_id",
    "hash_max_requested",
    "hash_base_requested",
    "hash_n_levels",
    "hash_interpolation",
    "learning_rate",
    "effective_hash_max",
    "effective_hash_base",
    "run_name",
    "model_psnr",
    "bilinear_psnr",
    "psnr_improvement",
    "model_ssim",
    "bilinear_ssim",
    "ssim_improvement",
    "model_lpips",
    "bilinear_lpips",
    "lpips_improvement",
    "final_test_psnr",
    "peak_train_psnr",
    "min_train_ssim",
    "max_train_lpips",
    "spot_model_psnr",
    "spot_model_ssim",
    "spot_model_lpips",
    "spot_lpips_improvement",
    "training_time_seconds",
    "wall_seconds",
    "exit_code",
    "metrics_path",
  ]
  csv_path = out_dir / "summary.csv"
  with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in results:
      writer.writerow({k: row.get(k) for k in fieldnames})

  print(f"\nWrote {summary_json}")
  print(f"Wrote {csv_path}")

  # Best per sample by LPIPS improvement (higher = better vs bilinear)
  by_sample: dict[str, list[dict]] = {}
  for row in results:
    by_sample.setdefault(str(row["sample_id"]), []).append(row)
  print("\nBest hash_max per sample (by LPIPS improvement vs bilinear):")
  for sid, rows in sorted(by_sample.items()):
    scored = [r for r in rows if isinstance(r.get("lpips_improvement"), (int, float))]
    if not scored:
      continue
    best = max(scored, key=lambda r: float(r["lpips_improvement"]))
    print(
      f"  {sid}: hmax={best['hash_max_requested']} "
      f"LPIPS Δ={best['lpips_improvement']:+.4f} "
      f"SSIM Δ={best.get('ssim_improvement', float('nan')):+.4f} "
      f"PSNR Δ={best.get('psnr_improvement', float('nan')):+.2f} dB"
    )

  failed = [r for r in results if r["exit_code"] != 0]
  if failed:
    raise SystemExit(f"{len(failed)} run(s) failed; see {summary_json}")


if __name__ == "__main__":
  main()
