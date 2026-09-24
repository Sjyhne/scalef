#!/usr/bin/env python3
"""Coarse-to-fine hyperparameter sweep for the hashgrid SR model.

Stage A  -- short runs over a wide grid, selects top-K configs.
Stage B  -- longer runs using only the top-K configs from Stage A.

Results are written to:
    single_samples/sweep_results/<timestamp>_<aoi>/
        sweep_summary.csv   -- one row per completed run, sorted by spot_psnr
        best_config.json    -- best config in full detail

Usage
-----
# Quick smoke test (cpu, 3 runs)
python scripts/sweep_hashgrid.py --device cpu --max-runs 3 --iters-coarse 200

# Full sweep on GPU 0
python scripts/sweep_hashgrid.py --device 0 --topk 8 --iters-coarse 1000 --iters-fine 3000

# Resume an interrupted sweep (re-uses existing metrics.json)
python scripts/sweep_hashgrid.py --device 0 --resume
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------

COARSE_GRID = {
    "hash_log2_hashmap_size": [19, 20, 21],
    "hash_interpolation":     ["smoothstep", "linear"],
    "hash_n_levels":          [12, 16],
    "learning_rate":          [1e-3, 2e-3],
    "recon_loss":             ["mse", "mae"],
}

# Keys varied only in Stage B (added on top of the best Stage A config)
FINE_EXTRA = {
    "learning_rate":    [5e-4, 1e-3, 2e-3, 3e-3],
    "use_laplace_nll":  [False, True],
}

# These are fixed for every run
FIXED = {
    "dataset":          "s2",
    "s2-dir":           "data/s2_revisits/bergen",
    "df":               4,
    "scale_factor":     4,
    "input_projection": "hashgrid",
    "sample_id":        "sample",
    "eval_every":       0,       # disable periodic eval for speed
    "eval-spot-hr-px":  32,
    "skip_artifacts":   True,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _config_to_run_name(cfg: dict) -> str:
    """Stable, human-readable run name from a config dict."""
    parts = []
    for k, v in sorted(cfg.items()):
        short_k = k.replace("hash_", "h_").replace("learning_rate", "lr").replace("recon_loss", "loss")
        if isinstance(v, float):
            parts.append(f"{short_k}{v:.0e}".replace("+", "").replace("e0", "e"))
        elif isinstance(v, bool):
            if v:
                parts.append(short_k)
        else:
            parts.append(f"{short_k}{v}")
    return "_".join(parts)[:120]


def _metrics_path(run_name: str) -> Path:
    return ROOT / "single_samples" / "s2" / "sample" / run_name / "metrics.json"


def _read_metric(run_name: str) -> float | None:
    p = _metrics_path(run_name)
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
        spot = d.get("fixed_spot") or {}
        v = spot.get("model_psnr")
        if v is not None:
            return float(v)
        return float(d.get("final_test_psnr") or d.get("model_psnr") or 0.0)
    except Exception:
        return None


def _build_cmd(cfg: dict, run_name: str, iters: int, device: str) -> list[str]:
    cmd = [sys.executable, str(ROOT / "optimize.py")]
    for k, v in FIXED.items():
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
        else:
            cmd += [f"--{k}", str(v)]
    cmd += ["--run_name", run_name, "--iters", str(iters), "--device", str(device)]
    for k, v in cfg.items():
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
        elif k == "learning_rate":
            cmd += ["--learning_rate", f"{v:.2e}"]
        else:
            cmd += [f"--{k}", str(v)]
    return cmd


def _run(cmd: list[str], run_name: str, results_dir: Path, dry_run: bool = False) -> float | None:
    print(f"\n>>> {run_name}")
    print("    " + " ".join(cmd))
    if dry_run:
        return None
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        print(f"    FAILED (exit {r.returncode})")
        return None
    val = _read_metric(run_name)
    print(f"    spot_psnr = {val}")
    return val


def _save_summary(rows: list[dict], results_dir: Path) -> None:
    if not rows:
        return
    results_dir.mkdir(parents=True, exist_ok=True)
    # Rows are heterogeneous across stages/configs; infer a stable superset header.
    # (Fixes CSV crash: "dict contains fields not in fieldnames: 'use_laplace_nll'".)
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(results_dir / "sweep_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    best = max(rows, key=lambda r: float(r.get("spot_psnr") or -999))
    (results_dir / "best_config.json").write_text(json.dumps(best, indent=2))
    print(f"\n=== Best so far: {best['run_name']}  spot_psnr={best['spot_psnr']} ===")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cpu", help="CUDA device index or 'cpu'")
    ap.add_argument("--max-runs", type=int, default=0, help="Cap total runs (0 = no cap, default: all)")
    ap.add_argument("--topk", type=int, default=6, help="Number of Stage-A configs promoted to Stage B (default: 6)")
    ap.add_argument("--iters-coarse", type=int, default=1000, help="Iterations for Stage A (default: 1000)")
    ap.add_argument("--iters-fine", type=int, default=3000, help="Iterations for Stage B (default: 3000)")
    ap.add_argument("--no-stage-b", action="store_true", help="Skip Stage B (only run Stage A)")
    ap.add_argument("--resume", action="store_true", help="Skip runs that already have metrics.json")
    ap.add_argument("--dry-run", action="store_true", help="Print commands only, do not run")
    ap.add_argument("--aoi", default="bergen", help="City subfolder under data/s2_revisits/ (default: bergen)")
    args = ap.parse_args()

    # Allow --aoi to override the fixed s2_dir
    FIXED["s2-dir"] = f"data/s2_revisits/{args.aoi}"
    FIXED["dataset"] = "s2"

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = ROOT / "single_samples" / "sweep_results" / f"{ts}_{args.aoi}"
    results_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Stage A: coarse grid
    # ------------------------------------------------------------------
    print(f"\n{'='*60}\nStage A — coarse grid (iters={args.iters_coarse})\n{'='*60}")
    keys_a = list(COARSE_GRID.keys())
    values_a = list(COARSE_GRID.values())
    configs_a = [dict(zip(keys_a, combo)) for combo in itertools.product(*values_a)]
    print(f"Total Stage-A configs: {len(configs_a)}")

    rows: list[dict] = []
    run_count = 0

    for cfg in configs_a:
        if args.max_runs > 0 and run_count >= args.max_runs:
            break
        run_name = "sweepA_" + _config_to_run_name(cfg)
        existing = _read_metric(run_name) if args.resume else None
        if existing is not None:
            print(f"    skip (resume): {run_name}  spot_psnr={existing}")
            rows.append({**cfg, "stage": "A", "iters": args.iters_coarse, "run_name": run_name, "spot_psnr": existing})
            continue
        cmd = _build_cmd(cfg, run_name, args.iters_coarse, args.device)
        val = _run(cmd, run_name, results_dir, dry_run=args.dry_run)
        rows.append({**cfg, "stage": "A", "iters": args.iters_coarse, "run_name": run_name, "spot_psnr": val})
        run_count += 1
        _save_summary(sorted(rows, key=lambda r: float(r.get("spot_psnr") or -999), reverse=True), results_dir)

    if not args.dry_run:
        _save_summary(sorted(rows, key=lambda r: float(r.get("spot_psnr") or -999), reverse=True), results_dir)

    if args.no_stage_b:
        print("\nStage B skipped (--no-stage-b).")
        return

    # ------------------------------------------------------------------
    # Stage B: fine sweep around top-K Stage-A configs
    # ------------------------------------------------------------------
    print(f"\n{'='*60}\nStage B — refine top-{args.topk} configs (iters={args.iters_fine})\n{'='*60}")
    completed_a = [r for r in rows if r.get("spot_psnr") is not None]
    top_a = sorted(completed_a, key=lambda r: float(r["spot_psnr"]), reverse=True)[:args.topk]

    for base_cfg_row in top_a:
        base_cfg = {k: base_cfg_row[k] for k in COARSE_GRID if k in base_cfg_row}
        keys_b = list(FINE_EXTRA.keys())
        values_b = list(FINE_EXTRA.values())
        for combo in itertools.product(*values_b):
            if args.max_runs > 0 and run_count >= args.max_runs:
                break
            override = dict(zip(keys_b, combo))
            cfg = {**base_cfg, **override}
            run_name = "sweepB_" + _config_to_run_name(cfg)
            existing = _read_metric(run_name) if args.resume else None
            if existing is not None:
                print(f"    skip (resume): {run_name}  spot_psnr={existing}")
                rows.append({**cfg, "stage": "B", "iters": args.iters_fine, "run_name": run_name, "spot_psnr": existing})
                continue
            cmd = _build_cmd(cfg, run_name, args.iters_fine, args.device)
            val = _run(cmd, run_name, results_dir, dry_run=args.dry_run)
            rows.append({**cfg, "stage": "B", "iters": args.iters_fine, "run_name": run_name, "spot_psnr": val})
            run_count += 1
            _save_summary(sorted(rows, key=lambda r: float(r.get("spot_psnr") or -999), reverse=True), results_dir)

    _save_summary(sorted(rows, key=lambda r: float(r.get("spot_psnr") or -999), reverse=True), results_dir)
    print(f"\nSweep complete. Results: {results_dir}")


if __name__ == "__main__":
    main()
