#!/usr/bin/env python3
"""Long-run validation: where would holdout early-stop fire vs 4–5k iters?

Runs ``optimize.py`` to completion (patience=0) while logging holdout val + HR
metrics, then compares:
  - simulated patience stop
  - holdout-val minimum
  - fixed checkpoints (4k / 5k)
  - oracle best LPIPS (needs HR GT — analysis only, not for stopping)

Example
-------
python scripts/validate_early_stop.py --city asker --iters 5000 --device 0
python scripts/validate_early_stop.py --cities asker bergen rana --iters 5000
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.stopping_analysis import analyze_stopping_trajectory

DEFAULT_CITIES = ["asker", "bergen", "rana"]


def _run_name(city: str, iters: int) -> str:
    return f"val_es_{city}_lr512_i{iters}"


def _metrics_path(city: str, iters: int) -> Path:
    return ROOT / "single_samples" / city / "sample" / _run_name(city, iters) / "metrics.json"


def _build_cmd(city: str, iters: int, device: str, eval_every: int) -> list[str]:
    s2_dir = ROOT / "data" / "s2_revisits" / f"{city}_lr512"
    return [
        sys.executable,
        str(ROOT / "optimize.py"),
        "--dataset",
        city,
        "--s2-dir",
        str(s2_dir),
        "--df",
        "4",
        "--scale_factor",
        "4",
        "--input_projection",
        "hashgrid",
        "--sample_id",
        "sample",
        "--run_name",
        _run_name(city, iters),
        "--iters",
        str(iters),
        "--eval_every",
        str(eval_every),
        "--device",
        str(device),
        "--lr_degradation",
        "s2_psf_m",
        "--spatial_holdout",
        "0.1",
        "--holdout_block",
        "8",
        "--early_stop_patience",
        "0",
        "--early_stop_min_iters",
        "1000",
        "--hash_interpolation",
        "linear",
        "--hash_log2_hashmap_size",
        "21",
        "--hash_n_levels",
        "16",
        "--learning_rate",
        "1e-3",
        "--recon_loss",
        "mse",
        "--spatial_alignment_path",
        str(ROOT / "eval" / "spatial_alignment.json"),
        "--eval-spot-hr-px",
        "32",
    ]


def _ensure_lr512(city: str) -> None:
    s2 = ROOT / "data" / "s2_revisits" / f"{city}_lr512"
    if (s2 / "meta.json").is_file():
        return
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "make_lr_size_variants.py"),
            "--city",
            city,
            "--sizes",
            "512",
        ],
        cwd=ROOT,
        check=True,
    )


def _plot(city: str, analysis: dict, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    traj = analysis["trajectory"]
    iters = traj["iterations"]
    fig, ax1 = plt.subplots(figsize=(10, 4.5))
    ax1.plot(iters, traj["model_lpips"], "-o", color="#d62728", ms=3, label="LPIPS vs HR GT")
    ax1.set_ylabel("LPIPS (lower better)", color="#d62728")
    ax1.tick_params(axis="y", labelcolor="#d62728")

    oracle = analysis["oracle_best_lpips"]["iter"]
    ax1.axvline(oracle, color="#d62728", ls="--", lw=1.2, alpha=0.8)

    ax2 = ax1.twinx()
    ax2.plot(iters, traj["val_loss"], "-s", color="#2ca02c", ms=3, label="holdout val MSE")
    ax2.set_ylabel("holdout val MSE", color="#2ca02c")
    ax2.tick_params(axis="y", labelcolor="#2ca02c")

    stop = analysis.get("simulated_stop_iter")
    if stop is not None:
        ax2.axvline(stop, color="#2ca02c", ls="--", lw=1.2, alpha=0.8)
    for t in (4000, 5000):
        if t <= max(iters):
            ax1.axvline(t, color="#888888", ls=":", lw=1.0, alpha=0.7)

    ax1.set_xlabel("iteration")
    ax1.set_title(f"{city}: holdout val vs LPIPS (dashed = oracle LPIPS / sim stop / 4k·5k)")
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _print_summary(city: str, analysis: dict) -> None:
    ck = analysis["checkpoints"]
    reg = analysis["regret_lpips_vs_oracle"]
    print(f"\n=== {city} ===")
    print(
        f"completed {analysis['completed_iters']} iters | "
        f"oracle LPIPS {analysis['oracle_best_lpips']['lpips']:.4f} @ {analysis['oracle_best_lpips']['iter']}"
    )
    print(
        f"holdout val min {analysis['best_val_loss']:.6f} @ {analysis['best_val_iter']} | "
        f"sim stop (patience {analysis['patience']}) @ {analysis['simulated_stop_iter']}"
    )
    for key in ("simulated_early_stop", "holdout_val_minimum", "fixed_4000", "fixed_5000", "final_logged"):
        p = ck.get(key)
        if not p:
            continue
        r = reg.get(key)
        r_s = f" regret LPIPS {r:+.4f}" if r is not None else ""
        print(
            f"  {key:22s} iter {p['actual_iter']:5d}  "
            f"LPIPS {p['lpips']:.4f}  PSNR {p.get('psnr') or 0:.2f} dB{r_s}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--city", default=None)
    ap.add_argument("--cities", nargs="+", default=None)
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--device", default="0")
    ap.add_argument("--patience", type=int, default=5, help="Patience for simulated stop analysis")
    ap.add_argument("--min-iters", type=int, default=1000)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "single_samples" / "sweep_results" / "early_stop_validation",
    )
    args = ap.parse_args()

    if args.city:
        cities = [args.city]
    elif args.cities:
        cities = list(args.cities)
    else:
        cities = DEFAULT_CITIES

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    rows: list[dict] = []

    for city in cities:
        mpath = _metrics_path(city, args.iters)
        if mpath.is_file() and not args.force:
            print(f"skip train {city} (exists {mpath}); analyzing only")
        elif args.dry_run:
            print(" ".join(_build_cmd(city, args.iters, args.device, args.eval_every)))
            continue
        else:
            _ensure_lr512(city)
            cmd = _build_cmd(city, args.iters, args.device, args.eval_every)
            print(f"\n>>> training {city} ({args.iters} iters)\n{' '.join(cmd)}\n", flush=True)
            rc = subprocess.run(cmd, cwd=ROOT).returncode
            if rc != 0:
                print(f"ERROR: optimize.py exited {rc} for {city}", file=sys.stderr)
                continue

        if not mpath.is_file():
            print(f"missing {mpath}", file=sys.stderr)
            continue

        metrics = json.loads(mpath.read_text())
        analysis = analyze_stopping_trajectory(
            metrics,
            patience=args.patience,
            min_iters=args.min_iters,
            fixed_iters=(4000, args.iters),
        )
        _print_summary(city, analysis)
        plot_path = args.out_dir / f"{city}_trajectory.png"
        _plot(city, analysis, plot_path)
        print(f"plot → {plot_path}")

        row = {
            "city": city,
            "iters": args.iters,
            "completed_iters": analysis["completed_iters"],
            "oracle_lpips_iter": analysis["oracle_best_lpips"]["iter"],
            "oracle_lpips": analysis["oracle_best_lpips"]["lpips"],
            "sim_stop_iter": analysis.get("simulated_stop_iter"),
            "val_min_iter": analysis["best_val_iter"],
            "metrics_path": str(mpath),
        }
        for key in ("simulated_early_stop", "fixed_4000", "fixed_5000", "final_logged"):
            p = analysis["checkpoints"].get(key)
            if p:
                row[f"{key}_iter"] = p["actual_iter"]
                row[f"{key}_lpips"] = p["lpips"]
                row[f"{key}_regret"] = analysis["regret_lpips_vs_oracle"].get(key)
        rows.append(row)

        (args.out_dir / f"{city}_analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")

    if rows:
        summary_path = args.out_dir / f"summary_{stamp}.csv"
        fields = sorted({k for r in rows for k in r})
        with summary_path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
