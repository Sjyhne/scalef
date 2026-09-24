#!/usr/bin/env python3
"""Train the best hashgrid config on Asker (or other city) at several LR sizes.

Example
-------
python scripts/sweep_lr_size.py --city asker --sizes 256 512 1024 --device 0 --iters 2000
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Bergen Stage-B winner (verified across cities)
WINNER = {
    "hash_interpolation": "linear",
    "hash_log2_hashmap_size": 21,
    "hash_n_levels": 16,
    "learning_rate": 0.001,
    "recon_loss": "mae",
}


def _run_name(side: int) -> str:
    return f"size_lr{side}"


def _metrics_path(city: str, side: int) -> Path:
    return ROOT / "single_samples" / city / "sample" / _run_name(side) / "metrics.json"


def _build_cmd(city: str, side: int, iters: int, device: str) -> list[str]:
    s2_dir = ROOT / "data" / "s2_revisits" / f"{city}_lr{side}"
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
        "hashgrid_tcnn",
        "--model",
        "mlp_tcnn",
        "--sample_id",
        "sample",
        "--run_name",
        _run_name(side),
        "--iters",
        str(iters),
        "--eval_every",
        "0",
        "--eval-spot-hr-px",
        "32",
        "--device",
        str(device),
        "--hash_interpolation",
        WINNER["hash_interpolation"],
        "--hash_log2_hashmap_size",
        str(WINNER["hash_log2_hashmap_size"]),
        "--hash_n_levels",
        str(WINNER["hash_n_levels"]),
        "--learning_rate",
        f"{WINNER['learning_rate']:.2e}",
        "--recon_loss",
        WINNER["recon_loss"],
        "--num_samples",
        "16",
    ]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--city", default="asker")
    p.add_argument("--sizes", type=int, nargs="+", default=[256, 512, 1024])
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--device", default="0")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-complete", action="store_true", default=True)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    # Ensure size variants exist.
    subprocess.check_call(
        [
            sys.executable,
            str(ROOT / "scripts" / "make_lr_size_variants.py"),
            "--city",
            args.city,
            "--sizes",
            *[str(s) for s in args.sizes],
        ],
        cwd=ROOT,
    )

    rows = []
    for side in args.sizes:
        mpath = _metrics_path(args.city, side)
        if args.skip_complete and not args.force and mpath.exists():
            print(f"skip complete lr{side}: {mpath}")
            rows.append({"lr_size": side, "metrics": json.loads(mpath.read_text()), "skipped": True})
            continue
        cmd = _build_cmd(args.city, side, args.iters, args.device)
        print(f"\n>>> {args.city} lr{side}", flush=True)
        print(" ".join(cmd), flush=True)
        if args.dry_run:
            continue
        proc = subprocess.run(cmd, cwd=ROOT)
        rec = {"lr_size": side, "exit": proc.returncode, "skipped": False}
        if mpath.exists():
            rec["metrics"] = json.loads(mpath.read_text())
        rows.append(rec)
        print(f"exit={proc.returncode} lr{side}", flush=True)

    out_dir = ROOT / "single_samples" / "sweep_results" / f"{args.city}_lr_size"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for r in rows:
        m = r.get("metrics") or {}
        spot = m.get("fixed_spot") or {}
        summary.append(
            {
                "lr_size": r["lr_size"],
                "psnr": (m.get("psnr") or {}).get("model"),
                "ssim": (m.get("ssim") or {}).get("model"),
                "lpips": (m.get("lpips") or {}).get("model"),
                "spot_psnr": spot.get("model_psnr"),
                "exit": r.get("exit"),
                "skipped": r.get("skipped", False),
            }
        )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (out_dir / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()) if summary else [])
        if summary:
            w.writeheader()
            w.writerows(summary)
    print(f"\nSummary → {out_dir}")
    for s in summary:
        print(
            f"  lr{s['lr_size']}: PSNR={s['psnr']}  SSIM={s['ssim']}  "
            f"LPIPS={s['lpips']}  spot={s['spot_psnr']}"
        )


if __name__ == "__main__":
    main()
