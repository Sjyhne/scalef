#!/usr/bin/env python3
"""Benchmark PSF training schedules vs baseline (default: asker, 3k iters)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

VARIANTS = {
    "baseline": [],
    "psf_curr_step": ["--psf_curriculum", "step"],
    "psf_sigma_lin": ["--psf_sigma_schedule", "linear"],
}


def _metrics_path(city: str, variant: str) -> Path:
    return ROOT / "single_samples" / city / "sample" / f"sched_{variant}_mae_3k" / "metrics.json"


def _run_variant(city: str, variant: str, device: int, iters: int, skip_existing: bool) -> dict:
    mpath = _metrics_path(city, variant)
    if skip_existing and mpath.is_file():
        return _summarize(city, variant, mpath, skipped=True)
    cmd = [
        sys.executable,
        str(ROOT / "optimize.py"),
        "--dataset",
        city,
        "--s2-dir",
        str(ROOT / "data" / "s2_revisits" / f"{city}_lr512"),
        "--run_name",
        f"sched_{variant}_mae_3k",
        "--recon_loss",
        "mae",
        "--iters",
        str(iters),
        "--device",
        str(device),
        "--spatial_alignment_path",
        str(ROOT / "eval" / "spatial_alignment.json"),
        "--no_qgis_export",
        *VARIANTS[variant],
    ]
    print(f"[gpu{device}] {city} {variant} ...", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    return _summarize(city, variant, mpath, skipped=False)


def _summarize(city: str, variant: str, mpath: Path, *, skipped: bool) -> dict:
    m = json.loads(mpath.read_text())
    es = m.get("early_stop") or {}
    return {
        "city": city,
        "variant": variant,
        "skipped": skipped,
        "lpips": m["lpips"]["model"],
        "psnr": m["psnr"]["model"],
        "completed_iters": m["completed_iters"],
        "best_iter": es.get("best_val_iter"),
        "psf_curriculum": m.get("psf_curriculum"),
        "psf_sigma_schedule": m.get("psf_sigma_schedule"),
        "metrics_path": str(mpath.relative_to(ROOT)),
    }


def _build_summary(rows: list[dict]) -> dict:
    base = next((r for r in rows if r["variant"] == "baseline"), None)
    deltas = []
    if base:
        for r in rows:
            if r["variant"] == "baseline":
                continue
            deltas.append(
                {
                    "variant": r["variant"],
                    "lpips": r["lpips"],
                    "lpips_delta_vs_baseline": r["lpips"] - base["lpips"],
                    "wins_lpips": r["lpips"] < base["lpips"],
                }
            )
    deltas.sort(key=lambda d: d["lpips_delta_vs_baseline"])
    return {"baseline_lpips": base["lpips"] if base else None, "per_variant": deltas}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="asker")
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS.keys()), choices=list(VARIANTS.keys()))
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "single_samples" / "sweep_results" / "training_schedules_bench.json",
    )
    args = ap.parse_args()

    rows = [_run_variant(args.city, v, args.device, args.iters, args.skip_existing) for v in args.variants]
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "city": args.city,
        "iters_budget": args.iters,
        "rows": rows,
        "summary": _build_summary(rows),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
