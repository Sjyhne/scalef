#!/usr/bin/env python3
"""Asker: LR-only holdout stop + sparse LR-tile sampling (audit LPIPS at the end)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

VARIANTS = {
    "full": [],
    "tile64_k4": ["--lr_tile", "64", "--lr_tiles_per_step", "4"],
    "tile128_k2": ["--lr_tile", "128", "--lr_tiles_per_step", "2"],
    "tile128_cover": ["--lr_tile", "128", "--lr_tiles_per_step", "0"],
    # Coverage vs full 512²: k2 sees 1/8, k4 sees 1/16. Scale iters + min_iters.
    "tile128_k2_x8": [
        "--lr_tile",
        "128",
        "--lr_tiles_per_step",
        "2",
        "--iters",
        "24000",
        "--early_stop_min_iters",
        "8000",
    ],
    "tile64_k4_x16": [
        "--lr_tile",
        "64",
        "--lr_tiles_per_step",
        "4",
        "--iters",
        "48000",
        "--early_stop_min_iters",
        "16000",
    ],
    "tile128_k2_x8_nohold": [
        "--lr_tile",
        "128",
        "--lr_tiles_per_step",
        "2",
        "--iters",
        "24000",
        "--spatial_holdout",
        "0",
        "--early_stop_patience",
        "0",
        "--skip_eval",
    ],
    "tile64_k4_x16_nohold": [
        "--lr_tile",
        "64",
        "--lr_tiles_per_step",
        "4",
        "--iters",
        "48000",
        "--spatial_holdout",
        "0",
        "--early_stop_patience",
        "0",
        "--skip_eval",
    ],
}


def _metrics_path(city: str, variant: str) -> Path:
    return ROOT / "single_samples" / city / "sample" / f"holdout_{variant}_charb_3k" / "metrics.json"


def _base_cmd(city: str, variant: str, device: int, iters: int) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "optimize.py"),
        "--dataset",
        city,
        "--s2-dir",
        str(ROOT / "data" / "s2_revisits" / f"{city}_lr512"),
        "--run_name",
        f"holdout_{variant}_charb_3k",
        "--recon_loss",
        "charbonnier",
        "--charbonnier_eps",
        "0.01",
        "--early_stop_metric",
        "holdout_mse",
        "--iters",
        str(iters),
        "--device",
        str(device),
        "--spatial_alignment_path",
        str(ROOT / "eval" / "spatial_alignment.json"),
        "--no_qgis_export",
        *VARIANTS[variant],
    ]


def _run_variant(city: str, variant: str, device: int, iters: int, skip_existing: bool) -> dict:
    mpath = _metrics_path(city, variant)
    if skip_existing and mpath.is_file():
        return _summarize(city, variant, mpath, skipped=True)
    cmd = _base_cmd(city, variant, device, iters)
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
        "audit_lpips": m["lpips"]["model"],
        "psnr": m["psnr"]["model"],
        "completed_iters": m["completed_iters"],
        "best_iter": es.get("best_val_iter"),
        "stop_metric": es.get("metric"),
        "training_time_s": m.get("training_time_seconds"),
        "peak_memory_gb": m.get("peak_memory_gb"),
        "lr_tile": m.get("lr_tile"),
        "lr_tiles_per_step": m.get("lr_tiles_per_step"),
        "metrics_path": str(mpath.relative_to(ROOT)),
    }


def _build_summary(rows: list[dict]) -> dict:
    base = next((r for r in rows if r["variant"] == "full"), None)
    deltas = []
    if base:
        for r in rows:
            if r["variant"] == "full":
                continue
            row = {
                "variant": r["variant"],
                "audit_lpips": r["audit_lpips"],
                "lpips_delta_vs_full": r["audit_lpips"] - base["audit_lpips"],
                "best_iter": r["best_iter"],
                "peak_memory_gb": r.get("peak_memory_gb"),
            }
            deltas.append(row)
    return {
        "full_audit_lpips": base["audit_lpips"] if base else None,
        "full_best_iter": base["best_iter"] if base else None,
        "full_peak_memory_gb": base.get("peak_memory_gb") if base else None,
        "lpips_stop_reference": 0.3667,
        "per_variant": deltas,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="asker")
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--gpus", type=int, default=4)
    ap.add_argument("--gpu-offset", type=int, default=2)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument(
        "--variants",
        nargs="+",
        default=["full", "tile64_k4", "tile128_k2", "tile128_cover"],
        choices=list(VARIANTS),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "single_samples" / "sweep_results" / "holdout_sparse_asker.json",
    )
    args = ap.parse_args()

    variants = list(args.variants)
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, min(args.gpus, len(variants)))) as pool:
        futures = {
            pool.submit(
                _run_variant,
                args.city,
                variant,
                args.gpu_offset + i,
                args.iters,
                args.skip_existing,
            ): variant
            for i, variant in enumerate(variants)
        }
        for fut in as_completed(futures):
            variant = futures[fut]
            rows.append(fut.result())
            print(f"DONE {variant}", flush=True)

    rows.sort(key=lambda r: r["variant"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "iters_budget": args.iters,
        "early_stop_metric": "holdout_mse",
        "rows": rows,
        "summary": _build_summary(rows),
    }
    args.out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["summary"], indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
