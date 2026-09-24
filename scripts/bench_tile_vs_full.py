#!/usr/bin/env python3
"""Asker benchmark: full-field vs fused LR-tile sampling under matched holdout stop.

Success bands vs full (audit LPIPS):
  match:      Δ ≤ +0.005 and (faster or less VRAM)
  speed_mode: Δ ≤ +0.015 and (≥2× VRAM save or ≥1.2× faster)
"""

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
    "k2": ["--lr_tile", "128", "--lr_tiles_per_step", "2"],
    "k4": ["--lr_tile", "128", "--lr_tiles_per_step", "4"],
    "k8": ["--lr_tile", "128", "--lr_tiles_per_step", "8"],
}

COVERAGE = {
    "full": 1.0,
    "k2": 2 * (128**2) / (512**2),
    "k4": 4 * (128**2) / (512**2),
    "k8": 8 * (128**2) / (512**2),
}


def _metrics_path(city: str, variant: str, iters: int) -> Path:
    return (
        ROOT
        / "single_samples"
        / city
        / "sample"
        / f"bench_{variant}_holdout_{iters // 1000}k"
        / "metrics.json"
    )


def _base_cmd(city: str, variant: str, device: int, iters: int) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "optimize.py"),
        "--dataset",
        city,
        "--s2-dir",
        str(ROOT / "data" / "s2_revisits" / f"{city}_lr512"),
        "--run_name",
        f"bench_{variant}_holdout_{iters // 1000}k",
        "--recon_loss",
        "charbonnier",
        "--charbonnier_eps",
        "0.01",
        "--early_stop_metric",
        "holdout_mse",
        "--early_stop_patience",
        "3",
        "--early_stop_min_iters",
        "1000",
        "--iters",
        str(iters),
        "--device",
        str(device),
        "--spatial_holdout",
        "0.1",
        "--spatial_alignment_path",
        str(ROOT / "eval" / "spatial_alignment.json"),
        "--no_qgis_export",
        *VARIANTS[variant],
    ]


def _summarize(city: str, variant: str, mpath: Path, *, skipped: bool) -> dict:
    m = json.loads(mpath.read_text())
    es = m.get("early_stop") or {}
    return {
        "city": city,
        "variant": variant,
        "skipped": skipped,
        "coverage_per_step": COVERAGE[variant],
        "audit_lpips": m["lpips"]["model"],
        "psnr": m["psnr"]["model"],
        "ssim": m["ssim"]["model"],
        "completed_iters": m["completed_iters"],
        "best_iter": es.get("best_val_iter"),
        "stopped_iter": es.get("stopped_iter"),
        "training_time_s": m.get("training_time_seconds"),
        "peak_memory_gb": m.get("peak_memory_gb"),
        "lr_tile": m.get("lr_tile"),
        "lr_tiles_per_step": m.get("lr_tiles_per_step"),
        "spatial_holdout": m.get("spatial_holdout"),
        "metrics_path": str(mpath.relative_to(ROOT)),
    }


def _run_variant(
    city: str, variant: str, device: int, iters: int, skip_existing: bool
) -> dict:
    mpath = _metrics_path(city, variant, iters)
    if skip_existing and mpath.is_file():
        return _summarize(city, variant, mpath, skipped=True)
    print(f"[gpu{device}] {city} {variant} ...", flush=True)
    subprocess.run(_base_cmd(city, variant, device, iters), cwd=ROOT, check=True)
    return _summarize(city, variant, mpath, skipped=False)


def _band(delta: float, time_ratio: float | None, vram_ratio: float | None) -> str:
    faster = time_ratio is not None and time_ratio <= 1.0 / 1.2
    leaner = vram_ratio is not None and vram_ratio <= 0.5
    if delta <= 0.005 and (faster or leaner or (time_ratio is not None and time_ratio < 1.0)):
        return "match"
    if delta <= 0.015 and (faster or leaner):
        return "speed_mode"
    if delta <= 0.015:
        return "near"
    return "fail"


def _build_summary(rows: list[dict]) -> dict:
    base = next((r for r in rows if r["variant"] == "full"), None)
    per = []
    for r in rows:
        if r["variant"] == "full":
            continue
        delta = r["audit_lpips"] - base["audit_lpips"] if base else None
        time_ratio = (
            r["training_time_s"] / base["training_time_s"]
            if base and base.get("training_time_s") and r.get("training_time_s")
            else None
        )
        vram_ratio = (
            r["peak_memory_gb"] / base["peak_memory_gb"]
            if base and base.get("peak_memory_gb") and r.get("peak_memory_gb")
            else None
        )
        per.append(
            {
                "variant": r["variant"],
                "coverage_per_step": r["coverage_per_step"],
                "audit_lpips": r["audit_lpips"],
                "lpips_delta_vs_full": delta,
                "best_iter": r["best_iter"],
                "stopped_iter": r["stopped_iter"],
                "training_time_s": r["training_time_s"],
                "time_ratio_vs_full": time_ratio,
                "peak_memory_gb": r["peak_memory_gb"],
                "vram_ratio_vs_full": vram_ratio,
                "band": _band(delta, time_ratio, vram_ratio) if delta is not None else None,
            }
        )
    return {
        "full_audit_lpips": base["audit_lpips"] if base else None,
        "full_best_iter": base["best_iter"] if base else None,
        "full_time_s": base.get("training_time_s") if base else None,
        "full_peak_memory_gb": base.get("peak_memory_gb") if base else None,
        "bands": {
            "match": "ΔLPIPS≤+0.005 and faster or ≤0.5× VRAM",
            "speed_mode": "ΔLPIPS≤+0.015 and (≥1.2× faster or ≤0.5× VRAM)",
            "near": "ΔLPIPS≤+0.015 but not clearly cheaper",
            "fail": "ΔLPIPS>+0.015",
        },
        "per_variant": per,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--city", default="asker")
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--gpus", type=int, default=4)
    ap.add_argument("--gpu-offset", type=int, default=0)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument(
        "--variants",
        nargs="+",
        default=list(VARIANTS),
        choices=list(VARIANTS),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "single_samples" / "sweep_results" / "bench_tile_vs_full.json",
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

    rows.sort(key=lambda r: list(VARIANTS).index(r["variant"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "city": args.city,
        "iters_budget": args.iters,
        "early_stop_metric": "holdout_mse",
        "recon_loss": "charbonnier",
        "charbonnier_eps": 0.01,
        "spatial_holdout": 0.1,
        "fused_tiles": True,
        "rows": rows,
        "summary": _build_summary(rows),
    }
    args.out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["summary"], indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
