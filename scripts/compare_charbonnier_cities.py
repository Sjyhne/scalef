#!/usr/bin/env python3
"""Compare MAE vs Charbonnier (eps sweep) across focus cities (LPIPS early stop).

Runs one job per GPU via a work queue so CUDA init never races on reclaim.
"""

from __future__ import annotations

import argparse
import json
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FOCUS_CITIES = ["asker", "vennesla", "trondheim", "bergen", "rana", "tromso", "amli"]

# (variant_key, recon_loss, charbonnier_eps or None)
VARIANTS: list[tuple[str, str, float | None]] = [
    ("mae", "mae", None),
    ("charb_1e-4", "charbonnier", 1e-4),
    ("charb_1e-3", "charbonnier", 1e-3),
    ("charb_1e-2", "charbonnier", 1e-2),
]


def _run_name(variant: str, iters: int) -> str:
    return f"stop_lpips_{variant}_{iters // 1000}k"


def _metrics_path(city: str, variant: str, iters: int) -> Path:
    return ROOT / "single_samples" / city / "sample" / _run_name(variant, iters) / "metrics.json"


def _run_one(
    city: str,
    variant: str,
    loss: str,
    eps: float | None,
    device: int,
    iters: int,
) -> dict:
    mpath = _metrics_path(city, variant, iters)
    cmd = [
        sys.executable,
        str(ROOT / "optimize.py"),
        "--dataset",
        city,
        "--s2-dir",
        str(ROOT / "data" / "s2_revisits" / f"{city}_lr512"),
        "--run_name",
        _run_name(variant, iters),
        "--recon_loss",
        loss,
        "--iters",
        str(iters),
        "--device",
        str(device),
        "--spatial_alignment_path",
        str(ROOT / "eval" / "spatial_alignment.json"),
        "--no_qgis_export",
    ]
    if eps is not None:
        cmd += ["--charbonnier_eps", f"{eps:g}"]

    print(f"[gpu{device}] {city} {variant} ...", flush=True)
    # Brief pause so the previous process on this GPU can fully release CUDA context.
    time.sleep(1.0)
    subprocess.run(cmd, cwd=ROOT, check=True)
    return _summarize(city, variant, loss, eps, mpath, skipped=False)


def _summarize(
    city: str,
    variant: str,
    loss: str,
    eps: float | None,
    mpath: Path,
    *,
    skipped: bool,
) -> dict:
    m = json.loads(mpath.read_text())
    es = m.get("early_stop") or {}
    return {
        "city": city,
        "variant": variant,
        "recon_loss": loss,
        "charbonnier_eps": eps,
        "skipped": skipped,
        "lpips": m["lpips"]["model"],
        "psnr": m["psnr"]["model"],
        "ssim": m["ssim"]["model"],
        "completed_iters": m["completed_iters"],
        "best_iter": es.get("best_val_iter"),
        "stopped_iter": es.get("stopped_iter"),
        "training_time_s": m.get("training_time_seconds"),
        "metrics_path": str(mpath.relative_to(ROOT)),
    }


def _build_summary(rows: list[dict]) -> dict:
    by_city: dict[str, dict[str, dict]] = {}
    for r in rows:
        by_city.setdefault(r["city"], {})[r["variant"]] = r

    per_variant: dict[str, dict] = {}
    for variant, _, _ in VARIANTS:
        if variant == "mae":
            continue
        deltas = []
        for city, variants in sorted(by_city.items()):
            if "mae" not in variants or variant not in variants:
                continue
            mae = variants["mae"]
            cur = variants[variant]
            deltas.append(
                {
                    "city": city,
                    "lpips": cur["lpips"],
                    "mae_lpips": mae["lpips"],
                    "lpips_delta_vs_mae": cur["lpips"] - mae["lpips"],
                    "psnr_delta_vs_mae": cur["psnr"] - mae["psnr"],
                    "wins_lpips": cur["lpips"] < mae["lpips"],
                }
            )
        wins = sum(1 for d in deltas if d["wins_lpips"])
        mean_delta = sum(d["lpips_delta_vs_mae"] for d in deltas) / max(len(deltas), 1)
        per_variant[variant] = {
            "cities_compared": len(deltas),
            "wins_vs_mae": wins,
            "losses_vs_mae": len(deltas) - wins,
            "mean_lpips_delta_vs_mae": mean_delta,
            "per_city": deltas,
        }

    ranked = sorted(
        (
            {
                "variant": k,
                "mean_lpips_delta_vs_mae": v["mean_lpips_delta_vs_mae"],
                "wins_vs_mae": v["wins_vs_mae"],
                "cities_compared": v["cities_compared"],
            }
            for k, v in per_variant.items()
        ),
        key=lambda x: x["mean_lpips_delta_vs_mae"],
    )
    return {"vs_mae": per_variant, "ranked_by_mean_lpips_delta": ranked}


def _gpu_worker(
    device: int,
    work: queue.Queue,
    results: list,
    results_lock: threading.Lock,
    iters: int,
) -> None:
    while True:
        try:
            job = work.get_nowait()
        except queue.Empty:
            return
        city, variant, loss, eps = job
        try:
            row = _run_one(city, variant, loss, eps, device, iters)
            with results_lock:
                results.append(row)
            print(f"[gpu{device}] DONE {city} {variant} lpips={row['lpips']:.4f}", flush=True)
        except Exception as exc:
            print(f"[gpu{device}] FAILED {city} {variant}: {exc}", flush=True)
            with results_lock:
                results.append(
                    {
                        "city": city,
                        "variant": variant,
                        "recon_loss": loss,
                        "charbonnier_eps": eps,
                        "skipped": False,
                        "error": str(exc),
                    }
                )
        finally:
            work.task_done()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="+", default=FOCUS_CITIES)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "single_samples" / "sweep_results" / "recon_loss_charbonnier.json",
    )
    args = ap.parse_args()

    work: queue.Queue = queue.Queue()
    rows: list[dict] = []
    for city in args.cities:
        for variant, loss, eps in VARIANTS:
            mpath = _metrics_path(city, variant, args.iters)
            if args.skip_existing and mpath.is_file():
                rows.append(_summarize(city, variant, loss, eps, mpath, skipped=True))
                print(f"[skip] {city} {variant}", flush=True)
            else:
                work.put((city, variant, loss, eps))

    results_lock = threading.Lock()
    threads = []
    n_workers = max(1, min(args.gpus, work.qsize() or 1))
    for device in range(n_workers):
        t = threading.Thread(
            target=_gpu_worker,
            args=(device, work, rows, results_lock, args.iters),
            daemon=True,
        )
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    rows = [r for r in rows if "error" not in r]
    rows.sort(key=lambda r: (r["city"], r["variant"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "iters_budget": args.iters,
        "early_stop_metric": "lpips",
        "variants": [
            {"variant": v, "recon_loss": loss, "charbonnier_eps": eps}
            for v, loss, eps in VARIANTS
        ],
        "rows": rows,
        "summary": _build_summary(rows),
    }
    args.out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["summary"], indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
