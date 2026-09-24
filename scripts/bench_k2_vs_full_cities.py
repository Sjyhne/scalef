#!/usr/bin/env python3
"""Multi-city: full-field vs fused k2 under matched holdout stop.

Reports audit LPIPS delta so you can see how much worse k2 is across sites.
"""

from __future__ import annotations

import argparse
import json
import queue
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FOCUS_CITIES = ["asker", "vennesla", "trondheim", "bergen", "rana", "tromso", "amli"]

VARIANTS = {
    "full": [],
    "k2": ["--lr_tile", "128", "--lr_tiles_per_step", "2"],
}


RUN_TAG = ""


def _run_name(variant: str, iters: int) -> str:
    return f"bench_{variant}_holdout_{iters // 1000}k{RUN_TAG}"


def _metrics_path(city: str, variant: str, iters: int) -> Path:
    return ROOT / "single_samples" / city / "sample" / _run_name(variant, iters) / "metrics.json"


def _cmd(city: str, variant: str, device: int, iters: int) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "optimize.py"),
        "--dataset",
        city,
        "--s2-dir",
        str(ROOT / "data" / "s2_revisits" / f"{city}_lr512"),
        "--run_name",
        _run_name(variant, iters),
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
        "audit_lpips": m["lpips"]["model"],
        "psnr": m["psnr"]["model"],
        "ssim": m["ssim"]["model"],
        "completed_iters": m["completed_iters"],
        "best_iter": es.get("best_val_iter"),
        "stopped_iter": es.get("stopped_iter"),
        "training_time_s": m.get("training_time_seconds"),
        "peak_memory_gb": m.get("peak_memory_gb"),
        "metrics_path": str(mpath.relative_to(ROOT)),
    }


def _run_one(
    city: str, variant: str, device: int, iters: int, skip_existing: bool
) -> dict:
    mpath = _metrics_path(city, variant, iters)
    if skip_existing and mpath.is_file():
        print(f"[gpu{device}] skip {city}/{variant}", flush=True)
        return _summarize(city, variant, mpath, skipped=True)
    print(f"[gpu{device}] {city}/{variant} ...", flush=True)
    subprocess.run(_cmd(city, variant, device, iters), cwd=ROOT, check=True)
    return _summarize(city, variant, mpath, skipped=False)


def _worker(
    gpu: int,
    work: queue.Queue,
    results: list,
    lock: threading.Lock,
    iters: int,
    skip_existing: bool,
) -> None:
    while True:
        try:
            city, variant = work.get_nowait()
        except queue.Empty:
            return
        try:
            row = _run_one(city, variant, gpu, iters, skip_existing)
            with lock:
                results.append(row)
            print(f"DONE {city}/{variant}", flush=True)
        except Exception as exc:  # noqa: BLE001
            with lock:
                results.append(
                    {
                        "city": city,
                        "variant": variant,
                        "error": str(exc),
                    }
                )
            print(f"FAIL {city}/{variant}: {exc}", flush=True)
        finally:
            work.task_done()


def _build_summary(rows: list[dict]) -> dict:
    by_city: dict[str, dict] = {}
    for r in rows:
        if "error" in r:
            continue
        by_city.setdefault(r["city"], {})[r["variant"]] = r

    deltas = []
    for city, variants in sorted(by_city.items()):
        full = variants.get("full")
        k2 = variants.get("k2")
        if not full or not k2:
            continue
        d_lpips = k2["audit_lpips"] - full["audit_lpips"]
        deltas.append(
            {
                "city": city,
                "full_lpips": full["audit_lpips"],
                "k2_lpips": k2["audit_lpips"],
                "lpips_delta": d_lpips,
                "full_psnr": full["psnr"],
                "k2_psnr": k2["psnr"],
                "psnr_delta": k2["psnr"] - full["psnr"],
                "full_time_s": full.get("training_time_s"),
                "k2_time_s": k2.get("training_time_s"),
                "full_peak_gb": full.get("peak_memory_gb"),
                "k2_peak_gb": k2.get("peak_memory_gb"),
                "full_best_iter": full.get("best_iter"),
                "k2_best_iter": k2.get("best_iter"),
                "within_005": d_lpips <= 0.005,
                "within_015": d_lpips <= 0.015,
            }
        )

    if not deltas:
        return {"n_cities": 0, "per_city": []}

    lp = [d["lpips_delta"] for d in deltas]
    return {
        "n_cities": len(deltas),
        "mean_lpips_delta": sum(lp) / len(lp),
        "median_lpips_delta": sorted(lp)[len(lp) // 2],
        "min_lpips_delta": min(lp),
        "max_lpips_delta": max(lp),
        "n_within_005": sum(1 for d in deltas if d["within_005"]),
        "n_within_015": sum(1 for d in deltas if d["within_015"]),
        "mean_time_ratio_k2_over_full": (
            sum(
                d["k2_time_s"] / d["full_time_s"]
                for d in deltas
                if d.get("k2_time_s") and d.get("full_time_s")
            )
            / max(
                1,
                sum(
                    1
                    for d in deltas
                    if d.get("k2_time_s") and d.get("full_time_s")
                ),
            )
        ),
        "per_city": deltas,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cities", nargs="+", default=FOCUS_CITIES)
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--gpu-offset", type=int, default=0)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument(
        "--run-tag",
        default="",
        help="Suffix appended to run names, so a re-run does not overwrite earlier results.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "single_samples" / "sweep_results" / "bench_k2_vs_full_cities.json",
    )
    args = ap.parse_args()

    global RUN_TAG
    RUN_TAG = str(args.run_tag)

    work: queue.Queue = queue.Queue()
    for city in args.cities:
        for variant in VARIANTS:
            work.put((city, variant))

    results: list[dict] = []
    lock = threading.Lock()
    threads = [
        threading.Thread(
            target=_worker,
            args=(
                args.gpu_offset + i,
                work,
                results,
                lock,
                args.iters,
                args.skip_existing,
            ),
            daemon=True,
        )
        for i in range(max(1, min(args.gpus, work.qsize())))
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    results.sort(key=lambda r: (r.get("city", ""), r.get("variant", "")))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "iters_budget": args.iters,
        "early_stop_metric": "holdout_mse",
        "recon_loss": "charbonnier",
        "charbonnier_eps": 0.01,
        "spatial_holdout": 0.1,
        "k2": {"lr_tile": 128, "lr_tiles_per_step": 2, "fused": True},
        "cities": list(args.cities),
        "rows": results,
        "summary": _build_summary(results),
    }
    args.out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["summary"], indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
