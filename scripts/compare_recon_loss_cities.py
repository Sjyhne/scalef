#!/usr/bin/env python3
"""Run MAE vs MSE recon_loss comparison across focus cities (LPIPS early stop)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FOCUS_CITIES = ["asker", "vennesla", "trondheim", "bergen", "rana", "tromso", "amli"]
RUN_NAME = "stop_lpips_{loss}_3k"


def _metrics_path(city: str, loss: str) -> Path:
    return ROOT / "single_samples" / city / "sample" / RUN_NAME.format(loss=loss) / "metrics.json"


def _run_city(city: str, device: int, iters: int, skip_existing: bool) -> list[dict]:
    rows: list[dict] = []
    for loss in ("mae", "mse"):
        mpath = _metrics_path(city, loss)
        if skip_existing and mpath.is_file():
            rows.append(_summarize(city, loss, mpath, skipped=True))
            continue
        cmd = [
            sys.executable,
            str(ROOT / "optimize.py"),
            "--dataset",
            city,
            "--s2-dir",
            str(ROOT / "data" / "s2_revisits" / f"{city}_lr512"),
            "--run_name",
            RUN_NAME.format(loss=loss),
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
        print(f"[gpu{device}] {city} {loss} ...", flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)
        rows.append(_summarize(city, loss, mpath, skipped=False))
    return rows


def _summarize(city: str, loss: str, mpath: Path, *, skipped: bool) -> dict:
    m = json.loads(mpath.read_text())
    es = m.get("early_stop") or {}
    return {
        "city": city,
        "recon_loss": loss,
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="+", default=FOCUS_CITIES)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--gpus", type=int, default=6)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "single_samples" / "sweep_results" / "recon_loss_mae_vs_mse.json",
    )
    args = ap.parse_args()

    cities = list(args.cities)
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=min(args.gpus, len(cities))) as pool:
        futures = {
            pool.submit(_run_city, city, i % args.gpus, args.iters, args.skip_existing): city
            for i, city in enumerate(cities)
        }
        for fut in as_completed(futures):
            city = futures[fut]
            try:
                rows.extend(fut.result())
            except Exception as exc:
                print(f"FAILED {city}: {exc}", flush=True)
                raise

    rows.sort(key=lambda r: (r["city"], r["recon_loss"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "iters_budget": args.iters,
        "early_stop_metric": "lpips",
        "rows": rows,
        "summary": _build_summary(rows),
    }
    args.out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["summary"], indent=2))
    print(f"Wrote {args.out}")


def _build_summary(rows: list[dict]) -> dict:
    by_city: dict[str, dict[str, dict]] = {}
    for r in rows:
        by_city.setdefault(r["city"], {})[r["recon_loss"]] = r
    deltas = []
    for city, pair in sorted(by_city.items()):
        if "mae" not in pair or "mse" not in pair:
            continue
        mae, mse = pair["mae"], pair["mse"]
        deltas.append(
            {
                "city": city,
                "lpips_delta_mse_minus_mae": mse["lpips"] - mae["lpips"],
                "psnr_delta_mse_minus_mae": mse["psnr"] - mae["psnr"],
                "mae_wins_lpips": mae["lpips"] < mse["lpips"],
                "mae_lpips": mae["lpips"],
                "mse_lpips": mse["lpips"],
            }
        )
    mae_wins = sum(1 for d in deltas if d["mae_wins_lpips"])
    return {
        "cities_compared": len(deltas),
        "mae_wins_lpips": mae_wins,
        "mse_wins_lpips": len(deltas) - mae_wins,
        "mean_lpips_delta_mse_minus_mae": sum(d["lpips_delta_mse_minus_mae"] for d in deltas) / max(len(deltas), 1),
        "per_city": deltas,
    }


if __name__ == "__main__":
    main()
