#!/usr/bin/env python3
"""Small verification sweep: does the Bergen hashgrid winner hold on other AOIs?

Runs the Bergen-winning config plus a few nearby challengers on each city,
then ranks by center-spot PSNR (same selection metric as the original sweep).

Example
-------
python scripts/verify_best_hashgrid.py --devices 0,1,2,3 --iters 2000
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from queue import Queue
from threading import Lock

ROOT = Path(__file__).resolve().parent.parent

# Bergen Stage-B winner
WINNER = {
    "hash_interpolation": "linear",
    "hash_log2_hashmap_size": 21,
    "hash_n_levels": 16,
    "learning_rate": 0.001,
    "recon_loss": "mse",
    "use_laplace_nll": False,
}

# Nearby challengers that nearly tied / lost on Bergen
VARIANTS = {
    "winner": WINNER,
    "smoothstep": {**WINNER, "hash_interpolation": "smoothstep"},
    "compact": {
        "hash_interpolation": "linear",
        "hash_log2_hashmap_size": 19,
        "hash_n_levels": 12,
        "learning_rate": 0.001,
        "recon_loss": "mse",
        "use_laplace_nll": False,
    },
    "lr2e3": {**WINNER, "learning_rate": 0.002},
}

DEFAULT_CITIES = [
    "bergen",
    "kristiansand",
    "rana",
    "sandvika",
    "stavanger",
    "tromso",
    "trondheim",
]


def _run_name(variant: str) -> str:
    return f"verify_{variant}"


def _metrics_path(city: str, variant: str) -> Path:
    return ROOT / "single_samples" / city / "sample" / _run_name(variant) / "metrics.json"


def _read_metrics(city: str, variant: str) -> dict | None:
    p = _metrics_path(city, variant)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _build_cmd(city: str, variant: str, cfg: dict, iters: int, device: str) -> list[str]:
    cmd = [
        sys.executable,
        str(ROOT / "optimize.py"),
        "--dataset", city,
        "--s2-dir", f"data/s2_revisits/{city}",
        "--df", "4",
        "--scale_factor", "4",
        "--input_projection", "hashgrid",
        "--sample_id", "sample",
        "--run_name", _run_name(variant),
        "--iters", str(iters),
        "--eval_every", "0",
        "--eval-spot-hr-px", "32",
        "--device", str(device),
        "--hash_interpolation", str(cfg["hash_interpolation"]),
        "--hash_log2_hashmap_size", str(cfg["hash_log2_hashmap_size"]),
        "--hash_n_levels", str(cfg["hash_n_levels"]),
        "--learning_rate", f"{float(cfg['learning_rate']):.2e}",
        "--recon_loss", str(cfg["recon_loss"]),
    ]
    if cfg.get("use_laplace_nll"):
        cmd.append("--use_laplace_nll")
    return cmd


def _run_one(job: dict, device_pool: "Queue[str]", dry_run: bool = False) -> dict:
    city = job["city"]
    variant = job["variant"]
    cfg = job["cfg"]
    iters = job["iters"]

    # Claim a GPU at start, not at submission time: under Exclusive_Process mode
    # a pre-assigned device can still be held by another running job.
    device = device_pool.get()
    try:
        cmd = _build_cmd(city, variant, cfg, iters, device)
        print(f"\n>>> {city}/{variant}  device={device}", flush=True)
        row = {
            "city": city,
            "variant": variant,
            **cfg,
            "iters": iters,
            "device": device,
            "run_name": _run_name(variant),
            "spot_psnr": None,
            "full_psnr": None,
            "bilinear_psnr": None,
            "spot_delta": None,
            "status": "pending",
        }
        if dry_run:
            row["status"] = "dry_run"
            return row

        r = subprocess.run(cmd, cwd=str(ROOT))
    finally:
        device_pool.put(device)

    if r.returncode != 0:
        row["status"] = f"failed:{r.returncode}"
        print(f"    FAILED {city}/{variant} (exit {r.returncode})", flush=True)
        return row

    metrics = _read_metrics(city, variant)
    if not metrics:
        row["status"] = "missing_metrics"
        return row

    spot = (metrics.get("fixed_spot") or {}).get("model_psnr")
    bil_spot = (metrics.get("fixed_spot") or {}).get("bilinear_psnr")
    full = metrics.get("model_psnr")
    bil = metrics.get("bilinear_psnr")
    row["spot_psnr"] = float(spot) if spot is not None else None
    row["full_psnr"] = float(full) if full is not None else None
    row["bilinear_psnr"] = float(bil) if bil is not None else None
    if spot is not None and bil_spot is not None:
        row["spot_delta"] = float(spot) - float(bil_spot)
    row["status"] = "ok"
    spot_s = "nan" if row["spot_psnr"] is None else f"{row['spot_psnr']:.2f}"
    full_s = "nan" if row["full_psnr"] is None else f"{row['full_psnr']:.2f}"
    print(f"    {city}/{variant}: spot={spot_s} full={full_s}", flush=True)
    return row


def _save(rows: list[dict], results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(results_dir / "verify_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    ok = [r for r in rows if r.get("status") == "ok" and r.get("spot_psnr") is not None]
    by_city: dict[str, list[dict]] = {}
    for r in ok:
        by_city.setdefault(r["city"], []).append(r)

    ranking = []
    for city, city_rows in sorted(by_city.items()):
        ranked = sorted(city_rows, key=lambda r: float(r["spot_psnr"]), reverse=True)
        best = ranked[0]
        winner_row = next((r for r in ranked if r["variant"] == "winner"), None)
        ranking.append(
            {
                "city": city,
                "best_variant": best["variant"],
                "best_spot_psnr": best["spot_psnr"],
                "winner_spot_psnr": None if winner_row is None else winner_row["spot_psnr"],
                "winner_rank": None
                if winner_row is None
                else 1 + next(i for i, r in enumerate(ranked) if r["variant"] == "winner"),
                "n_variants": len(ranked),
                "winner_gap_to_best": None
                if winner_row is None
                else float(winner_row["spot_psnr"]) - float(best["spot_psnr"]),
            }
        )

    (results_dir / "city_ranking.json").write_text(json.dumps(ranking, indent=2))
    (results_dir / "all_rows.json").write_text(json.dumps(rows, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--devices", default="0", help="Comma-separated CUDA devices (default: 0)")
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument(
        "--cities",
        default=",".join(DEFAULT_CITIES),
        help="Comma-separated cities under data/s2_revisits/",
    )
    ap.add_argument(
        "--variants",
        default=",".join(VARIANTS.keys()),
        help="Comma-separated variant names",
    )
    ap.add_argument("--resume", action="store_true", help="Skip jobs that already have metrics.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    cities = [c.strip() for c in args.cities.split(",") if c.strip()]
    variant_names = [v.strip() for v in args.variants.split(",") if v.strip()]
    for v in variant_names:
        if v not in VARIANTS:
            raise SystemExit(f"Unknown variant {v!r}. Choose from: {', '.join(VARIANTS)}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = ROOT / "single_samples" / "sweep_results" / f"{ts}_verify_best"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "variants.json").write_text(json.dumps({k: VARIANTS[k] for k in variant_names}, indent=2))

    jobs = []
    for city in cities:
        for variant in variant_names:
            jobs.append(
                {
                    "city": city,
                    "variant": variant,
                    "cfg": VARIANTS[variant],
                    "iters": args.iters,
                }
            )

    print(f"Verification sweep: {len(cities)} cities × {len(variant_names)} variants = {len(jobs)} jobs")
    print(f"Devices: {devices}")
    print(f"Results: {results_dir}")

    rows: list[dict] = []
    lock = Lock()

    pending = []
    for job in jobs:
        if args.resume:
            existing = _read_metrics(job["city"], job["variant"])
            if existing is not None:
                spot = (existing.get("fixed_spot") or {}).get("model_psnr")
                row = {
                    "city": job["city"],
                    "variant": job["variant"],
                    **job["cfg"],
                    "iters": job["iters"],
                    "device": "resume",
                    "run_name": _run_name(job["variant"]),
                    "spot_psnr": float(spot) if spot is not None else None,
                    "full_psnr": float(existing.get("model_psnr") or 0.0),
                    "bilinear_psnr": float(existing.get("bilinear_psnr") or 0.0),
                    "spot_delta": None,
                    "status": "resume",
                }
                bil_spot = (existing.get("fixed_spot") or {}).get("bilinear_psnr")
                if spot is not None and bil_spot is not None:
                    row["spot_delta"] = float(spot) - float(bil_spot)
                rows.append(row)
                print(f"    skip (resume): {job['city']}/{job['variant']} spot={row['spot_psnr']}")
                continue
        pending.append(job)

    device_pool: Queue[str] = Queue()
    for d in devices:
        device_pool.put(d)

    def _worker(job: dict) -> dict:
        return _run_one(job, device_pool, dry_run=args.dry_run)

    with ThreadPoolExecutor(max_workers=max(1, len(devices))) as ex:
        futs = [ex.submit(_worker, job) for job in pending]
        for fut in as_completed(futs):
            row = fut.result()
            with lock:
                rows.append(row)
                _save(rows, results_dir)

    _save(rows, results_dir)

    # Print compact ranking table
    print("\n=== Per-city ranking (spot PSNR) ===")
    ranking = json.loads((results_dir / "city_ranking.json").read_text())
    wins = 0
    for r in ranking:
        mark = "OK" if r["best_variant"] == "winner" else f"ALT={r['best_variant']}"
        if r["best_variant"] == "winner":
            wins += 1
        spot = r.get("best_spot_psnr")
        gap = r.get("winner_gap_to_best")
        spot_s = "nan" if spot is None else f"{float(spot):.2f}"
        gap_s = "nan" if gap is None else f"{float(gap):+.3f}"
        print(
            f"  {r['city']:12} best={r['best_variant']:10} "
            f"spot={spot_s}  "
            f"winner_rank={r['winner_rank']}/{r['n_variants']}  "
            f"gap={gap_s}  [{mark}]"
        )
    print(f"\nWinner is best on {wins}/{len(ranking)} cities")
    print(f"Results saved to: {results_dir}")


if __name__ == "__main__":
    main()
