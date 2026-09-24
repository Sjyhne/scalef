#!/usr/bin/env python3
"""Grid over HR→LR degradation operator × hashgrid max resolution.

Two hypotheses are under test:

1. The default ``s2_psf`` degradation (DSen2, sigma = 1/scale = 0.25 HR px) is far
   narrower than the real Sentinel-2 MTF (``s2_psf_m``, sigma ~1.1-1.7 HR px at
   df=4). Fitting through a too-sharp operator forces the recovered HR to be
   blurrier than reality.
2. The hashgrid's finest level defaults to the LR grid, so the encoder cannot
   represent detail above LR Nyquist at all. ``--hash_max_resolution_mult``
   lifts that cap (mult = scale_factor puts the finest level on the HR grid).

Selection is on LPIPS, not PSNR: PSNR rewards blur, so a correctly sharpened
result can score lower on PSNR while being perceptually better.

Example
-------
python scripts/sweep_psf_resolution.py --devices 0,1,2,3,4,5,6,7 --iters 2000
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

# Verified best hashgrid config (see scripts/verify_best_hashgrid.py).
BASE = {
    "hash_interpolation": "linear",
    "hash_log2_hashmap_size": 21,
    "hash_n_levels": 16,
    "learning_rate": 0.001,
    "recon_loss": "mse",
    "lr_degradation": "s2_psf",
    "hash_max_resolution_mult": 1.0,
}

VARIANTS: dict[str, dict] = {
    # Current production config: narrow PSF, encoder capped at the LR grid.
    "baseline": {},
    # Isolate the degradation operator at fixed encoder capacity.
    "psfm_1x": {"lr_degradation": "s2_psf_m"},
    # Relax the encoder band-limit under the physical PSF.
    "psfm_2x": {"lr_degradation": "s2_psf_m", "hash_max_resolution_mult": 2.0},
    "psfm_4x": {"lr_degradation": "s2_psf_m", "hash_max_resolution_mult": 4.0},
    # Interaction check: does resolution help without fixing the PSF?
    "psf_4x": {"hash_max_resolution_mult": 4.0},
    # Are hash collisions binding at 4x? (finest level holds ~1.2M vertices)
    "psfm_4x_big": {
        "lr_degradation": "s2_psf_m",
        "hash_max_resolution_mult": 4.0,
        "hash_log2_hashmap_size": 23,
    },
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

# Lower is better -> sort ascending.
PRIMARY_METRIC = "spot_lpips"


def _cfg(variant: str) -> dict:
    return {**BASE, **VARIANTS[variant]}


def _run_name(variant: str) -> str:
    return f"psfres_{variant}"


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
        "--lr_degradation", str(cfg["lr_degradation"]),
        "--hash_interpolation", str(cfg["hash_interpolation"]),
        "--hash_log2_hashmap_size", str(cfg["hash_log2_hashmap_size"]),
        "--hash_n_levels", str(cfg["hash_n_levels"]),
        "--hash_max_resolution_mult", str(cfg["hash_max_resolution_mult"]),
        "--learning_rate", f"{float(cfg['learning_rate']):.2e}",
        "--recon_loss", str(cfg["recon_loss"]),
    ]
    return cmd


def _extract(metrics: dict) -> dict:
    spot = metrics.get("fixed_spot") or {}

    def _f(v):
        return None if v is None else float(v)

    return {
        "full_psnr": _f((metrics.get("psnr") or {}).get("model")),
        "full_ssim": _f((metrics.get("ssim") or {}).get("model")),
        "full_lpips": _f((metrics.get("lpips") or {}).get("model")),
        "bilinear_psnr": _f((metrics.get("psnr") or {}).get("bilinear")),
        "bilinear_ssim": _f((metrics.get("ssim") or {}).get("bilinear")),
        "bilinear_lpips": _f((metrics.get("lpips") or {}).get("bilinear")),
        "spot_psnr": _f(spot.get("model_psnr")),
        "spot_ssim": _f(spot.get("model_ssim")),
        "spot_lpips": _f(spot.get("model_lpips")),
        "train_seconds": _f(metrics.get("training_time_seconds")),
    }


def _empty_metrics() -> dict:
    return {
        k: None
        for k in (
            "full_psnr", "full_ssim", "full_lpips",
            "bilinear_psnr", "bilinear_ssim", "bilinear_lpips",
            "spot_psnr", "spot_ssim", "spot_lpips", "train_seconds",
        )
    }


def _run_one(job: dict, device_pool: "Queue[str]", dry_run: bool = False) -> dict:
    city, variant, cfg = job["city"], job["variant"], job["cfg"]

    # GPUs here are in Exclusive_Process mode, so a device must be claimed when
    # the job actually starts -- pre-assigning at submission time lets a freed
    # worker collide with a still-running job on the same device.
    device = device_pool.get()
    try:
        cmd = _build_cmd(city, variant, cfg, job["iters"], device)
        print(f"\n>>> {city}/{variant}  device={device}", flush=True)

        row = {
            "city": city,
            "variant": variant,
            "lr_degradation": cfg["lr_degradation"],
            "max_res_mult": cfg["hash_max_resolution_mult"],
            "log2_hashmap": cfg["hash_log2_hashmap_size"],
            "iters": job["iters"],
            "device": device,
            "run_name": _run_name(variant),
            **_empty_metrics(),
            "status": "pending",
        }
        if dry_run:
            row["status"] = "dry_run"
            return row

        r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    finally:
        device_pool.put(device)

    if r.returncode != 0:
        row["status"] = f"failed:{r.returncode}"
        tail = "\n".join((r.stderr or "").strip().splitlines()[-12:])
        print(f"    FAILED {city}/{variant} (exit {r.returncode})\n{tail}", flush=True)
        return row

    metrics = _read_metrics(city, variant)
    if not metrics:
        row["status"] = "missing_metrics"
        return row

    row.update(_extract(metrics))
    row["status"] = "ok"
    print(
        f"    {city}/{variant}: "
        f"LPIPS spot={row['spot_lpips']:.4f} full={row['full_lpips']:.4f}  "
        f"SSIM={row['full_ssim']:.4f}  PSNR={row['full_psnr']:.2f}",
        flush=True,
    )
    return row


def _save(rows: list[dict], results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for row in rows for k in row})
    with open(results_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    (results_dir / "all_rows.json").write_text(json.dumps(rows, indent=2))

    ok = [r for r in rows if r.get("status") in {"ok", "resume"} and r.get(PRIMARY_METRIC) is not None]
    by_city: dict[str, list[dict]] = {}
    for r in ok:
        by_city.setdefault(r["city"], []).append(r)

    ranking = []
    for city, city_rows in sorted(by_city.items()):
        ranked = sorted(city_rows, key=lambda r: float(r[PRIMARY_METRIC]))
        best = ranked[0]
        baseline = next((r for r in ranked if r["variant"] == "baseline"), None)
        ranking.append(
            {
                "city": city,
                "best_variant": best["variant"],
                "best_spot_lpips": best["spot_lpips"],
                "baseline_spot_lpips": None if baseline is None else baseline["spot_lpips"],
                "lpips_gain_vs_baseline": None
                if baseline is None
                else float(baseline["spot_lpips"]) - float(best["spot_lpips"]),
                "order": [r["variant"] for r in ranked],
            }
        )
    (results_dir / "city_ranking.json").write_text(json.dumps(ranking, indent=2))

    # Mean rank per variant across cities.
    tally: dict[str, list[float]] = {}
    for entry in ranking:
        for pos, variant in enumerate(entry["order"], start=1):
            tally.setdefault(variant, []).append(pos)
    agg = sorted(
        ({"variant": v, "mean_rank": sum(p) / len(p), "n": len(p)} for v, p in tally.items()),
        key=lambda d: d["mean_rank"],
    )
    (results_dir / "variant_ranking.json").write_text(json.dumps(agg, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--devices", default="0", help="Comma-separated CUDA devices")
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--cities", default=",".join(DEFAULT_CITIES))
    ap.add_argument("--variants", default=",".join(VARIANTS))
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
    results_dir = ROOT / "single_samples" / "sweep_results" / f"{ts}_psf_resolution"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "variants.json").write_text(
        json.dumps({v: _cfg(v) for v in variant_names}, indent=2)
    )

    rows: list[dict] = []
    pending: list[dict] = []
    for city in cities:
        for variant in variant_names:
            job = {"city": city, "variant": variant, "cfg": _cfg(variant), "iters": args.iters}
            if args.resume:
                existing = _read_metrics(city, variant)
                if existing is not None:
                    cfg = job["cfg"]
                    row = {
                        "city": city,
                        "variant": variant,
                        "lr_degradation": cfg["lr_degradation"],
                        "max_res_mult": cfg["hash_max_resolution_mult"],
                                    "log2_hashmap": cfg["hash_log2_hashmap_size"],
                        "iters": args.iters,
                        "device": "resume",
                        "run_name": _run_name(variant),
                        **_extract(existing),
                        "status": "resume",
                    }
                    rows.append(row)
                    print(f"    skip (resume): {city}/{variant}")
                    continue
            pending.append(job)

    print(
        f"PSF × resolution grid: {len(cities)} cities × {len(variant_names)} variants "
        f"= {len(cities) * len(variant_names)} jobs ({len(pending)} to run)"
    )
    print(f"Devices: {devices}\nResults: {results_dir}\nSelecting on: {PRIMARY_METRIC} (lower is better)")

    device_pool: Queue[str] = Queue()
    for d in devices:
        device_pool.put(d)

    lock = Lock()
    with ThreadPoolExecutor(max_workers=max(1, len(devices))) as ex:
        futs = [ex.submit(_run_one, job, device_pool, args.dry_run) for job in pending]
        for fut in as_completed(futs):
            row = fut.result()
            with lock:
                rows.append(row)
                _save(rows, results_dir)

    _save(rows, results_dir)

    print("\n=== Per-city ranking (spot LPIPS, lower is better) ===")
    for r in json.loads((results_dir / "city_ranking.json").read_text()):
        gain = r["lpips_gain_vs_baseline"]
        gain_s = "nan" if gain is None else f"{gain:+.4f}"
        print(
            f"  {r['city']:14} best={r['best_variant']:14} "
            f"lpips={float(r['best_spot_lpips']):.4f}  vs baseline {gain_s}"
        )

    print("\n=== Variant mean rank across cities ===")
    for a in json.loads((results_dir / "variant_ranking.json").read_text()):
        print(f"  {a['variant']:14} mean_rank={a['mean_rank']:.2f}  (n={a['n']})")

    print(f"\nResults saved to: {results_dir}")


if __name__ == "__main__":
    main()
