#!/usr/bin/env python3
"""Benchmark HR→LR degradation operators across cities (with spatial alignment).

Compares ``area``, ``s2_psf``, ``s2_psf_m``, and ``psfm_2x`` on fixed LR-size
crops. Evaluation uses harmonized + spatially aligned NIB HR when
``eval/spatial_alignment.json`` is present.

Example
-------
python scripts/compare_degradations.py \\
  --cities asker vennesla trondheim tromso bergen rana amli \\
  --lr-size 512 --devices 0,1,2,3,4,5,6,7 --force
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue

ROOT = Path(__file__).resolve().parent.parent

BASE = {
    "hash_interpolation": "linear",
    "hash_log2_hashmap_size": 21,
    "hash_n_levels": 16,
    "learning_rate": 0.001,
    "recon_loss": "mae",
}

VARIANTS = {
    "area": {"lr_degradation": "area", "hash_max_resolution_mult": 1.0},
    "s2_psf": {"lr_degradation": "s2_psf", "hash_max_resolution_mult": 1.0},
    "s2_psf_m": {"lr_degradation": "s2_psf_m", "hash_max_resolution_mult": 1.0},
    "psfm_2x": {"lr_degradation": "s2_psf_m", "hash_max_resolution_mult": 2.0},
}

FOCUS_CITIES = [
    "asker",
    "vennesla",
    "trondheim",
    "bergen",
    "rana",
    "tromso",
    "amli",
]


def _run_name(variant: str, lr_size: int) -> str:
    # Distinct from older unaligned deg_* runs.
    return f"degA_{variant}_lr{lr_size}"


def _metrics_path(city: str, variant: str, lr_size: int) -> Path:
    return ROOT / "single_samples" / city / "sample" / _run_name(variant, lr_size) / "metrics.json"


def _ensure_lr_variant(city: str, lr_size: int) -> Path:
    s2_dir = ROOT / "data" / "s2_revisits" / f"{city}_lr{lr_size}"
    if (s2_dir / "meta.json").is_file():
        return s2_dir
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "make_lr_size_variants.py"),
        "--city",
        city,
        "--sizes",
        str(lr_size),
    ]
    print(f"creating LR variant: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    if not (s2_dir / "meta.json").is_file():
        raise FileNotFoundError(f"failed to create {s2_dir}/meta.json")
    return s2_dir


def _build_cmd(city: str, variant: str, cfg: dict, lr_size: int, iters: int, device: str) -> list[str]:
    s2_dir = ROOT / "data" / "s2_revisits" / f"{city}_lr{lr_size}"
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
        _run_name(variant, lr_size),
        "--iters",
        str(iters),
        "--eval_every",
        "0",
        "--eval-spot-hr-px",
        "32",
        "--device",
        str(device),
        "--hash_interpolation",
        BASE["hash_interpolation"],
        "--hash_log2_hashmap_size",
        str(BASE["hash_log2_hashmap_size"]),
        "--hash_n_levels",
        str(BASE["hash_n_levels"]),
        "--learning_rate",
        f"{BASE['learning_rate']:.2e}",
        "--recon_loss",
        BASE["recon_loss"],
        "--lr_degradation",
        str(cfg["lr_degradation"]),
        "--hash_max_resolution_mult",
        str(cfg["hash_max_resolution_mult"]),
        "--num_samples",
        "16",
        "--spatial_alignment_path",
        str(ROOT / "eval" / "spatial_alignment.json"),
    ]


def _row_from_metrics(city: str, variant: str, lr_size: int, status: str) -> dict:
    cfg = VARIANTS[variant]
    mpath = _metrics_path(city, variant, lr_size)
    m = json.loads(mpath.read_text())
    spot = m.get("fixed_spot") or {}
    return {
        "city": city,
        "variant": variant,
        "status": status,
        "lr_degradation": cfg["lr_degradation"],
        "max_res_mult": cfg["hash_max_resolution_mult"],
        "psnr": m.get("psnr", {}).get("model"),
        "ssim": m.get("ssim", {}).get("model"),
        "lpips": m.get("lpips", {}).get("model"),
        "bil_psnr": m.get("psnr", {}).get("bilinear"),
        "bil_ssim": m.get("ssim", {}).get("bilinear"),
        "bil_lpips": m.get("lpips", {}).get("bilinear"),
        "d_psnr": m.get("psnr", {}).get("improvement"),
        "d_ssim": m.get("ssim", {}).get("improvement"),
        "d_lpips": m.get("lpips", {}).get("improvement"),
        "spot_psnr": spot.get("model_psnr"),
        "spot_lpips": spot.get("model_lpips"),
        "eval_mask": m.get("eval_mask"),
        "metrics_path": str(mpath),
    }


def _run_one(job: dict, pool: Queue, dry_run: bool) -> dict:
    device = pool.get()
    city, variant, cfg, lr_size, iters = (
        job["city"],
        job["variant"],
        job["cfg"],
        job["lr_size"],
        job["iters"],
    )
    cmd = _build_cmd(city, variant, cfg, lr_size, iters, device)
    print(f"\n>>> {city}/{variant}  device={device}", flush=True)
    print(" ".join(cmd), flush=True)
    row = {
        "city": city,
        "variant": variant,
        "device": device,
        "lr_degradation": cfg["lr_degradation"],
        "max_res_mult": cfg["hash_max_resolution_mult"],
    }
    try:
        if dry_run:
            row["status"] = "dry_run"
            return row
        proc = subprocess.run(cmd, cwd=ROOT)
        row["exit"] = proc.returncode
        row["status"] = "ok" if proc.returncode == 0 else "fail"
        mpath = _metrics_path(city, variant, lr_size)
        if mpath.exists():
            row.update(_row_from_metrics(city, variant, lr_size, row["status"]))
    finally:
        pool.put(device)
    print(f"exit={row.get('exit')} {city}/{variant}", flush=True)
    return row


def _print_table(rows: list[dict]) -> None:
    print(
        f"\n{'city':10s} {'variant':10s} {'deg':8s} {'mult':>4} "
        f"{'PSNR':>7} {'SSIM':>6} {'LPIPS':>6} "
        f"{'ΔP':>6} {'ΔS':>6} {'ΔL':>6}"
    )
    for r in rows:
        print(
            f"{r.get('city', '?'):10s} {r['variant']:10s} {r.get('lr_degradation', '?'):8s} "
            f"{float(r.get('max_res_mult') or 0):>4.1f} "
            f"{(r.get('psnr') or float('nan')):7.2f} {(r.get('ssim') or float('nan')):6.3f} "
            f"{(r.get('lpips') or float('nan')):6.3f} "
            f"{(r.get('d_psnr') or float('nan')):6.2f} {(r.get('d_ssim') or float('nan')):6.3f} "
            f"{(r.get('d_lpips') or float('nan')):6.3f}"
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--city", default=None, help="Single city (legacy)")
    p.add_argument(
        "--cities",
        nargs="+",
        default=None,
        help=f"Cities to benchmark (default: {FOCUS_CITIES})",
    )
    p.add_argument("--lr-size", type=int, default=512)
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    p.add_argument("--variants", nargs="+", default=["area", "s2_psf", "s2_psf_m", "psfm_2x"])
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true", help="Re-run even if metrics.json exists")
    args = p.parse_args()

    if args.cities:
        cities = list(args.cities)
    elif args.city:
        cities = [args.city]
    else:
        cities = list(FOCUS_CITIES)

    for name in args.variants:
        if name not in VARIANTS:
            raise SystemExit(f"unknown variant {name}; choose from {list(VARIANTS)}")

    align_path = ROOT / "eval" / "spatial_alignment.json"
    if not align_path.is_file():
        raise SystemExit(f"missing {align_path} — run audit_hr_lr_spatial_alignment.py first")

    for city in cities:
        _ensure_lr_variant(city, args.lr_size)

    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    pool: Queue = Queue()
    for d in devices:
        pool.put(d)

    run_jobs = []
    rows: list[dict] = []
    for city in cities:
        for name in args.variants:
            mpath = _metrics_path(city, name, args.lr_size)
            if mpath.exists() and not args.force and not args.dry_run:
                print(f"skip complete: {city}/{name} ({mpath})")
                rows.append(_row_from_metrics(city, name, args.lr_size, "skipped"))
                continue
            run_jobs.append(
                {
                    "city": city,
                    "variant": name,
                    "cfg": {**BASE, **VARIANTS[name]},
                    "lr_size": args.lr_size,
                    "iters": args.iters,
                }
            )

    print(
        f"PSF bench: {len(cities)} cities × {len(args.variants)} variants, "
        f"{len(run_jobs)} to run, {len(devices)} devices, spatial align ON",
        flush=True,
    )

    with ThreadPoolExecutor(max_workers=max(1, len(devices))) as ex:
        futs = [ex.submit(_run_one, j, pool, args.dry_run) for j in run_jobs]
        for fut in as_completed(futs):
            rows.append(fut.result())

    order_c = {c: i for i, c in enumerate(cities)}
    order_v = {n: i for i, n in enumerate(args.variants)}
    rows.sort(key=lambda r: (order_c.get(r.get("city", ""), 99), order_v.get(r["variant"], 99)))

    tag = "_".join(cities) if len(cities) <= 3 else f"{len(cities)}cities"
    out_dir = ROOT / "single_samples" / "sweep_results" / f"degA_lr{args.lr_size}_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    keys = [
        "city",
        "variant",
        "status",
        "lr_degradation",
        "max_res_mult",
        "psnr",
        "ssim",
        "lpips",
        "bil_psnr",
        "d_psnr",
        "d_ssim",
        "d_lpips",
        "spot_psnr",
        "spot_lpips",
        "exit",
    ]
    with (out_dir / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    by_var: dict[str, list[dict]] = {v: [] for v in args.variants}
    for r in rows:
        if r.get("psnr") is not None and r.get("status") in {"ok", "skipped"}:
            by_var[r["variant"]].append(r)
    mean_rows = []
    for v, rs in by_var.items():
        if not rs:
            continue
        mean_rows.append(
            {
                "variant": v,
                "n": len(rs),
                "psnr": sum(r["psnr"] for r in rs) / len(rs),
                "ssim": sum(r["ssim"] for r in rs) / len(rs),
                "lpips": sum(r["lpips"] for r in rs) / len(rs),
                "d_psnr": sum((r.get("d_psnr") or 0.0) for r in rs) / len(rs),
                "d_ssim": sum((r.get("d_ssim") or 0.0) for r in rs) / len(rs),
                "d_lpips": sum((r.get("d_lpips") or 0.0) for r in rs) / len(rs),
            }
        )
    (out_dir / "means.json").write_text(json.dumps(mean_rows, indent=2) + "\n")

    _print_table(rows)
    print(f"\nMeans over cities → {out_dir / 'means.json'}")
    print(f"{'variant':10s} {'n':>3} {'PSNR':>7} {'SSIM':>6} {'LPIPS':>6} {'ΔP':>6} {'ΔL':>6}")
    for r in mean_rows:
        print(
            f"{r['variant']:10s} {r['n']:3d} {r['psnr']:7.2f} {r['ssim']:6.3f} "
            f"{r['lpips']:6.3f} {r['d_psnr']:6.2f} {r['d_lpips']:6.3f}"
        )
    print(f"\nSummary → {out_dir}")


if __name__ == "__main__":
    main()
