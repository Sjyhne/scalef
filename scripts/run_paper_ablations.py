#!/usr/bin/env python3
"""Paper ablation campaign: quality + scalability on frozen NIB/MISR cities.

See ``docs/HASHGRID_SUPERF.md`` §8.2.

Primary paper claim is **speed/scalability** (hash vs Fourier, fused-k Pareto,
AOI-size ladder). Every row logs LPIPS/PSNR/SSIM **and** wall time + VRAM, plus
extrapolated hours/MGRS (441 LR512 AOIs).

Example
-------
    python scripts/run_paper_ablations.py --gpus 8
    python scripts/run_paper_ablations.py --cities asker --variants B0 B1 S_aoi256 S_aoi1024
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

DEFAULT_CITIES = ["asker", "bergen", "rana", "tromso", "amli", "vennesla", "trondheim"]

# ~21×21 non-overlapping LR512 windows on a 10980² MGRS granule (see PRODUCTION.md).
AOIS_PER_MGRS_LR512 = 441

# variant_id -> extra optimize.py flags (production base always applied)
VARIANTS: dict[str, list[str]] = {
    "B0": [],  # ship recipe (hash + k4)
    "B1": ["--input_projection", "fourier", "--fourier_scale", "10"],  # harsh @512
    "B2_k1": ["--lr_tiles_per_step", "1", "--lr_tile", "512"],
    "B2_k2": ["--lr_tiles_per_step", "2", "--lr_tile", "256"],
    "B2_full": ["--lr_tiles_per_step", "1", "--lr_tile", "0"],  # 0 = full field
    "B3_mae": ["--recon_loss", "mae"],
    "B4_area": ["--lr_degradation", "area"],  # no MTF (area downsample)
    # Hash AOI-size ladder (Asker).
    "S_aoi256": [],
    "S_aoi512": [],
    "S_aoi1024": [],
    "S_aoi2048": [],
}

AOI_SIZE_BY_VARIANT: dict[str, int] = {
    "S_aoi256": 256,
    "S_aoi512": 512,
    "S_aoi1024": 1024,
    "S_aoi2048": 2048,
}

# Fair Fourier track: small AOI + scale (Asker). Full-field (no fused-k) so 64/128 aren't
# over-tiled. Hash controls H_aoi* match window size for a fair encoder compare.
for _aoi in (64, 128, 256, 512):
    VARIANTS[f"H_aoi{_aoi}"] = [
        "--input_projection",
        "hashgrid_tcnn",
        "--lr_tile",
        "0",
        "--lr_tiles_per_step",
        "1",
    ]
    AOI_SIZE_BY_VARIANT[f"H_aoi{_aoi}"] = _aoi
    # Scale 1 is too low; >10 rarely useful — keep 2 / 5 / 10 for fair Fourier.
    for _scale in (2, 5, 10):
        vid = f"F_aoi{_aoi}_s{_scale}"
        VARIANTS[vid] = [
            "--input_projection",
            "fourier",
            "--fourier_scale",
            str(_scale),
            "--lr_tile",
            "0",
            "--lr_tiles_per_step",
            "1",
        ]
        AOI_SIZE_BY_VARIANT[vid] = _aoi

# Legacy scale ids from the first fair sweep (still loadable for old metrics).
for _aoi in (64, 128, 256, 512):
    for _scale in (1, 20, 40):
        vid = f"F_aoi{_aoi}_s{_scale}"
        if vid in VARIANTS:
            continue
        VARIANTS[vid] = [
            "--input_projection",
            "fourier",
            "--fourier_scale",
            str(_scale),
            "--lr_tile",
            "0",
            "--lr_tiles_per_step",
            "1",
        ]
        AOI_SIZE_BY_VARIANT[vid] = _aoi

# Backward-compatible alias if old jobs referenced B4_nopsf
VARIANTS["B4_nopsf"] = list(VARIANTS["B4_area"])

def _s2_dir(city: str, variant: str = "B0") -> Path:
    aoi = AOI_SIZE_BY_VARIANT.get(variant)
    if aoi is not None:
        path = ROOT / "data" / "s2_revisits" / f"{city}_lr{aoi}"
        if path.is_dir():
            return path
        raise FileNotFoundError(f"{city}_lr{aoi}")
    lr512 = ROOT / "data" / "s2_revisits" / f"{city}_lr512"
    if lr512.is_dir():
        return lr512
    parent = ROOT / "data" / "s2_revisits" / city
    if parent.is_dir():
        return parent
    raise FileNotFoundError(city)


def _run_name(city: str, variant: str) -> str:
    return f"paper_{variant}_{city}"


def _metrics_path(city: str, variant: str) -> Path:
    return ROOT / "single_samples" / city / "sample" / _run_name(city, variant) / "metrics.json"


def _aois_per_mgrs(variant: str) -> int:
    """Rough AOI count on a ~10980² granule for extrapolation."""
    aoi = AOI_SIZE_BY_VARIANT.get(variant, 512)
    # Non-overlapping tiles that fit in 10980 (floor).
    n = max(1, 10980 // aoi)
    return n * n


def _build_cmd(city: str, variant: str, device: int, iters: int) -> list[str]:
    """Build argv; apply variant overrides by rewriting known flags."""
    base = {
        "--lr_degradation": "s2_psf_m",
        "--recon_loss": "charbonnier",
        "--charbonnier_eps": "0.01",
        "--lr_tile": "128",
        "--lr_tiles_per_step": "4",
        "--lr_tile_mix": "within",
        "--input_projection": "hashgrid_tcnn",
    }
    # Larger AOIs: keep ~same micro-batch footprint as METHOD (tile grows with field).
    # Small AOIs / F_* / H_*: variants set full-field via VARIANTS flags.
    aoi = AOI_SIZE_BY_VARIANT.get(variant)
    if aoi == 1024 and not variant.startswith(("F_", "H_")):
        base["--lr_tile"] = "256"
        base["--lr_tiles_per_step"] = "4"
    elif aoi == 2048 and not variant.startswith(("F_", "H_")):
        base["--lr_tile"] = "512"
        base["--lr_tiles_per_step"] = "2"
    elif aoi is not None and aoi <= 128 and not variant.startswith(("F_", "H_")):
        # S_aoi64/128 if added later: full field
        base["--lr_tile"] = "0"
        base["--lr_tiles_per_step"] = "1"

    extra = list(VARIANTS[variant])
    i = 0
    while i < len(extra):
        if extra[i].startswith("--") and i + 1 < len(extra) and not extra[i + 1].startswith("--"):
            base[extra[i]] = extra[i + 1]
            i += 2
        else:
            i += 1

    s2 = _s2_dir(city, variant)
    cmd = [
        sys.executable,
        str(ROOT / "optimize.py"),
        "--dataset",
        city,
        "--s2-dir",
        str(s2),
        "--run_name",
        _run_name(city, variant),
        "--device",
        str(device),
        "--iters",
        str(iters),
        "--early_stop_metric",
        "holdout_mse",
        "--early_stop_patience",
        "8",
        "--early_stop_min_iters",
        "1000",
        "--eval_every",
        "200",
        "--spatial_holdout",
        "0.1",
        "--force_hr_eval",
        "--spatial_alignment_path",
        str(ROOT / "eval" / "spatial_alignment.json"),
        "--no_qgis_export",
    ]
    for k, v in base.items():
        if v is None:
            continue
        cmd += [k, str(v)]
    return cmd


def _summarize(city: str, variant: str, mpath: Path, *, skipped: bool, error: str | None = None) -> dict:
    row: dict = {
        "city": city,
        "variant": variant,
        "skipped": skipped,
        "error": error,
        "metrics_path": str(mpath.relative_to(ROOT)) if mpath.is_file() else None,
        "aois_per_mgrs_assumed": _aois_per_mgrs(variant),
    }
    if not mpath.is_file():
        return row
    m = json.loads(mpath.read_text())
    es = m.get("early_stop") or {}
    lp = m.get("lpips") or {}
    ps = m.get("psnr") or {}
    ss = m.get("ssim") or {}
    t_s = m.get("training_time_seconds")
    n_aoi = _aois_per_mgrs(variant)
    hours_mgrs = None
    if isinstance(t_s, (int, float)) and t_s > 0:
        hours_mgrs = round(float(t_s) * n_aoi / 3600.0, 3)
    row.update(
        {
            "lpips": lp.get("model"),
            "lpips_bilinear": lp.get("bilinear"),
            "psnr": ps.get("model"),
            "ssim": ss.get("model"),
            "completed_iters": m.get("completed_iters"),
            "best_iter": es.get("best_val_iter"),
            "stopped_iter": es.get("stopped_iter"),
            "training_time_s": t_s,
            "peak_gpu_mem_gb": m.get("peak_training_gpu_mem_gb") or m.get("peak_gpu_memory_gb"),
            "hours_per_mgrs_extrapolated": hours_mgrs,
        }
    )
    return row


def _speed_table(results: list[dict]) -> list[dict]:
    """Compact Pareto-friendly rows for the paper speed story."""
    rows = []
    for r in results:
        if r.get("error") or r.get("training_time_s") is None:
            continue
        rows.append(
            {
                "city": r["city"],
                "variant": r["variant"],
                "lpips": r.get("lpips"),
                "psnr": r.get("psnr"),
                "training_time_s": r.get("training_time_s"),
                "peak_gpu_mem_gb": r.get("peak_gpu_mem_gb"),
                "hours_per_mgrs_extrapolated": r.get("hours_per_mgrs_extrapolated"),
                "aois_per_mgrs_assumed": r.get("aois_per_mgrs_assumed"),
            }
        )
    return rows


def main() -> None:
    default_variants = [
        "B0",
        "B1",
        "B2_k1",
        "B2_k2",
        "B2_full",
        "B3_mae",
        "B4_area",
        "S_aoi256",
        "S_aoi1024",
        "S_aoi2048",
    ]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cities", nargs="+", default=DEFAULT_CITIES)
    ap.add_argument(
        "--variants",
        nargs="+",
        default=default_variants,
        choices=sorted(VARIANTS.keys()),
    )
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--gpu-offset", type=int, default=0)
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--no-skip-existing", action="store_false", dest="skip_existing")
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "single_samples" / "sweep_results" / "paper_ablations.json",
    )
    args = ap.parse_args()

    jobs: list[tuple[str, str]] = []
    for city in args.cities:
        for variant in args.variants:
            try:
                _s2_dir(city, variant)
            except FileNotFoundError:
                # AOI ladder only exists for Asker today — skip quietly.
                if variant in AOI_SIZE_BY_VARIANT:
                    print(f"SKIP no AOI stack: {city} {variant}", flush=True)
                    continue
                print(f"SKIP missing data: {city}", flush=True)
                break
            jobs.append((city, variant))

    results: list[dict] = []
    lock = threading.Lock()
    q: queue.Queue[tuple[str, str] | None] = queue.Queue()
    for job in jobs:
        q.put(job)
    for _ in range(args.gpus):
        q.put(None)

    def worker(gpu_local: int) -> None:
        device = args.gpu_offset + gpu_local
        while True:
            item = q.get()
            if item is None:
                return
            city, variant = item
            mpath = _metrics_path(city, variant)
            if args.skip_existing and mpath.is_file():
                row = _summarize(city, variant, mpath, skipped=True)
                with lock:
                    results.append(row)
                    print(f"[gpu{device}] skip {city} {variant}", flush=True)
                continue
            cmd = _build_cmd(city, variant, device, args.iters)
            print(f"[gpu{device}] {city} {variant} ...", flush=True)
            time.sleep(1.0)
            try:
                subprocess.run(cmd, cwd=ROOT, check=True)
                row = _summarize(city, variant, mpath, skipped=False)
            except Exception as exc:  # noqa: BLE001
                row = _summarize(city, variant, mpath, skipped=False, error=str(exc))
                print(f"[gpu{device}] FAIL {city} {variant}: {exc}", flush=True)
            with lock:
                results.append(row)

    threads = [threading.Thread(target=worker, args=(g,)) for g in range(args.gpus)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    sorted_results = sorted(results, key=lambda r: (r.get("city") or "", r.get("variant") or ""))
    out = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cities": list(args.cities),
        "variants": list(args.variants),
        "iters": args.iters,
        "aois_per_mgrs_lr512_reference": AOIS_PER_MGRS_LR512,
        "note": "Primary claim = scalability/speed; see speed_table + hours_per_mgrs_extrapolated.",
        "results": sorted_results,
        "speed_table": _speed_table(sorted_results),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"Wrote {args.out} ({len(sorted_results)} rows, {len(out['speed_table'])} with timing)", flush=True)


if __name__ == "__main__":
    main()
