#!/usr/bin/env python3
"""Within-frame K tiles vs the same K drawn from random frames.

The control is the current K=4 recipe: 128-pixel windows, four per step, all
from the frame the loader just yielded. ``cross_iid_k4`` keeps that fuse size
and draws each window from an independent frame, with that frame's affine and
color. Stopping, loss, and halo match the confirmatory K=4 fits.
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

# name -> CLI extras (beyond matched holdout / charbonnier recipe)
VARIANTS = {
    "full": [],
    "within_k4": ["--lr_tile", "128", "--lr_tiles_per_step", "4", "--lr_tile_mix", "within"],
    "within_k2": ["--lr_tile", "128", "--lr_tiles_per_step", "2", "--lr_tile_mix", "within"],
    "cross_epoch_k2": [
        "--lr_tile",
        "128",
        "--lr_tiles_per_step",
        "2",
        "--lr_tile_mix",
        "cross_epoch",
    ],
    "cross_iid_k2": [
        "--lr_tile",
        "128",
        "--lr_tiles_per_step",
        "2",
        "--lr_tile_mix",
        "cross_iid",
    ],
    "cross_epoch_k4": [
        "--lr_tile",
        "128",
        "--lr_tiles_per_step",
        "4",
        "--lr_tile_mix",
        "cross_epoch",
    ],
    "cross_iid_k4": [
        "--lr_tile",
        "128",
        "--lr_tiles_per_step",
        "4",
        "--lr_tile_mix",
        "cross_iid",
    ],
    "cross_same_tile_k2": [
        "--lr_tile",
        "128",
        "--lr_tiles_per_step",
        "2",
        "--lr_tile_mix",
        "cross_same_tile",
    ],
    "cross_same_tile_k4": [
        "--lr_tile",
        "128",
        "--lr_tiles_per_step",
        "4",
        "--lr_tile_mix",
        "cross_same_tile",
    ],
    "cross_same_tile_k8": [
        "--lr_tile",
        "128",
        "--lr_tiles_per_step",
        "8",
        "--lr_tile_mix",
        "cross_same_tile",
    ],
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
        "8",
        "--early_stop_min_iters",
        "1000",
        "--eval_every",
        "200",
        "--iters",
        str(iters),
        "--device",
        str(device),
        "--seed",
        "6",
        "--spatial_holdout",
        "0.1",
        "--holdout_block",
        "0",
        "--hr_render_tile",
        "2048",
        "--lr_stats_pixels",
        "all",
        "--lr_tile_halo",
        "-1",
        "--lr_degradation",
        "s2_psf_m",
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
        "lr_tile": m.get("lr_tile"),
        "lr_tiles_per_step": m.get("lr_tiles_per_step"),
        "lr_tile_mix": m.get("lr_tile_mix"),
        "metrics_path": str(mpath.relative_to(ROOT)),
    }


def _run_variant(
    city: str, variant: str, device: int, iters: int, skip_existing: bool
) -> dict:
    mpath = _metrics_path(city, variant, iters)
    # Reuse prior within-k2 / full metrics when present under old names.
    aliases = {
        "full": _metrics_path(city, "full", iters),
        "within_k2": _metrics_path(city, "k2", iters),
    }
    if skip_existing and mpath.is_file():
        return _summarize(city, variant, mpath, skipped=True)
    if skip_existing and variant in aliases and aliases[variant].is_file():
        return _summarize(city, variant, aliases[variant], skipped=True)
    print(f"[gpu{device}] {city}/{variant} ...", flush=True)
    subprocess.run(_base_cmd(city, variant, device, iters), cwd=ROOT, check=True)
    return _summarize(city, variant, mpath, skipped=False)


PAPER7 = ("asker", "bergen", "rana", "tromso", "amli", "vennesla", "trondheim")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--city", default="asker")
    p.add_argument(
        "--cities",
        nargs="+",
        default=None,
        help="Sites to run. 'paper7' expands to the ablation panel.",
    )
    p.add_argument("--iters", type=int, default=5000)
    p.add_argument("--gpus", type=int, default=6)
    p.add_argument("--gpu-offset", type=int, default=0)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument(
        "--variants",
        nargs="+",
        default=list(VARIANTS.keys()),
        choices=list(VARIANTS.keys()),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "single_samples" / "sweep_results" / "bench_cross_frame_tiles.json",
    )
    args = p.parse_args()
    if args.cities:
        cities: list[str] = []
        for name in args.cities:
            cities.extend(PAPER7 if name == "paper7" else (name,))
    else:
        cities = [args.city]

    jobs = []
    n = 0
    for city in cities:
        for variant in args.variants:
            device = args.gpu_offset + (n % args.gpus)
            jobs.append((city, variant, device))
            n += 1

    rows = []
    with ThreadPoolExecutor(max_workers=args.gpus) as ex:
        futs = {
            ex.submit(_run_variant, city, variant, device, args.iters, args.skip_existing): (
                city,
                variant,
            )
            for city, variant, device in jobs
        }
        for fut in as_completed(futs):
            city, variant = futs[fut]
            try:
                row = fut.result()
                rows.append(row)
                print(
                    f"DONE {city}/{variant} lpips={row['audit_lpips']:.4f} "
                    f"time={row['training_time_s']:.1f}s",
                    flush=True,
                )
            except Exception as e:
                print(f"FAIL {city}/{variant}: {e}", flush=True)
                raise

    rows.sort(key=lambda r: (r["city"], r["variant"]))
    by_city: dict[str, dict[str, dict]] = {}
    for r in rows:
        by_city.setdefault(r["city"], {})[r["variant"]] = r

    def _delta(rows_for_city: dict[str, dict], base: str) -> list[dict]:
        ref = rows_for_city.get(base)
        if ref is None:
            return []
        out_rows = []
        for name, r in rows_for_city.items():
            if name == base:
                continue
            out_rows.append(
                {
                    "city": r["city"],
                    "variant": name,
                    "lpips": r["audit_lpips"],
                    "lpips_delta": float(r["audit_lpips"]) - float(ref["audit_lpips"]),
                    "time_s": r["training_time_s"],
                    "time_ratio": (
                        float(r["training_time_s"]) / float(ref["training_time_s"])
                        if ref["training_time_s"]
                        else None
                    ),
                    "peak_gb": r["peak_memory_gb"],
                    "best_iter": r["best_iter"],
                    "base_lpips": ref["audit_lpips"],
                    "base_time_s": ref["training_time_s"],
                }
            )
        return out_rows

    vs_within = []
    vs_full = []
    for city_rows in by_city.values():
        vs_within.extend(_delta(city_rows, "within_k4"))
        vs_full.extend(_delta(city_rows, "full"))

    out = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cities": cities,
        "iters_budget": args.iters,
        "recipe": "confirmatory K=4 flags; only lr_tile_mix changes",
        "rows": rows,
        "vs_within_k4": vs_within,
        "vs_full": vs_full,
    }
    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out.get("vs_within_k4") or out.get("vs_full"), indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
