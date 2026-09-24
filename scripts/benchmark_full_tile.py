#!/usr/bin/env python3
"""Estimate wall time to super-resolve a full Sentinel-2 MGRS tile.

Measures (or accepts) per-iteration training time at one or more LR patch
sizes, then extrapolates to a full ~10980×10980 LR granule processed as a
grid of independent per-patch optimizations (current production pattern).

Example
-------
python scripts/benchmark_full_tile.py --city asker --sizes 256 512 --iters 5000
python scripts/benchmark_full_tile.py --city asker --sizes 512 --iters 4000 --dry-run \\
    --ms-per-iter 227
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_TILE_LR = 10980


def _tile_lr_side(meta_path: Path) -> int:
    meta = json.loads(meta_path.read_text())
    return int(meta.get("width") or meta.get("height") or DEFAULT_TILE_LR)


def _profile_ms(city: str, side: int, *, profile_iters: int, warmup: int, log2: int, degradation: str) -> dict:
    s2_dir = ROOT / "data" / "s2_revisits" / f"{city}_lr{side}"
    if not s2_dir.is_dir():
        raise FileNotFoundError(f"missing {s2_dir} — run make_lr_size_variants.py first")

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "profile_train_step.py"),
        "--city",
        f"{city}_lr{side}",
        "--iters",
        str(profile_iters),
        "--warmup",
        str(warmup),
        "--log2",
        str(log2),
        "--degradation",
        degradation,
        "--mult",
        "1.0",
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        return {
            "lr_side": side,
            "status": "error",
            "error": (proc.stderr or proc.stdout)[-2000:],
            "wall_seconds": elapsed,
        }

    out = proc.stdout
    ms_per_iter = None
    hr_queries = None
    lr_hw = None
    hr_hw = None
    for line in out.splitlines():
        if "queries/iter" in line and ":" in line:
            # asker_lr512: LR 512x512, HR 2048x2048 (4,194,304 queries/iter)
            parts = line.split(":", 1)[1]
            if "LR " in parts:
                lr_bit = parts.split("LR ", 1)[1].split(",", 1)[0].strip()
                lr_hw = lr_bit
            if "HR " in parts:
                hr_bit = parts.split("HR ", 1)[1].split("(", 1)[0].strip()
                hr_hw = hr_bit
            if "(" in parts and "queries/iter" in parts:
                q = parts.split("(", 1)[1].split(" queries/iter", 1)[0].replace(",", "")
                hr_queries = int(q)
        if line.strip().startswith("TOTAL"):
            ms_per_iter = float(line.split()[-4])

    return {
        "lr_side": side,
        "status": "ok" if ms_per_iter is not None else "parse_error",
        "ms_per_iter": ms_per_iter,
        "hr_queries_per_iter": hr_queries,
        "lr_hw": lr_hw,
        "hr_hw": hr_hw,
        "wall_seconds": elapsed,
        "log2": log2,
        "degradation": degradation,
    }


def _patch_grid(tile_side: int, patch_side: int) -> dict:
    n = math.ceil(tile_side / patch_side)
    return {
        "tile_lr_side": tile_side,
        "patch_lr_side": patch_side,
        "patches_per_axis": n,
        "num_patches": n * n,
        "coverage_note": "non-overlapping grid; overlap/blend would increase patch count",
    }


def _estimate(ms_per_iter: float, *, num_patches: int, train_iters: int, num_gpus: int = 1) -> dict:
    sec_per_patch = train_iters * (ms_per_iter / 1000.0)
    sec_total = sec_per_patch * num_patches
    sec_parallel = sec_total / max(1, num_gpus)
    return {
        "ms_per_iter": ms_per_iter,
        "train_iters_per_patch": train_iters,
        "num_patches": num_patches,
        "sec_per_patch": sec_per_patch,
        "sec_total_sequential": sec_total,
        "hours_total_sequential": sec_total / 3600.0,
        "num_gpus": num_gpus,
        "sec_total_parallel": sec_parallel,
        "hours_total_parallel": sec_parallel / 3600.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--city", default="asker")
    ap.add_argument("--sizes", type=int, nargs="+", default=[256, 512])
    ap.add_argument("--patch-size", type=int, default=512, help="LR patch size for full-tile tiling plan")
    ap.add_argument("--iters", type=int, default=3000, help="Training iters assumed per patch")
    ap.add_argument("--gpus", type=int, default=1, help="GPUs for parallel patch jobs")
    ap.add_argument("--tile-side", type=int, default=0, help="Override MGRS LR side (default: read meta.json)")
    ap.add_argument("--profile-iters", type=int, default=15)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--log2", type=int, default=21)
    ap.add_argument("--degradation", default="s2_psf_m")
    ap.add_argument("--ms-per-iter", type=float, default=0.0, help="Skip profiling; use this ms/iter for --patch-size")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "single_samples" / "sweep_results" / "full_tile_benchmark.json",
    )
    args = ap.parse_args()

    meta_path = ROOT / "data" / "s2_revisits" / args.city / "meta.json"
    tile_side = args.tile_side or (_tile_lr_side(meta_path) if meta_path.is_file() else DEFAULT_TILE_LR)

    profiles: list[dict] = []
    if args.dry_run and args.ms_per_iter > 0:
        profiles.append({"lr_side": args.patch_size, "status": "manual", "ms_per_iter": args.ms_per_iter})
    else:
        for side in args.sizes:
            print(f"Profiling {args.city}_lr{side}...", flush=True)
            profiles.append(
                _profile_ms(
                    args.city,
                    side,
                    profile_iters=args.profile_iters,
                    warmup=args.warmup,
                    log2=args.log2,
                    degradation=args.degradation,
                )
            )
            p = profiles[-1]
            if p.get("status") == "ok":
                print(f"  lr{side}: {p['ms_per_iter']:.1f} ms/iter", flush=True)
            else:
                print(f"  lr{side}: {p.get('status')} — {str(p.get('error', ''))[:200]}", flush=True)

    ok = [p for p in profiles if p.get("ms_per_iter")]
    ref = next((p for p in ok if p["lr_side"] == args.patch_size), ok[-1] if ok else None)
    if ref is None:
        raise SystemExit("No successful profile runs; pass --ms-per-iter with --dry-run")

    grid = _patch_grid(tile_side, args.patch_size)
    train_est = _estimate(
        float(ref["ms_per_iter"]),
        num_patches=grid["num_patches"],
        train_iters=args.iters,
        num_gpus=args.gpus,
    )

    # Forward-only inference estimate (~27% of train time at lr512 from stage breakdown)
    forward_frac = 0.27
    infer_ms = float(ref["ms_per_iter"]) * forward_frac
    infer_est = _estimate(infer_ms, num_patches=grid["num_patches"], train_iters=1, num_gpus=args.gpus)

    # Quadratic extrapolation to monolithic full tile (theoretical lower bound; usually OOM)
    if ref.get("hr_queries_per_iter"):
        tile_hr_side = tile_side * 4  # df=4
        ref_hr_side = int(ref["lr_side"]) * 4
        pixel_ratio = (tile_hr_side / ref_hr_side) ** 2
        mono_ms = float(ref["ms_per_iter"]) * pixel_ratio
    else:
        pixel_ratio = (tile_side / ref["lr_side"]) ** 2
        mono_ms = float(ref["ms_per_iter"]) * pixel_ratio

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "city": args.city,
        "config": {
            "patch_lr_side": args.patch_size,
            "train_iters_per_patch": args.iters,
            "log2_hashmap_size": args.log2,
            "lr_degradation": args.degradation,
            "gpus": args.gpus,
        },
        "tile": grid,
        "profiles": profiles,
        "reference_profile": ref,
        "training_estimate": train_est,
        "inference_forward_estimate": {
            "ms_per_patch": infer_ms,
            "sec_total_sequential": infer_est["sec_total_sequential"],
            "note": "single HR forward per patch after training; no backward",
        },
        "monolithic_extrapolation": {
            "hr_pixel_ratio_vs_reference": pixel_ratio,
            "ms_per_iter_theoretical": mono_ms,
            "hours_for_iters": (mono_ms * args.iters) / 1000 / 3600,
            "note": "single optimization on full tile; impractical — OOM on 80GB GPU already at lr1024 backward",
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")

    print("\n=== Full S2 tile processing estimate ===")
    print(f"Tile LR: {tile_side}×{tile_side}  |  Patch: {args.patch_size}×{args.patch_size}")
    print(f"Patches: {grid['patches_per_axis']}×{grid['patches_per_axis']} = {grid['num_patches']}")
    print(f"Reference: lr{ref['lr_side']} @ {ref['ms_per_iter']:.1f} ms/iter")
    print(
        f"Train {args.iters} iters/patch: "
        f"{train_est['hours_total_sequential']:.1f} h sequential, "
        f"{train_est['hours_total_parallel']:.1f} h on {args.gpus} GPU(s)"
    )
    print(
        f"Inference (1 forward/patch): {infer_est['sec_total_sequential']:.0f} s sequential "
        f"({infer_est['sec_total_parallel']:.0f} s on {args.gpus} GPUs)"
    )
    print(
        f"Monolithic extrapolation: {mono_ms/1000:.0f} s/iter "
        f"({report['monolithic_extrapolation']['hours_for_iters']:.0f} h for {args.iters} iters) — not feasible"
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
