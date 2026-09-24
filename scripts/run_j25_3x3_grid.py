#!/usr/bin/env python3
"""Fetch the July 2025 3x3 MGRS grid (aois_j25_3x3.json), then tile+train+mosaic."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--aois",
        type=Path,
        default=ROOT / "data" / "s2_revisits" / "aois_j25_3x3.json",
    )
    ap.add_argument("--out-root", type=Path, default=ROOT / "data" / "s2_revisits")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--size-km", type=float, default=100.0)
    ap.add_argument("--num-samples", type=int, default=None)
    ap.add_argument("--skip-fetch", action="store_true")
    ap.add_argument("--skip-batch", action="store_true")
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--gpu-offset", type=int, default=0)
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--log",
        type=Path,
        default=ROOT / "single_samples" / "sweep_results" / "j25_3x3_fetch.log",
    )
    ap.add_argument(
        "--batch-out",
        type=Path,
        default=ROOT / "single_samples" / "sweep_results" / "granule_batch_j25_3x3.json",
    )
    args = ap.parse_args()

    cfg = json.loads(args.aois.read_text())
    n = int(args.num_samples or cfg.get("num_samples") or 16)
    days_before = int(cfg.get("days_before") or 45)
    days_after = int(cfg.get("days_after") or 45)
    cloud_method = str(cfg.get("cloud_method") or "omnicloudmask")
    areas = cfg["areas"]
    parents = [a["id"] for a in areas]

    args.log.parent.mkdir(parents=True, exist_ok=True)

    if not args.skip_fetch:
        print(f"Fetching {len(areas)} MGRS tiles → {args.out_root}", flush=True)
        for area in areas:
            city = area["id"]
            out = args.out_root / city
            meta_path = out / "meta.json"
            if meta_path.is_file():
                meta = json.loads(meta_path.read_text())
                if len(meta.get("frames") or []) >= n:
                    print(f"skip complete: {city} ({len(meta['frames'])} frames)", flush=True)
                    continue

            cmd = [
                sys.executable,
                "-u",
                str(ROOT / "scripts" / "fetch_s2_revisits.py"),
                "--date",
                area["date"],
                "--lon",
                str(area["lon"]),
                "--lat",
                str(area["lat"]),
                "--size-km",
                str(args.size_km),
                "--mgrs-tile",
                str(area["mgrs_tile"]),
                "--num-samples",
                str(n),
                "--days-before",
                str(days_before),
                "--days-after",
                str(days_after),
                "--cloud-method",
                cloud_method,
                "--device",
                args.device,
                "--out",
                str(out),
            ]
            print(f"\n========== {city} ==========", flush=True)
            print(" ".join(cmd), flush=True)
            with args.log.open("a") as log:
                log.write(f"\n========== {city} ==========\n")
                log.write(" ".join(cmd) + "\n")
                log.flush()
                if args.dry_run:
                    continue
                proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
                log.write(f"exit={proc.returncode} {city}\n")
                print(f"exit={proc.returncode} {city}", flush=True)
                if proc.returncode != 0:
                    raise SystemExit(f"fetch failed: {city} (see {args.log})")

    if args.skip_batch or args.dry_run:
        print("skip batch", flush=True)
        return

    # Require all parents present before training.
    missing = [p for p in parents if not (args.out_root / p / "meta.json").is_file()]
    if missing:
        raise SystemExit(f"missing meta for: {missing}")

    batch_cmd = [
        sys.executable,
        "-u",
        str(ROOT / "scripts" / "run_granule_batch.py"),
        "--parents",
        *parents,
        "--gpus",
        str(args.gpus),
        "--gpu-offset",
        str(args.gpu_offset),
        "--iters",
        str(args.iters),
        "--skip-existing",
        "--out",
        str(args.batch_out),
    ]
    print("\n========== granule batch ==========", flush=True)
    print(" ".join(batch_cmd), flush=True)
    subprocess.run(batch_cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
