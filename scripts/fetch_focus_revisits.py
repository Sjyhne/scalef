#!/usr/bin/env python3
"""Batch-download S2 revisits for all AOIs in data/s2_revisits/aois.json."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--aois", type=Path, default=ROOT / "data" / "s2_revisits" / "aois.json")
    p.add_argument("--out-root", type=Path, default=ROOT / "data" / "s2_revisits")
    p.add_argument("--num-samples", type=int, default=None)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--cities", nargs="+", default=None, help="Subset of AOI ids")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-complete", action="store_true", default=True)
    p.add_argument("--force", action="store_true", help="Re-fetch even if meta.json has enough frames")
    p.add_argument(
        "--log",
        type=Path,
        default=None,
        help="Log file (default: <out-root>/fetch_focus.log)",
    )
    args = p.parse_args()

    aois = json.loads(args.aois.read_text())
    n = int(args.num_samples or aois.get("num_samples") or 16)
    days_before = int(aois.get("days_before") or 90)
    days_after = int(aois.get("days_after") or 90)
    cloud_method = str(aois.get("cloud_method") or "omnicloudmask")

    areas = aois["areas"]
    if args.cities:
        wanted = {c.lower() for c in args.cities}
        areas = [a for a in areas if a["id"].lower() in wanted]

    log_path = args.log if args.log is not None else args.out_root / "fetch_focus.log"
    args.out_root.mkdir(parents=True, exist_ok=True)
    print(f"Fetching {len(areas)} AOIs → {args.out_root}  (n={n}, device={args.device})")

    for area in areas:
        city = area["id"]
        out = args.out_root / city
        meta_path = out / "meta.json"
        if args.skip_complete and not args.force and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if len(meta.get("frames") or []) >= n:
                print(f"skip already complete: {city} ({len(meta['frames'])} frames)")
                continue

        bbox = area["bbox_wgs84"]
        area_days_before = int(area.get("days_before") or days_before)
        area_days_after = int(area.get("days_after") or days_after)
        cmd = [
            sys.executable,
            "-u",
            str(ROOT / "scripts" / "fetch_s2_revisits.py"),
            "--date",
            area["date"],
            "--bbox",
            *(str(v) for v in bbox),
            "--num-samples",
            str(n),
            "--days-before",
            str(area_days_before),
            "--days-after",
            str(area_days_after),
            "--cloud-method",
            cloud_method,
            "--device",
            args.device,
            "--out",
            str(out),
        ]
        if area.get("mgrs_tile"):
            cmd += ["--mgrs-tile", str(area["mgrs_tile"])]
        print(f"\n========== {city} {area['date']} ==========", flush=True)
        print(" ".join(cmd), flush=True)
        with log_path.open("a") as log:
            log.write(f"\n========== {city} {area['date']} ==========\n")
            log.write(" ".join(cmd) + "\n")
            log.flush()
            if args.dry_run:
                continue
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
            log.write(f"exit={proc.returncode} {city}\n")
            print(f"exit={proc.returncode} {city}", flush=True)
            if proc.returncode != 0:
                print(f"WARN: {city} failed; continuing", flush=True)

    print(f"\nDone. Log: {log_path}")


if __name__ == "__main__":
    main()
