#!/usr/bin/env python3
"""Retrain date-boundary tile pairs under controlled variants of the production recipe.

Every variant uses ``run_production._cmd`` (same seed, stack, schedule, identity
plan base date) and adds only its own flags, so seam differences between
variants isolate one change at a time. Outputs land under::

    single_samples/<parent>/sample/<prefix>_<variant>_<tile_id>/

Example
-------
    python scripts/run_boundary_variants.py --variants baseline clear clear_laplace --gpus 0 1 2 3
"""

from __future__ import annotations

import argparse
import json
import queue
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_production import _cmd  # noqa: E402

IDENTITY_DIR = ROOT / "production/national_2025/cross_identity/30466846596b_revised/32VNM"
DEFAULT_MANIFEST = IDENTITY_DIR / "manifest_icm_all.json"
DEFAULT_PLAN = IDENTITY_DIR / "identity_plan.json"
DEFAULT_PREFIX = "abx"
DEFAULT_PAIRS = [
    ((19, 7), (20, 7)),
    ((1, 21), (1, 22)),
    ((9, 18), (9, 19)),
    ((19, 13), (20, 13)),
    ((8, 19), (9, 19)),
    ((22, 5), (23, 5)),
]
VARIANT_FLAGS = {
    "baseline": ["--lr_stats_pixels", "all"],
    "clear": ["--lr_stats_pixels", "clear"],
    "clear_laplace": ["--lr_stats_pixels", "clear", "--use_laplace_nll"],
}


def tile_id(parent: str, xy: tuple[int, int]) -> str:
    return f"{parent}_t512_ovl12_y{xy[0]:02d}_x{xy[1]:02d}"


def swapped_dates(
    pairs, parent: str, assignment: dict[str, str]
) -> dict[tuple[int, int], str]:
    """Give each pair's tiles the other tile's base date (first pair wins on reuse)."""
    out: dict[tuple[int, int], str] = {}
    for a, b in pairs:
        da, db = assignment[tile_id(parent, a)], assignment[tile_id(parent, b)]
        out.setdefault(a, db)
        out.setdefault(b, da)
    return out


def build_jobs(args) -> list[tuple[str, dict, list[str]]]:
    manifest = json.loads(args.manifest.read_text())
    assignment = json.loads(args.identity_plan.read_text())["assignment"]
    parent = manifest["parent"]
    by_xy = {(int(t["iy"]), int(t["ix"])): t for t in manifest["tiles"]}
    xys = sorted({xy for pair in DEFAULT_PAIRS for xy in pair})
    swap = swapped_dates(DEFAULT_PAIRS, parent, assignment)

    jobs = []
    for variant in args.variants:
        base_variant, _, suffix = variant.partition("+")
        if base_variant not in VARIANT_FLAGS or suffix not in ("", "swap"):
            raise SystemExit(f"unknown variant {variant!r}")
        for xy in xys:
            tile = dict(by_xy[xy], parent=parent)
            tid = tile["tile_id"]
            tile["force_base_date"] = swap[xy] if suffix == "swap" else assignment[tid]
            run_prefix = f"{args.prefix}_{variant.replace('+', '_')}"
            metrics = ROOT / "single_samples" / parent / "sample" / f"{run_prefix}_{tid}" / "metrics.json"
            if metrics.is_file() and not args.overwrite:
                continue
            cmd = _cmd(
                tile,
                device=0,
                iters=args.iters,
                export_geotiff=True,
                run_prefix=run_prefix,
            )
            jobs.append((variant, tile, cmd + VARIANT_FLAGS[base_variant]))
    return jobs


def worker(gpu: int, jobs: queue.Queue, log_dir: Path, results: list, lock: threading.Lock) -> None:
    while True:
        try:
            variant, tile, cmd = jobs.get_nowait()
        except queue.Empty:
            return
        cmd = list(cmd)
        cmd[cmd.index("--device") + 1] = str(gpu)
        log = log_dir / f"{variant.replace('+', '_')}_{tile['tile_id']}.log"
        with open(log, "w") as fh:
            code = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode
        with lock:
            results.append({"variant": variant, "tile_id": tile["tile_id"], "returncode": code})
            print(f"[gpu{gpu}] {variant} {tile['tile_id']} exit={code}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--identity-plan", type=Path, default=DEFAULT_PLAN)
    ap.add_argument(
        "--variants",
        nargs="+",
        default=["baseline", "clear", "clear_laplace"],
        help=f"Any of {sorted(VARIANT_FLAGS)}, optionally suffixed '+swap'.",
    )
    ap.add_argument("--gpus", nargs="+", type=int, default=[0])
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--prefix", default=DEFAULT_PREFIX)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    jobs_list = build_jobs(args)
    print(f"{len(jobs_list)} jobs on GPUs {args.gpus}", flush=True)
    log_dir = ROOT / "logs" / "boundary_variants"
    log_dir.mkdir(parents=True, exist_ok=True)
    jobs: queue.Queue = queue.Queue()
    for job in jobs_list:
        jobs.put(job)
    results: list[dict] = []
    lock = threading.Lock()
    threads = [
        threading.Thread(target=worker, args=(gpu, jobs, log_dir, results, lock), daemon=True)
        for gpu in args.gpus
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    failed = [r for r in results if r["returncode"] != 0]
    print(f"done: {len(results) - len(failed)} ok, {len(failed)} failed", flush=True)
    if failed:
        for row in failed:
            print(f"  FAILED {row['variant']} {row['tile_id']}", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
