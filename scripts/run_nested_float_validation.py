#!/usr/bin/env python3
"""Retrain a nested-ladder validation subset with float GeoTIFF outputs.

One LR512 parent per NIB project (lowest parent tile id, fixed before scoring)
is refitted at LR64/128/256/512 with the lr512align_v2 recipe plus the partial-
update context halo, exporting float qgis/{sr_pred,hr_gt,s2_bilinear}.tif.
scripts/score_nested_float_validation.py then scores the shared LR64 windows
from those floats and from the display PNGs of the same runs.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.bench_complete_patches import _cmd, _metrics_path  # noqa: E402
from scripts.run_complete_patch_size_ladder import NEST_ROOT, _train_knobs  # noqa: E402

RUN_TAG = "floatval_v3"
SIDES = (512, 256, 128, 64)
EXTRA_FLAGS = ["--lr_stats_pixels", "all", "--lr_tile_halo", "-1"]
MANIFEST = ROOT / "single_samples" / "sweep_results" / f"nested_{RUN_TAG}_manifest.json"
FULL_MANIFEST = ROOT / "single_samples" / "sweep_results" / f"nested_{RUN_TAG}_full_manifest.json"
LOG_DIR = ROOT / "logs" / f"nested_{RUN_TAG}"


def select_parents(all_parents: bool = False) -> list[str]:
    tiles = json.loads((NEST_ROOT / "nested_lr512_manifest.json").read_text())["tiles"]
    if all_parents:
        return sorted({t["parent_tile_id"] for t in tiles})
    by_project: dict[str, list[str]] = {}
    for t in tiles:
        by_project.setdefault(t["project_folder"], []).append(t["parent_tile_id"])
    return sorted(min(v) for v in by_project.values())


def jobs(parents: list[str]) -> list[dict]:
    out = []
    for side in SIDES:
        knobs = _train_knobs(side, RUN_TAG)
        tiles = json.loads((NEST_ROOT / f"nested_lr{side}_manifest.json").read_text())["tiles"]
        for t in tiles:
            if t["parent_tile_id"] not in parents:
                continue
            cmd = _cmd(t, 0, 5000, export_geotiff=True, lr_tile=knobs["lr_tile"],
                       lr_tiles_per_step=knobs["lr_tiles_per_step"], run_prefix=knobs["run_prefix"])
            out.append({"side": side, "tile_id": t["tile_id"], "parent_tile_id": t["parent_tile_id"],
                        "command": [*cmd, *EXTRA_FLAGS],
                        "metrics": str(_metrics_path(t["parent_city"], t["tile_id"], knobs["run_prefix"]))})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3, 5, 6])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--all-parents", action="store_true",
                    help="refit every nested parent; existing floatval_v3 fits are reused")
    args = ap.parse_args()
    parents = select_parents(args.all_parents)
    manifest = FULL_MANIFEST if args.all_parents else MANIFEST
    rule = ("all nested LR512 parents" if args.all_parents
            else "lowest parent_tile_id per NIB project (fixed before scoring)")
    todo = jobs(parents)
    status: dict[str, str] = {}
    lock = threading.Lock()

    def write() -> None:
        manifest.write_text(json.dumps({
            "run_tag": RUN_TAG, "updated_utc": datetime.now(timezone.utc).isoformat(),
            "selection_rule": rule,
            "parents": parents, "extra_flags_vs_lr512align_v2": EXTRA_FLAGS,
            "jobs": [{**j, "status": status.get(j["tile_id"], "planned")} for j in todo],
        }, indent=1))

    manifest.parent.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    write()
    counts = {s: sum(j["side"] == s for j in todo) for s in SIDES}
    print(f"{len(parents)} parents, {len(todo)} jobs {counts} -> {manifest}")
    if args.dry_run:
        print(" ".join(todo[0]["command"]))
        return
    q: queue.Queue = queue.Queue()
    for j in todo:
        q.put(j)

    def worker(gpu: int) -> None:
        while True:
            try:
                j = q.get_nowait()
            except queue.Empty:
                return
            if Path(j["metrics"]).is_file() and (Path(j["metrics"]).parent / "qgis" / "sr_pred.tif").is_file():
                result = "skipped_existing"
            else:
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
                with open(LOG_DIR / f"{j['tile_id']}_lr{j['side']}.log", "w") as log:
                    rc = subprocess.run(j["command"], cwd=ROOT, env=env, stdout=log,
                                        stderr=subprocess.STDOUT).returncode
                result = "completed" if rc == 0 and Path(j["metrics"]).is_file() else f"failed_rc{rc}"
            with lock:
                status[j["tile_id"]] = result
                if len(status) % 25 == 0 or result.startswith("failed"):
                    write()
                    print(f"[{len(status)}/{len(todo)}] {j['tile_id']}: {result}", flush=True)

    threads = [threading.Thread(target=worker, args=(g,), daemon=True) for g in args.gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    write()
    failed = [k for k, v in status.items() if v.startswith("failed")]
    print(f"done; {len(failed)} failed")


if __name__ == "__main__":
    main()
