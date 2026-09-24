#!/usr/bin/env python3
"""Train the 1 + 4 + 16 AOI grid that tiles asker LR2048, then dump affines.

Longest job first so the 2048 arm occupies a GPU while the 512s drain.
Full coverage and patience 8 so the warp is not an early-stop artefact.
"""

from __future__ import annotations

import argparse
import json
import queue
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (run_name, s2_dir, extra)
JOBS: list[tuple[str, str, list[str]]] = [
    ("aff_g2048", "asker_g2048",
     ["--lr_tile", "512", "--lr_tiles_per_step", "0", "--grad_accum", "8",
      "--hash_log2_hashmap_size", "22"]),
]
for iy in range(2):
    for ix in range(2):
        JOBS.append((
            f"aff_g1024_y{iy}_x{ix}", f"asker_g1024_y{iy}_x{ix}",
            ["--lr_tile", "512", "--lr_tiles_per_step", "0", "--grad_accum", "2"],
        ))
for iy in range(4):
    for ix in range(4):
        JOBS.append((f"aff_g512_y{iy}_x{ix}", f"asker_g512_y{iy}_x{ix}", []))


def _cmd(run_name: str, s2_dir: str, extra: list[str], device: int, iters: int) -> list[str]:
    return [
        sys.executable, str(ROOT / "optimize.py"),
        "--dataset", "asker",
        "--s2-dir", str(ROOT / "data" / "s2_revisits" / s2_dir),
        "--run_name", run_name,
        "--recon_loss", "charbonnier", "--charbonnier_eps", "0.01",
        "--early_stop_metric", "holdout_mse",
        "--early_stop_patience", "8", "--early_stop_min_iters", "1000",
        "--iters", str(iters),
        "--device", str(device),
        "--spatial_holdout", "0.1", "--holdout_block", "0",
        "--eval_every", "200", "--hr_render_tile", "2048",
        "--lr_tile_mix", "within",
        "--spatial_alignment_path", str(ROOT / "eval" / "spatial_alignment.json"),
        "--no_qgis_export",
        *extra,
    ]


def _summarize(run_name: str) -> dict:
    rundir = ROOT / "single_samples" / "asker" / "sample" / run_name
    mpath = rundir / "metrics.json"
    apath = rundir / "affines.json"
    if not mpath.is_file():
        return {"run": run_name, "status": "missing"}
    m = json.loads(mpath.read_text())
    out = {
        "run": run_name,
        "lpips": m["lpips"]["model"],
        "lpips_vs_bilinear": m["lpips"]["improvement"],
        "completed_iters": m.get("completed_iters"),
        "training_time_s": round(m.get("training_time_seconds") or 0.0, 1),
        "affines": apath.is_file(),
    }
    if apath.is_file():
        dump = json.loads(apath.read_text())
        out["row0"] = dump.get("row0")
        out["col0"] = dump.get("col0")
        out["lr_width"] = dump.get("lr_width")
    return out


def _worker(gpu: int, work: queue.Queue, results: list, lock: threading.Lock,
            iters: int, skip_existing: bool) -> None:
    while True:
        try:
            run_name, s2_dir, extra = work.get_nowait()
        except queue.Empty:
            return
        try:
            mpath = ROOT / "single_samples" / "asker" / "sample" / run_name / "metrics.json"
            if skip_existing and mpath.is_file():
                print(f"[gpu{gpu}] skip {run_name}", flush=True)
            else:
                print(f"[gpu{gpu}] {run_name} ...", flush=True)
                log = ROOT / "single_samples" / "sweep_results" / f"{run_name}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                with log.open("w") as fh:
                    subprocess.run(
                        _cmd(run_name, s2_dir, extra, gpu, iters), cwd=ROOT, check=True,
                        stdout=fh, stderr=subprocess.STDOUT,
                    )
            row = _summarize(run_name)
            with lock:
                results.append(row)
            print(f"DONE {run_name}: lpips={row.get('lpips')} affines={row.get('affines')}", flush=True)
        except Exception as exc:  # noqa: BLE001
            with lock:
                results.append({"run": run_name, "error": str(exc)})
            print(f"FAIL {run_name}: {exc}", flush=True)
        finally:
            work.task_done()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iters", type=int, default=8000)
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--gpu-offset", type=int, default=0)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument(
        "--out", type=Path,
        default=ROOT / "single_samples" / "sweep_results" / "bench_affine_grid.json",
    )
    args = ap.parse_args()

    work: queue.Queue = queue.Queue()
    for job in JOBS:
        work.put(job)
    results: list[dict] = []
    lock = threading.Lock()
    threads = [
        threading.Thread(
            target=_worker,
            args=(args.gpu_offset + i, work, results, lock, args.iters, args.skip_existing),
            daemon=True,
        )
        for i in range(max(1, min(args.gpus, work.qsize())))
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    results.sort(key=lambda r: r.get("run", ""))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "iters_budget": args.iters,
        "rows": results,
    }, indent=2))
    print(json.dumps(results, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
