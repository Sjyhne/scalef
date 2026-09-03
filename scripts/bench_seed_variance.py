#!/usr/bin/env python3
"""Seed-variance sweep: what is the noise floor on LR512 LPIPS?

Every result in the coverage and capacity sweeps is a single run at the default
seed (6). The LR512 frontier spans only 0.019 LPIPS end to end (k2 0.3813 ->
full 0.3626), and the level ladder produced a 12-level outlier 0.026 worse than
both its 8- and 16-level neighbours, which capacity alone should not do. If
run-to-run noise is of that order then the frontier ordering is not resolvable
from one sample per config and the recommendation needs restating.

Arms:
  * k4 at 5 seeds -- the primary noise estimate on the recommended config.
  * 12 levels at 3 seeds -- does the ladder outlier reproduce, or was it noise?
  * k2 / k8 / full at 3 seeds each -- is the frontier ordering separable at all?

All arms are LR512, so each run costs 45-190 s.
"""

from __future__ import annotations

import argparse
import json
import queue
import statistics
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

K4 = ["--lr_tile", "128", "--lr_tiles_per_step", "4"]

# (group, extra args, seeds) -> expanded into one job per seed.
GROUPS: list[tuple[str, list[str], list[int]]] = [
    ("var_k4", K4, [6, 7, 8, 9, 10]),
    ("var_lvl12", [*K4, "--hash_n_levels", "12"], [6, 7, 8]),
    ("var_k2", ["--lr_tile", "128", "--lr_tiles_per_step", "2"], [6, 7, 8]),
    ("var_k8", ["--lr_tile", "128", "--lr_tiles_per_step", "8"], [6, 7, 8]),
    ("var_full", [], [6, 7, 8]),
]

# Longest first so the FIFO queue does not leave a slow arm for the tail.
JOBS: list[tuple[str, str, list[str], str]] = [
    (f"{group}_s{seed}", "asker_lr512", [*extra, "--seed", str(seed)], group)
    for group, extra, seeds in sorted(
        GROUPS, key=lambda g: {"var_full": 0, "var_k8": 1, "var_k4": 2}.get(g[0], 3)
    )
    for seed in seeds
]


def _cmd(run_name: str, s2_dir: str, extra: list[str], device: int, iters: int) -> list[str]:
    return [
        sys.executable, str(ROOT / "optimize.py"),
        "--dataset", "asker",
        "--s2-dir", str(ROOT / "data" / "s2_revisits" / s2_dir),
        "--run_name", run_name,
        "--recon_loss", "charbonnier", "--charbonnier_eps", "0.01",
        "--early_stop_metric", "holdout_mse",
        "--early_stop_patience", "3", "--early_stop_min_iters", "1000",
        "--iters", str(iters),
        "--device", str(device),
        "--spatial_holdout", "0.1", "--holdout_block", "0",
        "--eval_every", "200", "--hr_render_tile", "2048",
        "--lr_tile_mix", "within",
        "--spatial_alignment_path", str(ROOT / "eval" / "spatial_alignment.json"),
        "--no_qgis_export",
        *extra,
    ]


def _summarize(run_name: str, group: str, extra: list[str]) -> dict:
    mpath = ROOT / "single_samples" / "asker" / "sample" / run_name / "metrics.json"
    if not mpath.is_file():
        return {"run": run_name, "group": group, "status": "missing"}
    m = json.loads(mpath.read_text())
    es = m.get("early_stop") or {}
    iters = m.get("completed_iters") or 0
    t = m.get("training_time_seconds") or 0.0
    return {
        "run": run_name,
        "group": group,
        "args": " ".join(extra),
        "lpips": m["lpips"]["model"],
        "lpips_vs_bilinear": m["lpips"]["bilinear"] - m["lpips"]["model"],
        "psnr": m["psnr"]["model"],
        "ssim": m["ssim"]["model"],
        "holdout_mse": es.get("best_val_loss"),
        "completed_iters": iters,
        "best_iter": es.get("best_val_iter"),
        "training_time_s": round(t, 1),
    }


def _worker(gpu: int, work: queue.Queue, results: list, lock: threading.Lock,
            iters: int, skip_existing: bool) -> None:
    while True:
        try:
            run_name, s2_dir, extra, group = work.get_nowait()
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
            row = _summarize(run_name, group, extra)
            with lock:
                results.append(row)
            print(f"DONE {run_name}: lpips={row.get('lpips')}", flush=True)
        except Exception as exc:  # noqa: BLE001
            with lock:
                results.append({"run": run_name, "group": group, "error": str(exc)})
            print(f"FAIL {run_name}: {exc}", flush=True)
        finally:
            work.task_done()


def _stats(results: list[dict]) -> list[dict]:
    """Per-group mean / sd / range of LPIPS across seeds."""
    out = []
    groups = {}
    for r in results:
        if "lpips" in r:
            groups.setdefault(r["group"], []).append(r["lpips"])
    for group, vals in groups.items():
        vals = sorted(vals)
        out.append({
            "group": group,
            "n": len(vals),
            "mean": round(statistics.fmean(vals), 5),
            "sd": round(statistics.stdev(vals), 5) if len(vals) > 1 else None,
            "min": round(vals[0], 5),
            "max": round(vals[-1], 5),
            "spread": round(vals[-1] - vals[0], 5),
        })
    return sorted(out, key=lambda s: s["mean"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--gpus", type=int, default=2)
    ap.add_argument("--gpu-offset", type=int, default=0)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--only", nargs="*", default=None, help="Subset of group names.")
    ap.add_argument(
        "--out", type=Path,
        default=ROOT / "single_samples" / "sweep_results" / "bench_seed_variance.json",
    )
    args = ap.parse_args()

    jobs = JOBS if not args.only else [j for j in JOBS if j[3] in set(args.only)]
    work: queue.Queue = queue.Queue()
    for job in jobs:
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
    stats = _stats(results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "iters_budget": args.iters,
        "stats": stats,
        "rows": results,
    }, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
