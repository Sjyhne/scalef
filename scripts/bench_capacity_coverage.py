#!/usr/bin/env python3
"""Hashgrid capacity + tile coverage sweep.

Two questions:

1. Capacity — at LR512 the auto-sized hashgrid is collision-free (0/16 levels
   exceed 2^21), but at LR2048 the 4 finest levels do (total cells 11.2x the
   table). Does raising log2_hashmap_size recover quality on the large AOI?
2. Coverage — fused k>1 is ~2x cheaper per step since the affine fix, so
   revisit k4/k8 on the small AOI where they were partly dismissed on speed.
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

# (run_name, s2_dir, extra CLI args)
JOBS: list[tuple[str, str, list[str]]] = [
    # --- capacity: LR2048, tile 512, k1, only log2_hashmap_size varies ---
    ("cap_lr2048_k1_log22", "asker_lr2048",
     ["--lr_tile", "512", "--lr_tiles_per_step", "1", "--hash_log2_hashmap_size", "22"]),
    ("cap_lr2048_k1_log23", "asker_lr2048",
     ["--lr_tile", "512", "--lr_tiles_per_step", "1", "--hash_log2_hashmap_size", "23"]),
    ("cap_lr2048_k1_log24", "asker_lr2048",
     ["--lr_tile", "512", "--lr_tiles_per_step", "1", "--hash_log2_hashmap_size", "24"]),
    # --- coverage x capacity on the large AOI ---
    ("cap_lr2048_k4_log21", "asker_lr2048",
     ["--lr_tile", "512", "--lr_tiles_per_step", "4", "--hash_log2_hashmap_size", "21"]),
    ("cap_lr2048_k4_log23", "asker_lr2048",
     ["--lr_tile", "512", "--lr_tiles_per_step", "4", "--hash_log2_hashmap_size", "23"]),
    # --- small-AOI fused-k revisit (full/k2 already measured post-fix) ---
    ("cov_lr512_k4", "asker_lr512",
     ["--lr_tile", "128", "--lr_tiles_per_step", "4"]),
    ("cov_lr512_k8", "asker_lr512",
     ["--lr_tile", "128", "--lr_tiles_per_step", "8"]),
    ("cov_lr512_full", "asker_lr512", []),
    ("cov_lr512_k2", "asker_lr512",
     ["--lr_tile", "128", "--lr_tiles_per_step", "2"]),
    # --- LR2048 coverage series at fixed capacity. Loss scaling unblocked
    # k>=2 here: the loss mean covers k*tile^2*3 LR elements, and past ~1.5M
    # of them dL/dout underflowed fp16 to exactly zero, so these never trained.
    ("cov_lr2048_k2_log22", "asker_lr2048",
     ["--lr_tile", "512", "--lr_tiles_per_step", "2", "--hash_log2_hashmap_size", "22"]),
    ("cov_lr2048_k4_log22", "asker_lr2048",
     ["--lr_tile", "512", "--lr_tiles_per_step", "4", "--hash_log2_hashmap_size", "22"]),
    ("cov_lr2048_k8_log22", "asker_lr2048",
     ["--lr_tile", "512", "--lr_tiles_per_step", "8", "--hash_log2_hashmap_size", "22"]),
    # k8/k16 at tile 512 OOM inside tinycudann as a single fused forward.
    # Accumulate over micro-batches of 2 tiles: tcnn's arena lives outside
    # torch's allocator, so 4 tiles per micro-batch still exhausts an 80 GB
    # card even though torch only reports ~11 GB for a plain k4 step.
    ("cov_lr2048_k8_log22_ga4", "asker_lr2048",
     ["--lr_tile", "512", "--lr_tiles_per_step", "8",
      "--hash_log2_hashmap_size", "22", "--grad_accum", "4"]),
    ("cov_lr2048_k16_log22_ga8", "asker_lr2048",
     ["--lr_tile", "512", "--lr_tiles_per_step", "0",
      "--hash_log2_hashmap_size", "22", "--grad_accum", "8"]),
    # --- control: LR512 is already collision-free at log2=21, so raising the
    # table there should do nothing. If it helps anyway, the LR2048 gain is
    # generic extra capacity rather than collision relief.
    ("ctl_lr512_full_log23", "asker_lr512",
     ["--hash_log2_hashmap_size", "23"]),
    ("ctl_lr512_k2_log23", "asker_lr512",
     ["--lr_tile", "128", "--lr_tiles_per_step", "2", "--hash_log2_hashmap_size", "23"]),
    # --- budget: at LR512 full-field every iteration updates every pixel, but at
    # LR2048 a k1 tile touches 1/16 of the field, so per-pixel updates are ~16x
    # scarcer. k16 (=0, all tiles) restores full coverage; the long arm instead
    # buys budget with iterations.
    ("cap_lr2048_k16_log22", "asker_lr2048",
     ["--lr_tile", "512", "--lr_tiles_per_step", "0", "--hash_log2_hashmap_size", "22"]),
    ("cap_lr2048_k1_log22_long", "asker_lr2048",
     ["--lr_tile", "512", "--lr_tiles_per_step", "1", "--hash_log2_hashmap_size", "22",
      "--early_stop_patience", "8"]),
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


def _summarize(run_name: str, s2_dir: str, extra: list[str]) -> dict:
    mpath = ROOT / "single_samples" / "asker" / "sample" / run_name / "metrics.json"
    if not mpath.is_file():
        return {"run": run_name, "status": "missing"}
    m = json.loads(mpath.read_text())
    es = m.get("early_stop") or {}
    iters = m.get("completed_iters") or 0
    t = m.get("training_time_seconds") or 0.0
    return {
        "run": run_name,
        "s2_dir": s2_dir,
        "args": " ".join(extra),
        "lpips": m["lpips"]["model"],
        "lpips_bilinear": m["lpips"]["bilinear"],
        "lpips_vs_bilinear": m["lpips"]["bilinear"] - m["lpips"]["model"],
        "psnr": m["psnr"]["model"],
        "psnr_bilinear": m["psnr"]["bilinear"],
        "ssim": m["ssim"]["model"],
        "completed_iters": iters,
        "best_iter": es.get("best_val_iter"),
        "best_val": es.get("best_val_loss"),
        "training_time_s": round(t, 1),
        "ms_per_iter": round(t / iters * 1000, 1) if iters else None,
        "peak_memory_gb": round(m.get("peak_memory_gb") or 0.0, 2),
    }


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
            row = _summarize(run_name, s2_dir, extra)
            with lock:
                results.append(row)
            print(f"DONE {run_name}: lpips={row.get('lpips')}", flush=True)
        except Exception as exc:  # noqa: BLE001
            with lock:
                results.append({"run": run_name, "error": str(exc)})
            print(f"FAIL {run_name}: {exc}", flush=True)
        finally:
            work.task_done()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--gpu-offset", type=int, default=0)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--only", nargs="*", default=None, help="Subset of run names.")
    ap.add_argument(
        "--run-tag", default="",
        help="Suffix appended to every run name, so a re-run keeps old results.",
    )
    ap.add_argument(
        "--out", type=Path,
        default=ROOT / "single_samples" / "sweep_results" / "bench_capacity_coverage.json",
    )
    args = ap.parse_args()

    jobs = JOBS if not args.only else [j for j in JOBS if j[0] in set(args.only)]
    if args.run_tag:
        jobs = [(name + args.run_tag, s2_dir, extra) for name, s2_dir, extra in jobs]
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
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "iters_budget": args.iters,
        "note": "baseline log21 k1/k2 on LR2048 = lr2048_k*_t512_holdout_v4",
        "rows": results,
    }, indent=2))
    print(json.dumps(results, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
