#!/usr/bin/env python3
"""Hashgrid level-ladder sweep: how much encoder capacity does each AOI need?

The ladder is auto-sized from the LR shape (max = lr_size, base = max/4), so
LR512 and LR2048 already run *physically identical* ladders -- 16 levels from
40 m down to 10 m per cell. The only difference is cell count: 262k vs 4.19M at
the finest level. Raising log2_hashmap_size to remove that oversubscription did
nothing (0.5140 -> 0.5134 at k1), so collisions are not the constraint.

Two open questions this sweep answers:

1. Is LR512 over-provisioned? Nothing establishes that 16 levels is *needed* --
   it is just what auto-sizing produced. If 8 levels holds quality, that is a
   free speedup on top of k4.
2. Is LR2048 under-provisioned? Denser levels within the same 40->10 m span add
   parameters without reaching new spatial scales. The coverage evidence points
   at optimization budget instead (LR2048 keeps improving k1->k8 while LR512
   flattens), so the budget arm tests the likelier cause alongside.

Because n_levels also sets the decoder input width (n_levels x
n_features_per_level), each ladder has a controlled variant holding that product
at 32 to separate the ladder from the MLP input width.

The up-ladder runs at log2=24, where no level is capped for LR2048 (finest level
4.19M cells < 16.8M entries), so the encoder is collision-free and only the
ladder varies. The budget arm stays at log2=22 to compare directly against the
established k8 baseline of 0.4121.
"""

from __future__ import annotations

import argparse
import json
import queue
import re
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

LR2048_K8 = ["--lr_tile", "512", "--lr_tiles_per_step", "8", "--grad_accum", "4"]
LR512_K4 = ["--lr_tile", "128", "--lr_tiles_per_step", "4"]

# (run_name, s2_dir, extra CLI args). Longest jobs first so the FIFO queue
# hands them out before the short LR512 arms fill the remaining workers.
JOBS: list[tuple[str, str, list[str]]] = [
    # --- budget arm: k8 early-stopped at 3600 (best 3000) under patience 3.
    # Relax it to see whether HR quality was still improving when the LR-space
    # holdout metric went flat. Same log22 as the 0.4121 baseline.
    ("lad_lr2048_k8_budget", "asker_lr2048",
     [*LR2048_K8, "--hash_log2_hashmap_size", "22",
      "--early_stop_patience", "8", "--iters", "15000"]),
    # --- LR2048 up-ladder at collision-free log24, only n_levels varies ---
    ("lad_lr2048_k8_lvl16_log24", "asker_lr2048",
     [*LR2048_K8, "--hash_log2_hashmap_size", "24", "--hash_n_levels", "16"]),
    ("lad_lr2048_k8_lvl20_log24", "asker_lr2048",
     [*LR2048_K8, "--hash_log2_hashmap_size", "24", "--hash_n_levels", "20"]),
    ("lad_lr2048_k8_lvl24_log24", "asker_lr2048",
     [*LR2048_K8, "--hash_log2_hashmap_size", "24", "--hash_n_levels", "24"]),
    # Controlled: 32 x 1 = 32 decoder inputs, same width as the 16 x 2 baseline.
    ("lad_lr2048_k8_lvl32f1_log24", "asker_lr2048",
     [*LR2048_K8, "--hash_log2_hashmap_size", "24",
      "--hash_n_levels", "32", "--hash_n_features_per_level", "1"]),
    # --- LR512 down-ladder, free (features stay at 2, decoder input shrinks) ---
    ("lad_lr512_k4_lvl16", "asker_lr512", [*LR512_K4, "--hash_n_levels", "16"]),
    ("lad_lr512_k4_lvl12", "asker_lr512", [*LR512_K4, "--hash_n_levels", "12"]),
    ("lad_lr512_k4_lvl8", "asker_lr512", [*LR512_K4, "--hash_n_levels", "8"]),
    ("lad_lr512_k4_lvl4", "asker_lr512", [*LR512_K4, "--hash_n_levels", "4"]),
    ("lad_lr512_k4_lvl2", "asker_lr512", [*LR512_K4, "--hash_n_levels", "2"]),
    # --- LR512 down-ladder, controlled (levels x features held at 32) ---
    ("lad_lr512_k4_lvl8f4", "asker_lr512",
     [*LR512_K4, "--hash_n_levels", "8", "--hash_n_features_per_level", "4"]),
    ("lad_lr512_k4_lvl4f8", "asker_lr512",
     [*LR512_K4, "--hash_n_levels", "4", "--hash_n_features_per_level", "8"]),
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


def _encoder_params(lr_size: int, n_levels: int, features: int, log2: int) -> int:
    """Trainable hashgrid entries: sum over levels of min(cells, 2**log2) x features."""
    mx = lr_size
    base = max(8, mx // 4)
    growth = (mx / base) ** (1.0 / (n_levels - 1)) if n_levels > 1 else 1.0
    cap = 2 ** log2
    total = 0
    for i in range(n_levels):
        res = max(1, round(base * growth ** i))
        total += min(res * res, cap) * features
    return total


def _describe(s2_dir: str, extra: list[str]) -> dict:
    joined = " ".join(extra)

    def flag(name: str, default: int) -> int:
        m = re.search(rf"--{name}\s+(\d+)", joined)
        return int(m.group(1)) if m else default

    lr_size = 2048 if "lr2048" in s2_dir else 512
    levels = flag("hash_n_levels", 16)
    features = flag("hash_n_features_per_level", 2)
    log2 = flag("hash_log2_hashmap_size", 21)
    return {
        "lr_size": lr_size,
        "levels": levels,
        "features": features,
        "log2_hashmap": log2,
        "decoder_in": levels * features,
        "encoder_params": _encoder_params(lr_size, levels, features, log2),
    }


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
        **_describe(s2_dir, extra),
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
            print(f"DONE {run_name}: lpips={row.get('lpips')} "
                  f"levels={row.get('levels')} params={row.get('encoder_params')}", flush=True)
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
    ap.add_argument("--run-tag", default="", help="Suffix appended to every run name.")
    ap.add_argument(
        "--out", type=Path,
        default=ROOT / "single_samples" / "sweep_results" / "bench_capacity_ladder.json",
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
        "note": "ladder auto-sizes to LR shape; LR512 and LR2048 span identical 40->10 m",
        "rows": results,
    }, indent=2))
    print(json.dumps(results, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
