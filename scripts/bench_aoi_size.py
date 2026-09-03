#!/usr/bin/env python3
"""AOI-size ladder: how does quality scale with the ground extent of one INR?

At k16 the LR2048 route runs full coverage, equal iterations, matched per-area
capacity and a collision-free encoder, and is still ~0.037 LPIPS behind LR512.
Collisions, hash headroom, per-area capacity, ladder resolution and per-pixel
update scarcity are all excluded, which leaves the single global affine spanning
a 20 km AOI as the leading suspect -- and the possibility that part of the gap
is not real, since the two routes are scored on different ground (bilinear
differs, 0.4401 vs 0.4442).

This sweep trains full coverage at every available AOI size (2.56 / 5.12 /
10.24 / 20.48 km) so the degradation curve can be read directly. A global-affine
failure predicts smooth degradation with extent. GeoTIFF export is enabled so
metrics can be recomputed offline on the common centre window bounded by the
smallest AOI (1024 HR px = 2.56 km), removing the crop confound.

It also answers the production question of which AOI size to use: covering a
10980 px granule takes 1936 / 484 / 121 / 36 AOIs at 256 / 512 / 1024 / 2048,
so AOI size trades against per-AOI overhead.

Patience is 8 rather than the usual 3. The seed-variance sweep showed random
tile sampling makes holdout_mse jitter, and patience 3 trips on that jitter:
five k4 seeds stopped anywhere between 2000 and 4800 iterations, and final LPIPS
tracked the stopping point (0.3698 at 4800 vs 0.3884 at 2000). Full coverage is
much steadier, but the larger AOIs here still sample tiles, so the loose
patience keeps stopping behaviour from confounding the size comparison.
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
GRANULE_PX = 10980

# (run_name, s2_dir, lr_px, extra args). Full coverage everywhere: the small
# AOIs fit as a single field, the large ones tile with exact gradient
# accumulation (2 tiles per micro-batch, which is what fits tcnn's arena).
JOBS: list[tuple[str, str, int, list[str]]] = [
    ("aoi_lr2048_full", "asker_lr2048", 2048,
     ["--lr_tile", "512", "--lr_tiles_per_step", "0", "--grad_accum", "8",
      "--hash_log2_hashmap_size", "22"]),
    ("aoi_lr1024_full", "asker_lr1024", 1024,
     ["--lr_tile", "512", "--lr_tiles_per_step", "0", "--grad_accum", "2"]),
    ("aoi_lr512_full", "asker_lr512", 512, []),
    ("aoi_lr256_full", "asker_lr256", 256, []),
]


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
        # GeoTIFF export left ON so metrics can be recomputed on common ground.
        *extra,
    ]


def _summarize(run_name: str, s2_dir: str, lr_px: int, extra: list[str]) -> dict:
    rundir = ROOT / "single_samples" / "asker" / "sample" / run_name
    mpath = rundir / "metrics.json"
    if not mpath.is_file():
        return {"run": run_name, "status": "missing"}
    m = json.loads(mpath.read_text())
    es = m.get("early_stop") or {}
    iters = m.get("completed_iters") or 0
    t = m.get("training_time_seconds") or 0.0
    aois = -(-GRANULE_PX // lr_px) ** 2
    return {
        "run": run_name,
        "s2_dir": s2_dir,
        "lr_px": lr_px,
        "extent_km": round(lr_px * 10 / 1000, 2),
        "aois_per_granule": aois,
        "args": " ".join(extra),
        "lpips": m["lpips"]["model"],
        "lpips_bilinear": m["lpips"]["bilinear"],
        "lpips_vs_bilinear": m["lpips"]["bilinear"] - m["lpips"]["model"],
        "psnr": m["psnr"]["model"],
        "psnr_bilinear": m["psnr"]["bilinear"],
        "ssim": m["ssim"]["model"],
        "completed_iters": iters,
        "best_iter": es.get("best_val_iter"),
        "holdout_mse": es.get("best_val_loss"),
        "training_time_s": round(t, 1),
        "granule_hours_8gpu": round(t * aois / 8 / 3600, 3),
        "peak_memory_gb": round(m.get("peak_memory_gb") or 0.0, 2),
        "geotiffs": sorted(p.name for p in rundir.glob("*.tif")),
    }


def _worker(gpu: int, work: queue.Queue, results: list, lock: threading.Lock,
            iters: int, skip_existing: bool) -> None:
    while True:
        try:
            run_name, s2_dir, lr_px, extra = work.get_nowait()
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
            row = _summarize(run_name, s2_dir, lr_px, extra)
            with lock:
                results.append(row)
            print(f"DONE {run_name}: lpips={row.get('lpips')} "
                  f"gain={row.get('lpips_vs_bilinear')}", flush=True)
        except Exception as exc:  # noqa: BLE001
            with lock:
                results.append({"run": run_name, "error": str(exc)})
            print(f"FAIL {run_name}: {exc}", flush=True)
        finally:
            work.task_done()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iters", type=int, default=8000)
    ap.add_argument("--gpus", type=int, default=4)
    ap.add_argument("--gpu-offset", type=int, default=0)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument(
        "--out", type=Path,
        default=ROOT / "single_samples" / "sweep_results" / "bench_aoi_size.json",
    )
    args = ap.parse_args()

    jobs = JOBS if not args.only else [j for j in JOBS if j[0] in set(args.only)]
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

    results.sort(key=lambda r: r.get("lr_px", 0))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "iters_budget": args.iters,
        "note": "full coverage at every AOI size, patience 8, GeoTIFF export on",
        "rows": results,
    }, indent=2))
    print(json.dumps(results, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
