#!/usr/bin/env python3
"""Train production recipe on every complete patch-grid tile.

Reads the manifest from ``make_complete_patch_tiles.py`` and runs ``optimize.py``
with the METHOD.md production knobs (Charbonnier, s2_psf_m, holdout_mse p8).

Default is LR512 k4. Size ladder uses ``--lr-tile`` / ``--run-prefix`` (see
``run_complete_patch_size_ladder.py``).

Results land under::

    single_samples/{parent_city}/sample/{run_prefix}_{tile_id}/
"""

from __future__ import annotations

import argparse
import json
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = (
    ROOT / "data" / "s2_revisits" / "map" / "patch_grid_lr512" / "complete_patch_tiles_manifest.json"
)
DEFAULT_OUT = ROOT / "single_samples" / "sweep_results" / "bench_complete_patches_k4.json"
ALIGN_PATH = ROOT / "eval" / "spatial_alignment.json"


def _run_name(tile_id: str, run_prefix: str) -> str:
    # Legacy LR512 runs used prod_k4_{tile_id} with no extra prefix collision.
    if run_prefix == "prod_k4" or tile_id.startswith(run_prefix):
        return f"prod_k4_{tile_id}" if run_prefix == "prod_k4" else tile_id
    return f"{run_prefix}_{tile_id}"


def _metrics_path(parent_city: str, tile_id: str, run_prefix: str) -> Path:
    return (
        ROOT
        / "single_samples"
        / parent_city
        / "sample"
        / _run_name(tile_id, run_prefix)
        / "metrics.json"
    )


def _cmd(
    tile: dict,
    device: int,
    iters: int,
    *,
    export_geotiff: bool,
    lr_tile: int,
    lr_tiles_per_step: int,
    run_prefix: str,
) -> list[str]:
    parent = tile["parent_city"]
    tile_id = tile["tile_id"]
    s2_dir = ROOT / tile["s2_dir"]
    cmd = [
        sys.executable,
        str(ROOT / "optimize.py"),
        "--dataset",
        parent,
        "--s2-dir",
        str(s2_dir),
        "--run_name",
        _run_name(tile_id, run_prefix),
        "--lr_degradation",
        "s2_psf_m",
        "--recon_loss",
        "charbonnier",
        "--charbonnier_eps",
        "0.01",
        "--lr_tile",
        str(int(lr_tile)),
        "--lr_tiles_per_step",
        str(int(lr_tiles_per_step)),
        "--lr_tile_mix",
        "within",
        "--early_stop_metric",
        "holdout_mse",
        "--early_stop_patience",
        "8",
        "--early_stop_min_iters",
        "1000",
        "--iters",
        str(iters),
        "--eval_every",
        "200",
        "--hr_render_tile",
        "2048",
        "--spatial_holdout",
        "0.1",
        "--holdout_block",
        "0",
        "--device",
        str(device),
    ]
    if ALIGN_PATH.is_file():
        cmd += ["--spatial_alignment_path", str(ALIGN_PATH)]
    if not export_geotiff:
        cmd.append("--no_qgis_export")
    return cmd


def _summarize(tile: dict, mpath: Path, *, skipped: bool, side: int | None) -> dict:
    m = json.loads(mpath.read_text())
    es = m.get("early_stop") or {}
    lp = m.get("lpips") or {}
    return {
        "tile_id": tile["tile_id"],
        "parent_city": tile["parent_city"],
        "s2_dir": tile["s2_dir"],
        "side": side if side is not None else tile.get("side"),
        "patch_row": tile.get("patch_row"),
        "patch_col": tile.get("patch_col"),
        "skipped": skipped,
        "lpips": lp.get("model"),
        "lpips_bilinear": lp.get("bilinear"),
        "lpips_vs_bilinear": lp.get("improvement"),
        "psnr": (m.get("psnr") or {}).get("model"),
        "ssim": (m.get("ssim") or {}).get("model"),
        "completed_iters": m.get("completed_iters"),
        "best_iter": es.get("best_val_iter"),
        "holdout_mse": es.get("best_val"),
        "training_time_s": m.get("training_time_seconds"),
        "peak_memory_gb": m.get("peak_memory_gb"),
        "metrics_path": str(mpath.relative_to(ROOT)),
        "eval_valid_fraction": (m.get("eval_mask") or {}).get("valid_fraction"),
    }


def _run_one(
    tile: dict,
    device: int,
    iters: int,
    skip_existing: bool,
    export_geotiff: bool,
    *,
    lr_tile: int,
    lr_tiles_per_step: int,
    run_prefix: str,
    side: int | None,
) -> dict:
    mpath = _metrics_path(tile["parent_city"], tile["tile_id"], run_prefix)
    if skip_existing and mpath.is_file():
        print(f"[gpu{device}] skip {tile['tile_id']}", flush=True)
        return _summarize(tile, mpath, skipped=True, side=side)
    print(f"[gpu{device}] {tile['tile_id']} ...", flush=True)
    cmd = _cmd(
        tile,
        device,
        iters,
        export_geotiff=export_geotiff,
        lr_tile=lr_tile,
        lr_tiles_per_step=lr_tiles_per_step,
        run_prefix=run_prefix,
    )
    last_exc: Exception | None = None
    for attempt in range(1, 5):
        try:
            subprocess.run(cmd, cwd=ROOT, check=True)
            last_exc = None
            break
        except subprocess.CalledProcessError as exc:
            last_exc = exc
            wait_s = 20 * attempt
            print(
                f"[gpu{device}] retry {attempt}/4 {tile['tile_id']} after exit {exc.returncode}; "
                f"sleep {wait_s}s",
                flush=True,
            )
            time.sleep(wait_s)
    if last_exc is not None:
        raise last_exc
    return _summarize(tile, mpath, skipped=False, side=side)


def _worker(
    gpu: int,
    work: queue.Queue,
    results: list,
    lock: threading.Lock,
    iters: int,
    skip_existing: bool,
    export_geotiff: bool,
    lr_tile: int,
    lr_tiles_per_step: int,
    run_prefix: str,
    side: int | None,
) -> None:
    while True:
        try:
            tile = work.get_nowait()
        except queue.Empty:
            return
        try:
            row = _run_one(
                tile,
                gpu,
                iters,
                skip_existing,
                export_geotiff,
                lr_tile=lr_tile,
                lr_tiles_per_step=lr_tiles_per_step,
                run_prefix=run_prefix,
                side=side,
            )
            with lock:
                results.append(row)
            print(f"DONE {tile['tile_id']}", flush=True)
        except Exception as exc:  # noqa: BLE001
            with lock:
                results.append(
                    {
                        "tile_id": tile["tile_id"],
                        "parent_city": tile["parent_city"],
                        "s2_dir": tile["s2_dir"],
                        "error": str(exc),
                    }
                )
            print(f"FAIL {tile['tile_id']}: {exc}", flush=True)
        finally:
            work.task_done()


def _aggregate(rows: list[dict]) -> dict:
    ok = [r for r in rows if "error" not in r and r.get("lpips") is not None]
    if not ok:
        return {"n_ok": 0}
    lp = [r["lpips"] for r in ok]
    gain = [r["lpips_vs_bilinear"] for r in ok if r.get("lpips_vs_bilinear") is not None]
    by_city: dict[str, list[float]] = {}
    for r in ok:
        by_city.setdefault(r["parent_city"], []).append(r["lpips"])
    return {
        "n_ok": len(ok),
        "n_fail": sum(1 for r in rows if "error" in r),
        "mean_lpips": sum(lp) / len(lp),
        "median_lpips": sorted(lp)[len(lp) // 2],
        "min_lpips": min(lp),
        "max_lpips": max(lp),
        "mean_lpips_vs_bilinear": (sum(gain) / len(gain)) if gain else None,
        "mean_train_s": sum(r["training_time_s"] or 0 for r in ok) / len(ok),
        "per_city_mean_lpips": {
            c: sum(v) / len(v) for c, v in sorted(by_city.items())
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--gpu-offset", type=int, default=0)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument(
        "--export-geotiff",
        action="store_true",
        help="Write QGIS GeoTIFFs (off by default for bulk speed; metrics/PNGs still saved).",
    )
    ap.add_argument("--limit", type=int, default=0, help="Optional cap for a smoke test.")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--lr-tile", type=int, default=128)
    ap.add_argument("--lr-tiles-per-step", type=int, default=4)
    ap.add_argument(
        "--run-prefix",
        type=str,
        default="prod_k4",
        help="Run dir prefix (default prod_k4 keeps legacy LR512 paths).",
    )
    ap.add_argument("--side", type=int, default=0, help="LR side length for logging (0=from manifest).")
    args = ap.parse_args()

    man_path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    man = json.loads(man_path.read_text())
    tiles = list(man["tiles"])
    if args.limit > 0:
        tiles = tiles[: args.limit]
    side = int(args.side) if args.side > 0 else man.get("side")

    # Legacy LR512 skip-existing: run_prefix prod_k4 + tile asker_p00_01 → prod_k4_asker_p00_01
    run_prefix = args.run_prefix
    if run_prefix == "prod_k4_lr512":
        run_prefix = "prod_k4"

    work: queue.Queue = queue.Queue()
    for t in tiles:
        work.put(t)

    results: list[dict] = []
    lock = threading.Lock()
    n_workers = max(1, min(args.gpus, work.qsize()))
    threads = [
        threading.Thread(
            target=_worker,
            args=(
                args.gpu_offset + i,
                work,
                results,
                lock,
                args.iters,
                args.skip_existing,
                args.export_geotiff,
                int(args.lr_tile),
                int(args.lr_tiles_per_step),
                run_prefix,
                side,
            ),
            daemon=True,
        )
        for i in range(n_workers)
    ]
    print(
        f"Running {len(tiles)} tiles on {n_workers} GPUs "
        f"(lr_tile={args.lr_tile} k={args.lr_tiles_per_step} prefix={run_prefix})",
        flush=True,
    )
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    results.sort(key=lambda r: (r.get("parent_city", ""), r.get("tile_id", "")))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(man_path.relative_to(ROOT)) if man_path.is_relative_to(ROOT) else str(man_path),
        "iters_budget": args.iters,
        "side": side,
        "config": {
            "lr_tile": int(args.lr_tile),
            "lr_tiles_per_step": int(args.lr_tiles_per_step),
            "run_prefix": run_prefix,
            "early_stop_patience": 8,
            "recon_loss": "charbonnier",
            "lr_degradation": "s2_psf_m",
            "export_geotiff": bool(args.export_geotiff),
        },
        "n_tiles": len(tiles),
        "rows": results,
        "summary": _aggregate(results),
        "skipped_at_materialize": man.get("skipped") or [],
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["summary"], indent=2), flush=True)
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
