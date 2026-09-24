#!/usr/bin/env python3
"""Run production SR (no HR GT) over a granule tile manifest.

Reads the manifest from ``scripts/make_granule_tiles.py`` and launches
``optimize.py`` with the METHOD.md production knobs plus ``--allow_no_hr``.
GeoTIFF export is on by default (``sr_pred.tif`` / ``s2_bilinear.tif`` /
``s2_lr.tif`` under each run's ``qgis/`` folder).

Example
-------
    python scripts/make_granule_tiles.py --src data/s2_revisits/asker --mainland-only
    python scripts/run_production.py \\
        --manifest data/s2_revisits/asker/granule_tiles_lr512_manifest.json \\
        --gpus 8 --skip-existing
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.land_mask_lr512 import DEFAULT_LAND_MASK, filter_manifest_dict  # noqa: E402

DEFAULT_RUN_PREFIX = "prod_k4"


def _run_name(tile_id: str, prefix: str = DEFAULT_RUN_PREFIX) -> str:
    return f"{prefix}_{tile_id}"


def _metrics_path(dataset: str, tile_id: str, prefix: str = DEFAULT_RUN_PREFIX) -> Path:
    return ROOT / "single_samples" / dataset / "sample" / _run_name(tile_id, prefix) / "metrics.json"


def _sr_geotiff_path(dataset: str, tile_id: str, prefix: str = DEFAULT_RUN_PREFIX) -> Path:
    return (
        ROOT
        / "single_samples"
        / dataset
        / "sample"
        / _run_name(tile_id, prefix)
        / "qgis"
        / "sr_pred.tif"
    )


def _cmd(
    tile: dict,
    device: int,
    iters: int,
    *,
    export_geotiff: bool,
    run_prefix: str = DEFAULT_RUN_PREFIX,
    max_base_cloud_frac: float = 0.02,
    force_base_date: str | None = None,
) -> list[str]:
    parent = tile["parent"]
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
        "--allow_no_hr",
        "--lr_degradation",
        "s2_psf_m",
        "--recon_loss",
        "charbonnier",
        "--charbonnier_eps",
        "0.01",
        "--lr_tile",
        "128",
        "--lr_tiles_per_step",
        "4",
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
        "--max_base_cloud_frac",
        str(float(max_base_cloud_frac)),
        "--device",
        str(device),
    ]
    tile_date = tile.get("force_base_date") or force_base_date
    if tile_date:
        cmd.extend(["--force_base_date", str(tile_date)])
    if not export_geotiff:
        cmd.append("--no_qgis_export")
    return cmd


def _summarize(
    tile: dict, mpath: Path, *, skipped: bool, run_prefix: str = DEFAULT_RUN_PREFIX
) -> dict:
    m = json.loads(mpath.read_text())
    es = m.get("early_stop") or {}
    gpu_memory = m.get("gpu_memory") or {}
    qgis = _sr_geotiff_path(tile["parent"], tile["tile_id"], run_prefix)
    return {
        "tile_id": tile["tile_id"],
        "parent": tile["parent"],
        "s2_dir": tile["s2_dir"],
        "iy": tile.get("iy"),
        "ix": tile.get("ix"),
        "skipped": skipped,
        "has_hr_gt": m.get("has_hr_gt", False),
        "completed_iters": m.get("completed_iters"),
        "best_iter": es.get("best_val_iter"),
        "holdout_mse": es.get("best_val"),
        "training_time_s": m.get("training_time_seconds"),
        "peak_memory_gb": m.get("peak_memory_gb"),
        "process_peak_gpu_memory_gb": gpu_memory.get("process_peak_used_gb"),
        "torch_peak_reserved_gpu_memory_gb": gpu_memory.get("torch_peak_reserved_gb"),
        "gpu_memory_source": gpu_memory.get("process_memory_source"),
        "metrics_path": str(mpath.relative_to(ROOT)),
        "sr_pred_tif": str(qgis.relative_to(ROOT)) if qgis.is_file() else None,
    }


def _existing_run_matches(
    tile: dict,
    *,
    run_prefix: str,
    export_geotiff: bool,
    force_base_date: str | None,
) -> bool:
    """Only resume an output produced for the requested identity date."""
    metrics_path = _metrics_path(tile["parent"], tile["tile_id"], run_prefix)
    if not metrics_path.is_file():
        return False
    try:
        metrics = json.loads(metrics_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    requested_date = tile.get("force_base_date") or force_base_date
    recorded_date = (metrics.get("args") or {}).get("force_base_date")
    if (str(recorded_date)[:10] if recorded_date else None) != (
        str(requested_date)[:10] if requested_date else None
    ):
        return False
    if export_geotiff and not _sr_geotiff_path(
        tile["parent"], tile["tile_id"], run_prefix
    ).is_file():
        return False
    return True


def _run_one(
    tile: dict,
    device: int,
    iters: int,
    skip_existing: bool,
    export_geotiff: bool,
    run_prefix: str = DEFAULT_RUN_PREFIX,
    *,
    max_base_cloud_frac: float = 0.02,
    force_base_date: str | None = None,
) -> dict:
    mpath = _metrics_path(tile["parent"], tile["tile_id"], run_prefix)
    if skip_existing and _existing_run_matches(
        tile,
        run_prefix=run_prefix,
        export_geotiff=export_geotiff,
        force_base_date=force_base_date,
    ):
        print(f"[gpu{device}] skip {tile['tile_id']}", flush=True)
        return _summarize(tile, mpath, skipped=True, run_prefix=run_prefix)
    print(f"[gpu{device}] {tile['tile_id']} ...", flush=True)
    subprocess.run(
        _cmd(
            tile,
            device,
            iters,
            export_geotiff=export_geotiff,
            run_prefix=run_prefix,
            max_base_cloud_frac=max_base_cloud_frac,
            force_base_date=force_base_date,
        ),
        cwd=ROOT,
        check=True,
    )
    return _summarize(tile, mpath, skipped=False, run_prefix=run_prefix)


def _worker(
    gpu: int,
    work: queue.Queue,
    results: list,
    lock: threading.Lock,
    iters: int,
    skip_existing: bool,
    export_geotiff: bool,
    run_prefix: str = DEFAULT_RUN_PREFIX,
    max_base_cloud_frac: float = 0.02,
    force_base_date: str | None = None,
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
                run_prefix=run_prefix,
                max_base_cloud_frac=max_base_cloud_frac,
                force_base_date=force_base_date,
            )
            with lock:
                results.append(row)
            print(f"DONE {tile['tile_id']}", flush=True)
        except Exception as exc:  # noqa: BLE001
            with lock:
                results.append(
                    {
                        "tile_id": tile["tile_id"],
                        "parent": tile["parent"],
                        "s2_dir": tile["s2_dir"],
                        "error": str(exc),
                    }
                )
            print(f"FAIL {tile['tile_id']}: {exc}", flush=True)
        finally:
            work.task_done()


def _aggregate(rows: list[dict]) -> dict:
    ok = [r for r in rows if "error" not in r]
    fail = [r for r in rows if "error" in r]
    times = [r["training_time_s"] for r in ok if r.get("training_time_s") is not None]
    process_memory = [
        r["process_peak_gpu_memory_gb"]
        for r in ok
        if r.get("process_peak_gpu_memory_gb") is not None
    ]
    return {
        "n_ok": len(ok),
        "n_fail": len(fail),
        "n_skipped": sum(1 for r in ok if r.get("skipped")),
        "mean_train_s": (sum(times) / len(times)) if times else None,
        "sum_train_s": sum(times) if times else None,
        "mean_process_peak_gpu_memory_gb": (
            sum(process_memory) / len(process_memory) if process_memory else None
        ),
        "max_process_peak_gpu_memory_gb": max(process_memory) if process_memory else None,
        "n_with_process_peak_gpu_memory": len(process_memory),
        "n_with_sr_tif": sum(1 for r in ok if r.get("sr_pred_tif")),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--gpu-offset", type=int, default=0)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument(
        "--run-prefix",
        default=DEFAULT_RUN_PREFIX,
        help=(
            "optimize.py --run_name prefix (default prod_k4). "
            "Use a distinct prefix e.g. prod_k4_cloudmask so unmasked trains are not overwritten."
        ),
    )
    ap.add_argument(
        "--no-export-geotiff",
        action="store_true",
        help="Skip QGIS GeoTIFF write-out (faster smoke tests).",
    )
    ap.add_argument("--limit", type=int, default=0, help="Optional cap for a smoke test.")
    ap.add_argument(
        "--mainland-only",
        action="store_true",
        help=(
            "Drop ocean LR tiles using the default Norway mainland outline "
            "before training (for manifests tiled without --mainland-only)."
        ),
    )
    ap.add_argument(
        "--land-mask",
        type=Path,
        default=None,
        help="GeoJSON land outline (implies land filter at train time).",
    )
    ap.add_argument(
        "--min-land-frac",
        type=float,
        default=0.0,
        help="Min land-sample fraction when filtering (0 = any overlap).",
    )
    ap.add_argument(
        "--clear-counts-dir",
        type=Path,
        default=None,
        help="Optional clear-count GeoTIFF dir for --min-clear.",
    )
    ap.add_argument(
        "--min-clear",
        type=int,
        default=None,
        help="Skip LR cells below this clear-count (needs --clear-counts-dir).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "single_samples" / "sweep_results" / "production_run.json",
    )
    ap.add_argument(
        "--max-base-cloud-frac",
        type=float,
        default=0.02,
        help="Passed to optimize.py --max_base_cloud_frac (default 0.02).",
    )
    ap.add_argument(
        "--force-base-date",
        default=None,
        help="YYYY-MM-DD identity freeze for every tile in this run (optional).",
    )
    ap.add_argument(
        "--identity-plan",
        type=Path,
        default=None,
        help="JSON from plan_identity_icm.py; per-tile force_base_date from assignment.",
    )
    args = ap.parse_args()

    man_path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    man = json.loads(man_path.read_text())

    land_mask = args.land_mask
    if args.mainland_only and land_mask is None:
        land_mask = DEFAULT_LAND_MASK
    if land_mask is not None and not Path(land_mask).is_absolute():
        land_mask = ROOT / land_mask
    clear_dir = args.clear_counts_dir
    if clear_dir is not None and not Path(clear_dir).is_absolute():
        clear_dir = ROOT / clear_dir
    if args.min_clear is not None and clear_dir is None:
        raise SystemExit("--min-clear requires --clear-counts-dir")

    if land_mask is not None or args.min_clear is not None:
        src = Path(man["src"])
        src = src if src.is_absolute() else ROOT / src
        man, stats = filter_manifest_dict(
            man,
            src_dir=src,
            land_mask=land_mask,
            min_land_frac=float(args.min_land_frac),
            sample_n=9,
            clear_counts_dir=clear_dir,
            min_clear=args.min_clear,
        )
        print(
            f"train filter: kept {stats['n_kept']}/{stats['n_in']} "
            f"(dropped {stats['n_dropped']})",
            flush=True,
        )

    tiles = list(man["tiles"])
    if args.identity_plan is not None:
        ipath = (
            args.identity_plan
            if args.identity_plan.is_absolute()
            else ROOT / args.identity_plan
        )
        iplan = json.loads(ipath.read_text())
        assign = iplan.get("assignment") or {}
        for t in tiles:
            d = assign.get(t["tile_id"])
            if d:
                t["force_base_date"] = d
        print(
            f"identity plan: stamped {sum(1 for t in tiles if t.get('force_base_date'))}"
            f"/{len(tiles)} tiles from {ipath}",
            flush=True,
        )
    if args.limit > 0:
        tiles = tiles[: args.limit]
    if not tiles:
        raise SystemExit("manifest has no tiles")

    work: queue.Queue = queue.Queue()
    for t in tiles:
        work.put(t)

    results: list[dict] = []
    lock = threading.Lock()
    n_workers = max(1, min(args.gpus, work.qsize()))
    export_geotiff = not args.no_export_geotiff
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
                export_geotiff,
                args.run_prefix,
                float(args.max_base_cloud_frac),
                args.force_base_date,
            ),
            daemon=True,
        )
        for i in range(n_workers)
    ]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(man_path),
        "parent": man.get("parent"),
        "side": man.get("side"),
        "iters": args.iters,
        "run_prefix": args.run_prefix,
        "max_base_cloud_frac": float(args.max_base_cloud_frac),
        "force_base_date": args.force_base_date,
        "identity_plan": str(args.identity_plan) if args.identity_plan else None,
        "allow_no_hr": True,
        "export_geotiff": export_geotiff,
        "land_filter": man.get("land_filter"),
        "aggregate": _aggregate(results),
        "tiles": sorted(results, key=lambda r: r.get("tile_id") or ""),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    agg = summary["aggregate"]
    print(
        f"Done: {agg['n_ok']} ok / {agg['n_fail']} fail "
        f"({agg.get('n_with_sr_tif', 0)} with sr_pred.tif) → {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
