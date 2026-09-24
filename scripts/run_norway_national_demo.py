#!/usr/bin/env python3
"""Orchestrate a selected Norway MGRS demo from fetch through national mosaic.

Consumes ``select_norway_mgrs_block.py`` output and invokes existing processing
scripts. ``--dry-run`` performs no network, tiling, training, or mosaic work;
it writes the exact command plan. ``--resume`` skips successful stages whose
expected output still exists. Every attempt is recorded with timing and errors.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def commands_for_tile(tile: dict, args) -> list[tuple[str, list[str], Path]]:
    mgrs = tile["mgrs_tile"]
    bbox = tile.get("bbox")
    if not bbox or len(bbox) != 4:
        raise ValueError(f"{mgrs}: selection manifest needs bbox")
    start, end = args.date_range.split("/", 1)
    data_dir = ROOT / "data" / "s2_revisits" / mgrs
    manifest = data_dir / f"granule_tiles_lr{args.side}_manifest.json"
    prod_summary = ROOT / "single_samples" / "sweep_results" / f"production_{mgrs}_lr{args.side}.json"
    mosaic = ROOT / "production" / "mosaics" / f"{mgrs}_sr_2p5m.tif"
    fetch = [
        sys.executable,
        str(ROOT / "scripts" / "fetch_s2_revisits.py"),
        "--out",
        str(data_dir),
        "--start-date",
        start,
        "--end-date",
        end,
        "--bbox",
        *(str(value) for value in bbox),
        "--mgrs-tile",
        mgrs,
        "--num-samples",
        str(args.num_samples),
        "--cloud-method",
        args.cloud_method,
        "--max-cloud-frac",
        str(args.max_cloud_frac),
        "--min-valid-frac",
        str(args.min_valid_frac),
        "--max-stac-cloud",
        str(args.max_stac_cloud),
        "--device",
        args.cloud_device,
        "--no-preview",
    ]
    if args.include_shadow:
        fetch.append("--include-shadow")
    tile_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "make_granule_tiles.py"),
        "--src",
        str(data_dir),
        "--side",
        str(args.side),
        "--manifest",
        str(manifest),
        "--mainland-only",
        "--min-land-frac",
        str(args.min_land_frac),
        "--clear-counts-dir",
        str(args.heatmap_dir),
        "--min-clear",
        str(args.min_clear),
    ]
    train = [
        sys.executable,
        str(ROOT / "scripts" / "run_production.py"),
        "--manifest",
        str(manifest),
        "--iters",
        str(args.iters),
        "--gpus",
        str(args.gpus),
        "--gpu-offset",
        str(args.gpu_offset),
        "--skip-existing",
        "--out",
        str(prod_summary),
    ]
    granule_mosaic = [
        sys.executable,
        str(ROOT / "scripts" / "mosaic_granule_sr.py"),
        "--manifest",
        str(manifest),
        "--out",
        str(mosaic),
        "--allow-missing",
    ]
    return [
        ("fetch", fetch, data_dir / "meta.json"),
        ("tile", tile_cmd, manifest),
        ("train", train, prod_summary),
        ("granule_mosaic", granule_mosaic, mosaic),
    ]


def _write_log(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)


def _run_stage(
    *,
    key: str,
    command: list[str],
    expected: Path,
    log: dict,
    log_path: Path,
    dry_run: bool,
    resume: bool,
) -> bool:
    previous = (log.get("stages") or {}).get(key, {})
    command_text = shlex.join(command)
    if resume and previous.get("status") in {"success", "skipped"} and expected.exists():
        status = {
            **previous,
            "status": "skipped",
            "resume_reason": "prior success and expected output exists",
            "command": command,
            "command_shell": command_text,
        }
        log["stages"][key] = status
        _write_log(log_path, log)
        print(f"SKIP {key}")
        return True
    if dry_run:
        log["stages"][key] = {
            "status": "planned",
            "command": command,
            "command_shell": command_text,
            "expected_output": _rel(expected),
        }
        _write_log(log_path, log)
        print(f"PLAN {key}: {command_text}")
        return True
    started = datetime.now(timezone.utc).isoformat()
    tick = time.monotonic()
    print(f"RUN {key}: {command_text}", flush=True)
    try:
        completed = subprocess.run(command, cwd=ROOT, check=True)
        ok = expected.exists()
        if not ok:
            raise RuntimeError(f"command succeeded but expected output missing: {expected}")
        record = {
            "status": "success",
            "started_utc": started,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "duration_s": round(time.monotonic() - tick, 3),
            "returncode": completed.returncode,
            "command": command,
            "command_shell": command_text,
            "expected_output": _rel(expected),
        }
    except Exception as exc:  # noqa: BLE001
        record = {
            "status": "failed",
            "started_utc": started,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "duration_s": round(time.monotonic() - tick, 3),
            "command": command,
            "command_shell": command_text,
            "expected_output": _rel(expected),
            "error": str(exc),
            "returncode": getattr(exc, "returncode", None),
        }
        ok = False
        print(f"FAIL {key}: {exc}", flush=True)
    log["stages"][key] = record
    _write_log(log_path, log)
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--heatmap-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--cloud-method", choices=["omnicloudmask", "s2cloudless"], default="omnicloudmask")
    parser.add_argument("--cloud-device", default="cpu")
    parser.add_argument("--max-cloud-frac", type=float, default=0.15)
    parser.add_argument("--min-valid-frac", type=float, default=0.85)
    parser.add_argument("--max-stac-cloud", type=float, default=60.0)
    parser.add_argument("--include-shadow", action="store_true")
    parser.add_argument("--side", type=int, default=512)
    parser.add_argument("--min-land-frac", type=float, default=0.0)
    parser.add_argument("--min-clear", type=int, default=None)
    parser.add_argument("--iters", type=int, default=5000)
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--gpu-offset", type=int, default=0)
    parser.add_argument("--cross-crs", default="EPSG:3035")
    parser.add_argument(
        "--cross-out",
        type=Path,
        default=ROOT / "production" / "mosaics" / "norway_demo_sr_2p5m.tif",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=ROOT / "production" / "national_demo" / "run_log.json",
    )
    args = parser.parse_args()
    resolve = lambda path: path if path.is_absolute() else ROOT / path
    selection_path = resolve(args.selection_manifest)
    args.heatmap_dir = resolve(args.heatmap_dir)
    args.cross_out = resolve(args.cross_out)
    log_path = resolve(args.log)
    selection = json.loads(selection_path.read_text())
    args.date_range = selection["selection"]["date_range"]
    if args.min_clear is None:
        args.min_clear = int(selection["selection"]["min_clear"])
    selected = set(selection["selected_mgrs"])
    tiles = sorted(
        [tile for tile in selection["tiles"] if tile["mgrs_tile"] in selected],
        key=lambda tile: tile["mgrs_tile"],
    )
    if len(tiles) != len(selected):
        raise SystemExit("selection manifest tiles do not cover selected_mgrs")
    if log_path.is_file() and args.resume:
        log = json.loads(log_path.read_text())
    else:
        log = {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "selection_manifest": str(selection_path),
            "date_range": args.date_range,
            "dry_run": bool(args.dry_run),
            "stages": {},
        }
    log["dry_run"] = bool(args.dry_run)
    log["last_started_utc"] = datetime.now(timezone.utc).isoformat()
    _write_log(log_path, log)

    available_mosaics = []
    for tile in tiles:
        mgrs = tile["mgrs_tile"]
        prereq_ok = True
        for stage, command, expected in commands_for_tile(tile, args):
            key = f"{mgrs}:{stage}"
            if not prereq_ok:
                log["stages"][key] = {
                    "status": "blocked",
                    "reason": "earlier tile stage failed",
                    "command": command,
                    "command_shell": shlex.join(command),
                    "expected_output": _rel(expected),
                }
                _write_log(log_path, log)
                continue
            prereq_ok = _run_stage(
                key=key,
                command=command,
                expected=expected,
                log=log,
                log_path=log_path,
                dry_run=args.dry_run,
                resume=args.resume,
            )
            if not prereq_ok and not args.continue_on_error:
                raise SystemExit(f"stopped after {key}")
        mosaic = ROOT / "production" / "mosaics" / f"{mgrs}_sr_2p5m.tif"
        if args.dry_run or mosaic.is_file():
            available_mosaics.append(mosaic)

    if len(available_mosaics) >= 2:
        cross_cmd = [
            sys.executable,
            str(ROOT / "scripts" / "mosaic_national_sr.py"),
            "--sources",
            *(str(path) for path in available_mosaics),
            "--out",
            str(args.cross_out),
            "--dst-crs",
            args.cross_crs,
            "--resolution",
            "2.5",
        ]
        _run_stage(
            key="national:cross_granule_mosaic",
            command=cross_cmd,
            expected=args.cross_out,
            log=log,
            log_path=log_path,
            dry_run=args.dry_run,
            resume=args.resume,
        )
    else:
        log["stages"]["national:cross_granule_mosaic"] = {
            "status": "blocked",
            "reason": "fewer than two per-granule mosaics available",
        }
    log["finished_utc"] = datetime.now(timezone.utc).isoformat()
    _write_log(log_path, log)
    print(f"Run log -> {log_path}")


if __name__ == "__main__":
    main()
