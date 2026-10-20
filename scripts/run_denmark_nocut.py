#!/usr/bin/env python3
"""Refit Denmark proper with no spatial hold-out.

Uses the frozen identity manifests (3,046 LR512 cells, shore tiles included).
Outputs use ``prod_k4_ocm_ovl5_nocut`` and do not replace the original or
base-keep mosaics. 32UMF is skipped when its nocut metrics and GeoTIFF already
match the identity date. Restart is safe.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IDENTITY = ROOT / "production/denmark_national_2025/identity"
RUNS = ROOT / "production/denmark_national_2025/runs"
LOG_DIR = ROOT / "logs/denmark_nocut"
DRIVER_LOG = LOG_DIR / "driver.log"
PID_PATH = LOG_DIR / "driver.pid"
STATUS_PATH = LOG_DIR / "status.json"
PREFIX = "prod_k4_ocm_ovl5_nocut"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _granules() -> list[tuple[str, Path, int]]:
    rows = []
    for manifest in sorted(IDENTITY.glob("*_ovl5/manifest.json")):
        payload = json.loads(manifest.read_text())
        mgrs = str(payload["parent"])
        rows.append((mgrs, manifest.parent, len(payload["tiles"])))
    rows.sort(key=lambda row: -row[2])
    return rows


def _write_status(payload: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(payload, indent=2) + "\n")


def _detach() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if os.fork() > 0:
        os._exit(0)
    os.setsid()
    if os.fork() > 0:
        os._exit(0)
    fd = os.open(str(DRIVER_LOG), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    if fd > 2:
        os.close(fd)
    PID_PATH.write_text(str(os.getpid()) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5, 6, 7])
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if any(gpu < 0 for gpu in args.gpus):
        parser.error("GPU ids must be non-negative")
    granules = _granules()
    n_tiles = sum(n for _, _, n in granules)
    print(f"{len(granules)} granules, {n_tiles} cells, GPUs {args.gpus}", flush=True)
    if args.dry_run:
        for mgrs, folder, n in granules:
            print(f"{mgrs} {n} {folder}")
        return
    if args.detach:
        _detach()

    offset = min(args.gpus)
    if args.gpus != list(range(offset, offset + len(args.gpus))):
        raise SystemExit("run_production.py needs a contiguous GPU range")
    state = {
        "started_utc": _now(),
        "gpus": args.gpus,
        "prefix": PREFIX,
        "n_granules": len(granules),
        "n_tiles": n_tiles,
        "granules": {},
        "state": "running",
    }
    _write_status(state)
    for mgrs, folder, n in granules:
        out = RUNS / f"production_{mgrs}_lr512_ovl5_nocut.json"
        log_path = LOG_DIR / f"{mgrs}.log"
        cmd = [
            sys.executable,
            str(ROOT / "scripts/run_production.py"),
            "--manifest",
            str(folder / "manifest.json"),
            "--iters",
            "5000",
            "--gpus",
            str(len(args.gpus)),
            "--gpu-offset",
            str(offset),
            "--skip-existing",
            "--run-prefix",
            PREFIX,
            "--max-base-cloud-frac",
            "0.02",
            "--identity-plan",
            str(folder / "identity_plan.json"),
            "--frame-screen",
            str(folder / "frame_screen.json"),
            "--s2-boa-offset",
            "remove",
            "--spatial-holdout",
            "0",
            "--out",
            str(out),
        ]
        print(f"[{_now()}] {mgrs} {n} cells", flush=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a") as handle:
            handle.write(f"\n# {_now()}\n")
            proc = subprocess.run(cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
        summary = {}
        if out.is_file():
            summary = json.loads(out.read_text()).get("aggregate") or {}
        state["granules"][mgrs] = {
            "utc": _now(),
            "returncode": proc.returncode,
            "n_tiles": n,
            "n_ok": summary.get("n_ok"),
            "n_fail": summary.get("n_fail"),
            "n_skipped": summary.get("n_skipped"),
        }
        state["current"] = mgrs
        _write_status(state)
        print(
            f"[{_now()}] {mgrs} rc={proc.returncode} "
            f"ok={summary.get('n_ok')} fail={summary.get('n_fail')} "
            f"skipped={summary.get('n_skipped')}",
            flush=True,
        )
    state["state"] = "finished"
    state["finished_utc"] = _now()
    _write_status(state)
    failed = [mgrs for mgrs, row in state["granules"].items() if row["returncode"] or row.get("n_fail")]
    print(f"finished; granules with failures: {failed or 'none'}", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
