#!/usr/bin/env python3
"""Denmark proper, frozen 5% production recipe, GPUs 0–3.

Faroe Islands and Greenland are not in the inventory. Each MGRS granule is
planned and fetched on its own, then OmniCloudMask, 5% tiling, admission, and
the fit run on GPUs 0–3. Later granules download while an earlier granule
trains. The mosaic ramps across the overlap and does not use the GPU.

Restart is safe: finished stages are skipped, and fits use --skip-existing.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "configs/countries/denmark.json"
INVENTORY = ROOT / "production/country_inventory/denmark/mgrs_list.derived.json"
LAND = ROOT / "production/country_inventory/denmark/denmark_land_outline_ne10m.geojson"
S2_ROOT = ROOT / "data/s2_revisits/denmark_national_2025"
PROD = ROOT / "production/denmark_national_2025"
STATUS = PROD / "status"
PREFIX = "prod_k4_ocm_ovl5"
CENTER = "2025-07-15"


def _selected_gpus() -> tuple[int, ...]:
    raw = os.environ.get("DENMARK_GPUS", "0,1,2,3")
    gpus = tuple(int(part) for part in raw.split(",") if part.strip())
    if not gpus:
        raise SystemExit("DENMARK_GPUS is empty")
    return gpus


GPUS = _selected_gpus()

_status_lock = threading.Lock()
_state: dict = {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log(msg: str) -> None:
    print(f"[{_now()}] {msg}", flush=True)


def _save_state() -> None:
    STATUS.mkdir(parents=True, exist_ok=True)
    payload = dict(_state)
    payload["updated_utc"] = _now()
    (STATUS / "state.json").write_text(json.dumps(payload, indent=2) + "\n")


def _mark(mgrs: str, stage: str, **extra) -> None:
    with _status_lock:
        row = _state.setdefault("granules", {}).setdefault(mgrs, {})
        row[stage] = {"utc": _now(), **extra}
        _state["current"] = {"mgrs": mgrs, "stage": stage}
        _save_state()


def _run(cmd: list[str], *, log_name: str) -> None:
    log_path = STATUS / "logs" / log_name
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _log(f"+ {' '.join(cmd)}")
    with log_path.open("a") as fh:
        fh.write(f"\n# {_now()}\n+ {' '.join(cmd)}\n")
        fh.flush()
        proc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"exit {proc.returncode}: {log_name} (see {log_path})")


def _stamp(mgrs: str, stage: str) -> Path:
    return STATUS / f"{mgrs}.{stage}"


def _done(mgrs: str, stage: str) -> bool:
    return _stamp(mgrs, stage).is_file()


def _finish(mgrs: str, stage: str) -> None:
    path = _stamp(mgrs, stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_now() + "\n")
    _mark(mgrs, stage, ok=True)


def granule_order() -> list[str]:
    inv = json.loads(INVENTORY.read_text())
    fractions = inv["per_tile_land_fraction"]
    return sorted(inv["mgrs"], key=lambda tile: (-float(fractions[tile]), tile))


def fetch_one(mgrs: str) -> None:
    if _done(mgrs, "fetched"):
        _log(f"{mgrs}: fetch already done")
        return
    _mark(mgrs, "fetch", ok=False)
    _run(
        [
            sys.executable,
            str(ROOT / "scripts/fetch_national_mgrs.py"),
            "--mgrs",
            mgrs,
            "--country-config",
            str(CONFIG),
            "--date",
            CENTER,
        ],
        log_name=f"{mgrs}.fetch.log",
    )
    meta = S2_ROOT / mgrs / "meta.json"
    ondisk = PROD / "plans" / mgrs / "plan_ondisk.json"
    if not meta.is_file() or not ondisk.is_file():
        raise RuntimeError(f"{mgrs}: fetch finished without {meta} and {ondisk}")
    _finish(mgrs, "fetched")


def ocm_one(mgrs: str) -> None:
    if _done(mgrs, "ocm"):
        _log(f"{mgrs}: OmniCloudMask already done")
        return
    _mark(mgrs, "ocm", ok=False)
    s2 = S2_ROOT / mgrs
    procs = []
    for gpu in GPUS:
        log_path = STATUS / "logs" / f"{mgrs}.ocm.gpu{gpu}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(ROOT / "scripts/ocm_granules.py"),
            "--dirs",
            str(s2),
            "--frame-shard",
            f"{gpu}/{len(GPUS)}",
            "--device",
            f"cuda:{gpu}",
            "--log",
            str(STATUS / "logs" / f"{mgrs}.ocm.gpu{gpu}.json"),
        ]
        _log(f"+ {' '.join(cmd)}")
        fh = log_path.open("a")
        procs.append(
            (
                gpu,
                fh,
                subprocess.Popen(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT),
            )
        )
    failed = []
    for gpu, fh, proc in procs:
        code = proc.wait()
        fh.close()
        if code != 0:
            failed.append(f"gpu {gpu} exit {code}")
    if failed:
        raise RuntimeError(f"{mgrs}: OmniCloudMask failed: {', '.join(failed)}")
    _finish(mgrs, "ocm")


def tile_one(mgrs: str) -> Path:
    manifest = S2_ROOT / mgrs / "granule_tiles_lr512_ovl5_manifest.json"
    if _done(mgrs, "tiled") and manifest.is_file():
        _log(f"{mgrs}: tiles already written")
        return manifest
    _mark(mgrs, "tile", ok=False)
    _run(
        [
            sys.executable,
            str(ROOT / "scripts/make_granule_tiles.py"),
            "--src",
            str(S2_ROOT / mgrs),
            "--out-root",
            str(S2_ROOT),
            "--side",
            "512",
            "--overlap-frac",
            "0.05",
            "--manifest",
            str(manifest),
            "--land-mask",
            str(LAND),
            "--min-land-frac",
            "0.0",
            "--cell-plan",
            str(PROD / "plans" / mgrs / "plan_ondisk.json"),
        ],
        log_name=f"{mgrs}.tile.log",
    )
    _finish(mgrs, "tiled")
    return manifest


def prepare_one(mgrs: str, manifest: Path) -> Path | None:
    out = PROD / "identity" / f"{mgrs}_ovl5"
    prepared = out / "manifest.json"
    if _done(mgrs, "prepared") and prepared.is_file():
        _log(f"{mgrs}: admission already done")
        return prepared
    _mark(mgrs, "prepare", ok=False)
    try:
        _run(
            [
                sys.executable,
                str(ROOT / "scripts/prepare_production_ocm_block.py"),
                "--parent",
                mgrs,
                "--manifest",
                str(manifest),
                "--center",
                CENTER,
                "--out-dir",
                str(out),
            ],
            log_name=f"{mgrs}.prepare.log",
        )
    except RuntimeError as exc:
        text = (STATUS / "logs" / f"{mgrs}.prepare.log").read_text(errors="replace")
        if "no cell has a frame under the admission rule" in text:
            _log(f"{mgrs}: no cell passed admission; skipping the fit")
            _mark(mgrs, "prepared", ok=True, n_tiles=0, note="admission_empty")
            _stamp(mgrs, "prepared").write_text(_now() + "\n")
            _stamp(mgrs, "trained").write_text("admission_empty\n")
            return None
        raise exc
    n_tiles = len(json.loads(prepared.read_text()).get("tiles") or [])
    _mark(mgrs, "prepared", ok=True, n_tiles=n_tiles)
    _stamp(mgrs, "prepared").write_text(_now() + "\n")
    if n_tiles == 0:
        _stamp(mgrs, "trained").write_text("no tiles\n")
        return None
    return prepared


def ensure_baselines(mgrs: str) -> None:
    """Fill processing baselines for products this granule downloaded.

    Offset removal reads data/s2_revisits/processing_baselines.json. A fetch that
    predates that cache leaves every fit failing before the first step.
    """
    _run(
        [
            sys.executable,
            str(ROOT / "scripts/cache_processing_baselines.py"),
            str(S2_ROOT),
        ],
        log_name=f"{mgrs}.baselines.log",
    )


def train_one(mgrs: str, prepared: Path) -> None:
    summary = PROD / "runs" / f"production_{mgrs}_lr512_ovl5.json"
    if _done(mgrs, "trained"):
        _log(f"{mgrs}: fit already done")
        return
    _mark(mgrs, "train", ok=False)
    ensure_baselines(mgrs)
    summary.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            sys.executable,
            str(ROOT / "scripts/run_production.py"),
            "--manifest",
            str(prepared),
            "--iters",
            "5000",
            "--gpus",
            str(len(GPUS)),
            "--gpu-offset",
            str(GPUS[0]),
            "--skip-existing",
            "--run-prefix",
            PREFIX,
            "--max-base-cloud-frac",
            "0.02",
            "--identity-plan",
            str(prepared.parent / "identity_plan.json"),
            "--frame-screen",
            str(prepared.parent / "frame_screen.json"),
            "--s2-boa-offset",
            "remove",
            "--out",
            str(summary),
        ],
        log_name=f"{mgrs}.train.log",
    )
    agg = json.loads(summary.read_text())["aggregate"]
    _mark(
        mgrs,
        "train_result",
        n_ok=agg.get("n_ok"),
        n_fail=agg.get("n_fail"),
        sum_train_s=agg.get("sum_train_s"),
    )
    n_fail = int(agg.get("n_fail") or 0)
    if n_fail > 0:
        raise RuntimeError(f"{mgrs}: {n_fail} fit(s) failed; not marking the granule trained")
    _finish(mgrs, "trained")


def mosaic_one(mgrs: str, prepared: Path) -> None:
    out = PROD / "mosaics" / f"{mgrs}_ovl5_sr_2p5m.tif"
    if _done(mgrs, "mosaic") and out.is_file():
        _log(f"{mgrs}: mosaic already written")
        return
    _mark(mgrs, "mosaic", ok=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            sys.executable,
            str(ROOT / "scripts/mosaic_granule_sr.py"),
            "--manifest",
            str(prepared),
            "--run-prefix",
            PREFIX,
            "--method",
            "feather",
            "--out",
            str(out),
            "--allow-missing",
        ],
        log_name=f"{mgrs}.mosaic.log",
    )
    _finish(mgrs, "mosaic")


def _fetch_worker(order: list[str], ready: queue.Queue) -> None:
    for mgrs in order:
        try:
            fetch_one(mgrs)
            ready.put((mgrs, None))
        except Exception as exc:  # noqa: BLE001 - keep the rest of the country moving
            _log(f"{mgrs}: FETCH FAILED: {exc}")
            _mark(mgrs, "fetch_error", error=str(exc))
            ready.put((mgrs, exc))
    ready.put(None)


def _mosaic_worker(jobs: queue.Queue, failures: list[str]) -> None:
    while True:
        item = jobs.get()
        if item is None:
            return
        mgrs, prepared = item
        try:
            mosaic_one(mgrs, prepared)
        except Exception as exc:  # noqa: BLE001
            _log(f"{mgrs}: MOSAIC FAILED: {exc}")
            _mark(mgrs, "mosaic_error", error=str(exc))
            failures.append(mgrs)


def _claim_driver() -> None:
    STATUS.mkdir(parents=True, exist_ok=True)
    pid_path = STATUS / "driver.pid"
    if pid_path.is_file():
        old = int(pid_path.read_text().strip() or "0")
        if old > 0 and Path(f"/proc/{old}").exists():
            raise SystemExit(f"Denmark driver already running as pid {old}")
    pid_path.write_text(str(os.getpid()) + "\n")


def main() -> None:
    _claim_driver()
    order = granule_order()
    _state.clear()
    _state.update(
        {
            "recipe": {
                "overlap_frac": 0.05,
                "stride": 486,
                "join": "feather across the overlap",
                "center": CENTER,
                "run_prefix": PREFIX,
                "gpus": list(GPUS),
                "s2_boa_offset": "remove",
                "admission": "thick+thin <= 0.15 and valid >= 0.85",
                "loss_mask": "omnicloudmask thick+thin+shadow",
            },
            "order": order,
            "started_utc": _now(),
            "granules": {},
        }
    )
    _save_state()
    _log(f"Denmark ovl5: {len(order)} granules, GPUs {list(GPUS)}, largest land first")
    ready: queue.Queue = queue.Queue()
    mosaics: queue.Queue = queue.Queue()
    mosaic_failures: list[str] = []
    threading.Thread(target=_fetch_worker, args=(order, ready), daemon=True).start()
    mosaic_thread = threading.Thread(
        target=_mosaic_worker, args=(mosaics, mosaic_failures), daemon=True
    )
    mosaic_thread.start()

    n_fail = 0
    while True:
        item = ready.get()
        if item is None:
            break
        mgrs, err = item
        if err is not None:
            n_fail += 1
            continue
        try:
            ocm_one(mgrs)
            manifest = tile_one(mgrs)
            prepared = prepare_one(mgrs, manifest)
            if prepared is None:
                continue
            train_one(mgrs, prepared)
            mosaics.put((mgrs, prepared))
        except Exception as exc:  # noqa: BLE001
            n_fail += 1
            _log(f"{mgrs}: FAILED: {exc}")
            _mark(mgrs, "error", error=str(exc))
    mosaics.put(None)
    mosaic_thread.join()
    n_fail += len(mosaic_failures)
    _state["finished_utc"] = _now()
    _state["n_granule_failures"] = n_fail
    _save_state()
    _log(f"Denmark ovl5 finished with {n_fail} granule failures")
    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
