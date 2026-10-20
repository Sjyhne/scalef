#!/usr/bin/env python3
"""No-holdout refit of the remaining method controls.

Loss and sensor-blur arms (MAE, area pooling without the MTF pre-blur) use the
seven sites and seeds 6/7/8. The window-side factorial uses seed 6. The Asker
5 m and 1 m query rows retrain that site's three seeds only far enough to
decode the fitted field; the published named-site metrics are left in place.

The 128-pixel K=4 cell is the named-site fit and is not trained again.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_confirmatory_paper import generate_jobs, load_eval_manifest  # noqa: E402

NAMESPACE = "confirmatory_nocut"
EVAL_MANIFEST = ROOT / "eval" / "confirmatory_eval_manifest.v2.json"
NAMED_SCREEN = ROOT / "paper" / "results" / "frame_screen_v1_named_masks_only.json"
STATUS_PATH = ROOT / "single_samples" / "sweep_results" / "nocut_controls" / "status.json"
FAIL_LOG = ROOT / "logs" / "nocut_controls" / "failures.jsonl"
LOG_DIR = ROOT / "logs" / "nocut_controls" / "jobs"
DRIVER_LOG = ROOT / "logs" / "nocut_controls" / "driver.log"
PID_PATH = ROOT / "logs" / "nocut_controls" / "driver.pid"
PAPER7 = ("asker", "bergen", "rana", "tromso", "amli", "vennesla", "trondheim")
FIELD = 512

# Wall times from the earlier factorial, used only to start the slow fits first.
_PATCH_SECONDS = {
    (64, 1): 33, (64, 2): 33, (64, 4): 41, (64, 8): 60, (64, 64): 287,
    (128, 1): 38, (128, 2): 45, (128, 4): 76, (128, 8): 127, (128, 16): 229,
    (256, 1): 72, (256, 2): 117, (256, 4): 214,
    (512, 1): 213,
}


def _set_flag(argv: list[str], flag: str, value: str) -> None:
    if flag in argv:
        argv[argv.index(flag) + 1] = value
    else:
        argv.extend((flag, value))


def _named_command(template: list[str], city: str, seed: int, run_name: str, named_stacks: set[str]) -> list[str]:
    command = list(template)
    _set_flag(command, "--run_name", run_name)
    _set_flag(command, "--seed", str(seed))
    command[command.index("--lr_stats_pixels") + 1] = "clear"
    command.extend(["--hr_harmonize_pixels", "clear"])
    s2_dir = command[command.index("--s2-dir") + 1]
    if Path(s2_dir).name in named_stacks:
        command.extend(["--frame_screen", str(NAMED_SCREEN)])
    return command


def _metrics(city: str, run_name: str) -> str:
    return str(ROOT / "single_samples" / city / NAMESPACE / run_name / "metrics.json")


def _patch_cells() -> list[tuple[int, int, int]]:
    cells = []
    for patch in (64, 128, 256, 512):
        n_windows = (FIELD // patch) ** 2
        ks = [k for k in (1, 2, 4, 8) if k <= n_windows]
        if n_windows not in ks:
            ks.append(n_windows)
        for k in ks:
            cells.append((patch, k, n_windows))
    return cells


def _queue(named_stacks: set[str]) -> list[dict]:
    manifest = load_eval_manifest(EVAL_MANIFEST)
    bases = {
        (job.city, job.seed): job
        for job in generate_jobs(
            families=("b0_17",),
            seeds=(6, 7, 8),
            eval_manifest=manifest,
            eval_manifest_path=EVAL_MANIFEST,
            city_set="paper7",
            iters=5000,
            gpus=1,
            validate_data=True,
            namespace=NAMESPACE,
            s2_boa_offset="remove",
            spatial_holdout=0.0,
        )
    }
    rows: list[dict] = []
    for city in PAPER7:
        for seed in (6, 7, 8):
            template = bases[(city, seed)].command
            for config, flag, value in (
                ("mae", "--recon_loss", "mae"),
                ("area", "--lr_degradation", "area"),
            ):
                run_name = f"{NAMESPACE}__loss_blur__{config}__{city}__seed{seed}"
                command = _named_command(template, city, seed, run_name, named_stacks)
                _set_flag(command, flag, value)
                rows.append(
                    {
                        "id": run_name,
                        "group": "loss_blur",
                        "command": command,
                        "metrics": _metrics(city, run_name),
                        "cost": 80,
                    }
                )
        for patch, k, n_windows in _patch_cells():
            run_name = f"{NAMESPACE}__patch_k__p{patch}_k{k}__{city}__seed6"
            if patch == 128 and k == 4:
                published = bases[(city, 6)]
                rows.append(
                    {
                        "id": run_name,
                        "group": "patch_k",
                        "command": [],
                        "metrics": published.expected_metrics_path,
                        "cost": 0,
                        "alias_of": published.run_name,
                    }
                )
                continue
            command = _named_command(bases[(city, 6)].command, city, 6, run_name, named_stacks)
            _set_flag(command, "--lr_tile", str(patch))
            _set_flag(command, "--lr_tiles_per_step", str(k))
            rows.append(
                {
                    "id": run_name,
                    "group": "patch_k",
                    "command": command,
                    "metrics": _metrics(city, run_name),
                    "cost": _PATCH_SECONDS[(patch, k)],
                }
            )
    for seed in (6, 7, 8):
        run_name = f"{NAMESPACE}__gsd_query__df4_query__asker__seed{seed}"
        command = _named_command(bases[("asker", seed)].command, "asker", seed, run_name, named_stacks)
        _set_flag(command, "--query_gsd_m", "5,1")
        rows.append(
            {
                "id": run_name,
                "group": "gsd_query",
                "command": command,
                "metrics": _metrics("asker", run_name),
                "cost": 140,
            }
        )
    rows.sort(key=lambda row: -int(row["cost"]))
    return rows


def _with_device(command: list[str], gpu: int) -> list[str]:
    command = list(command)
    command[command.index("--device") + 1] = str(gpu)
    return command


def _write_status(payload: dict) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _detach() -> None:
    DRIVER_LOG.parent.mkdir(parents=True, exist_ok=True)
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
    PID_PATH.write_text(str(os.getpid()) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 7])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    if 0 in args.gpus:
        parser.error("GPU 0 is left off this queue")
    named_stacks = set(json.loads(NAMED_SCREEN.read_text())["stacks"])
    queue = _queue(named_stacks)
    pending = [row for row in queue if row["command"] and not Path(row["metrics"]).is_file()]
    print(
        f"{len(queue)} jobs, {len(queue) - len(pending)} already have metrics, "
        f"{len(pending)} to run on GPUs {args.gpus}",
        flush=True,
    )
    if args.dry_run:
        for group in ("loss_blur", "patch_k", "gsd_query"):
            sample = next(row for row in queue if row["group"] == group and row["command"])
            print(f"\n# {group} {sample['id']}")
            print(shlex.join(sample["command"]))
        aliased = [row for row in queue if row.get("alias_of")]
        print(f"\n{len(aliased)} patch cells reuse the named-site seed-6 fits")
        return
    if args.detach:
        _detach()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    FAIL_LOG.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    counts = {"completed": 0, "failed": 0, "skipped": len(queue) - len(pending)}
    work: list[dict] = list(pending)

    def worker(gpu: int) -> None:
        while True:
            with lock:
                if not work:
                    return
                row = work.pop(0)
            command = _with_device(row["command"], gpu)
            log_path = LOG_DIR / f"{row['id']}.log"
            started = datetime.now(timezone.utc).isoformat()
            try:
                with log_path.open("w", encoding="utf-8") as log:
                    proc = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                ok = proc.returncode == 0 and Path(row["metrics"]).is_file()
                result = "completed" if ok else f"failed_rc{proc.returncode}"
            except Exception:
                result = "failed_exc"
                log_path.write_text(traceback.format_exc(), encoding="utf-8")
            if result != "completed" and "busy or unavailable" in log_path.read_text(errors="replace"):
                with lock:
                    work.insert(0, row)
                print(f"gpu {gpu} busy, requeued {row['id']}", flush=True)
                time.sleep(30)
                continue
            with lock:
                if result == "completed":
                    counts["completed"] += 1
                else:
                    counts["failed"] += 1
                    with FAIL_LOG.open("a", encoding="utf-8") as handle:
                        handle.write(
                            json.dumps(
                                {"id": row["id"], "result": result, "started": started, "log": str(log_path)}
                            )
                            + "\n"
                        )
                done = counts["completed"] + counts["failed"] + counts["skipped"]
                print(f"[{done}/{len(queue)}] gpu {gpu} {row['id']}: {result}", flush=True)
                _write_status(
                    {
                        "updated_utc": datetime.now(timezone.utc).isoformat(),
                        "total": len(queue),
                        "remaining": len(work),
                        **counts,
                        "gpus": args.gpus,
                        "state": "running",
                    }
                )

    _write_status(
        {
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "total": len(queue),
            "remaining": len(work),
            **counts,
            "gpus": args.gpus,
            "state": "running",
        }
    )
    threads = [threading.Thread(target=worker, args=(gpu,), daemon=True) for gpu in args.gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    _write_status(
        {
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "total": len(queue),
            "remaining": 0,
            **counts,
            "gpus": args.gpus,
            "state": "finished",
        }
    )
    print(
        f"finished: {counts['completed']} completed, {counts['failed']} failed, "
        f"{counts['skipped']} skipped",
        flush=True,
    )
    if counts["failed"]:
        raise SystemExit(f"{counts['failed']} jobs failed; see {FAIL_LOG}")


if __name__ == "__main__":
    main()
