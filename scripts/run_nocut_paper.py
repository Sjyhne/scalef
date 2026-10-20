#!/usr/bin/env python3
"""No-holdout refit of the paper fits. Does not touch Denmark production.

Confirmatory families use the current named-site protocol: eval manifest v2,
surface-reflectance offset removed, and the LR512 OmniCloudMask frame screen
where that stack is in the screen. Encoding stacks at LR 64/128/256 are not in
that screen, so they keep the cloud mask already stored in the stack.

The nested ladder follows the clear_v4 protocol (nested mask screen, clear-pixel
stats, harmonize on clear pixels, base date pinned from clear_v4) with the
block holdout removed. Run tag ``nocut`` and namespace ``confirmatory_nocut``
do not overwrite the reported fits.

GPUs are the physical device ids passed to ``--device``. GPU 0 is left off the
default list because it has been unavailable to this process.
"""

from __future__ import annotations

import argparse
import json
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
from scripts.run_nested_float_validation import jobs as nested_jobs  # noqa: E402
from scripts.run_nested_float_validation import select_parents  # noqa: E402

NAMESPACE = "confirmatory_nocut"
NEST_TAG = "nocut"
EVAL_MANIFEST = ROOT / "eval" / "confirmatory_eval_manifest.v2.json"
NAMED_SCREEN = ROOT / "paper" / "results" / "frame_screen_v1_named_masks_only.json"
NEST_SCREEN = ROOT / "paper" / "results" / "frame_screen_nested_masks_only.json"
STATUS_PATH = ROOT / "single_samples" / "sweep_results" / "nocut_paper" / "status.json"
FAIL_LOG = ROOT / "logs" / "nocut_paper" / "failures.jsonl"
LOG_DIR = ROOT / "logs" / "nocut_paper" / "jobs"

# k=4 on the seven paper cities is the same fit as the 17-site headline.
# joint repeats that headline. df4_query retrains the df=4 field to decode it.
CONFIRMATORY = (
    ("b0_17", None, "b0_17"),
    ("fixed_k", ("lr128_k1", "lr128_k2", "lr128_k8", "lr128_full"), "paper7"),
    ("encoding_size", None, "paper7"),
    ("misr_controls", ("frozen_affines", "frozen_radiometry"), "paper7"),
    ("gsd_ladder", ("df2_5m",), "paper7"),
)


def _with_device(command: list[str], gpu: int) -> list[str]:
    command = list(command)
    command[command.index("--device") + 1] = str(gpu)
    return command


def _confirmatory_queue(named_stacks: set[str]) -> list[dict]:
    manifest = load_eval_manifest(EVAL_MANIFEST)
    rows: list[dict] = []
    for family, configs, city_set in CONFIRMATORY:
        jobs = generate_jobs(
            families=(family,),
            seeds=(6, 7, 8),
            eval_manifest=manifest,
            eval_manifest_path=EVAL_MANIFEST,
            city_set=city_set,
            configs=configs,
            iters=5000,
            gpus=1,
            gpu_offset=0,
            validate_data=True,
            namespace=NAMESPACE,
            s2_boa_offset="remove",
            spatial_holdout=0.0,
        )
        for job in jobs:
            command = list(job.command)
            # v7 named-site protocol: stats and histogram match use clear pixels.
            command[command.index("--lr_stats_pixels") + 1] = "clear"
            command.extend(["--hr_harmonize_pixels", "clear"])
            if Path(job.s2_dir).name in named_stacks:
                command.extend(["--frame_screen", str(NAMED_SCREEN)])
            rows.append(
                {
                    "id": job.job_id,
                    "group": "confirmatory",
                    "command": command,
                    "metrics": job.expected_metrics_path,
                }
            )
    return rows


def _nested_queue(nest_stacks: set[str]) -> list[dict]:
    parents = select_parents(all_parents=True)
    built = nested_jobs(
        parents,
        NEST_TAG,
        NEST_SCREEN,
        "clear_v4",
        {
            "--spatial_holdout": "0",
            "--early_stop_patience": "0",
            "--lr_stats_pixels": "clear",
            "--hr_harmonize_pixels": "clear",
        },
    )
    rows = []
    missing = []
    for job in built:
        command = list(job["command"])
        s2_dir = command[command.index("--s2-dir") + 1]
        name = Path(s2_dir).name
        if name not in nest_stacks:
            missing.append(name)
        rows.append(
            {
                "id": f"nested_{job['side']}_{job['tile_id']}",
                "group": "nested",
                "command": command,
                "metrics": job["metrics"],
            }
        )
    if missing:
        sample = ", ".join(missing[:8])
        raise SystemExit(
            f"{len(missing)} nested stacks are absent from the mask screen, e.g. {sample}"
        )
    return rows


def _write_status(payload: dict) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 7])
    parser.add_argument(
        "--jobs-per-gpu",
        type=int,
        default=1,
        help="Concurrent fits per GPU. Small tiles leave the card mostly idle, so 2 overlaps startup with training.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--confirmatory-only",
        action="store_true",
        help="Skip the nested ladder.",
    )
    parser.add_argument(
        "--nested-only",
        action="store_true",
        help="Skip the named-site and ablation families.",
    )
    args = parser.parse_args()
    if args.jobs_per_gpu < 1:
        parser.error("--jobs-per-gpu must be at least 1")
    if args.confirmatory_only and args.nested_only:
        raise SystemExit("choose at most one of --confirmatory-only and --nested-only")

    named_stacks = set(json.loads(NAMED_SCREEN.read_text())["stacks"])
    nest_stacks = set(json.loads(NEST_SCREEN.read_text())["stacks"])
    queue: list[dict] = []
    if not args.nested_only:
        queue.extend(_confirmatory_queue(named_stacks))
    if not args.confirmatory_only:
        queue.extend(_nested_queue(nest_stacks))

    pending = [row for row in queue if not Path(row["metrics"]).is_file()]
    print(
        f"{len(queue)} jobs, {len(queue) - len(pending)} already have metrics, "
        f"{len(pending)} to run on GPUs {args.gpus} ({args.jobs_per_gpu} each)",
        flush=True,
    )
    if args.dry_run:
        for group in ("confirmatory", "nested"):
            sample = next(row for row in queue if row["group"] == group)
            print(f"\n# {group} {sample['id']}")
            print(shlex.join(sample["command"]))
        return

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
                    proc = subprocess.run(
                        command,
                        cwd=ROOT,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
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
                                {
                                    "id": row["id"],
                                    "result": result,
                                    "started": started,
                                    "log": str(log_path),
                                }
                            )
                            + "\n"
                        )
                done = counts["completed"] + counts["failed"] + counts["skipped"]
                if result != "completed" or done % 25 == 0:
                    print(
                        f"[{done}/{len(queue)}] gpu {gpu} {row['id']}: {result}",
                        flush=True,
                    )
                    _write_status(
                        {
                            "updated_utc": datetime.now(timezone.utc).isoformat(),
                            "total": len(queue),
                            "remaining": len(work),
                            **counts,
                            "gpus": args.gpus,
                            "namespace": NAMESPACE,
                            "nested_tag": NEST_TAG,
                        }
                    )

    _write_status(
        {
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "total": len(queue),
            "remaining": len(work),
            **counts,
            "gpus": args.gpus,
            "namespace": NAMESPACE,
            "nested_tag": NEST_TAG,
            "state": "running",
        }
    )
    slots = [gpu for gpu in args.gpus for _ in range(args.jobs_per_gpu)]
    threads = [threading.Thread(target=worker, args=(gpu,), daemon=True) for gpu in slots]
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
            "namespace": NAMESPACE,
            "nested_tag": NEST_TAG,
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
