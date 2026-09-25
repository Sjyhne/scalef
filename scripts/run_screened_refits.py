#!/usr/bin/env python3
"""Refit the confirmatory v5 jobs with a frame screen (``scripts/build_frame_screen.py``).

Each v5 job is rerun with its exact command plus ``--frame_screen`` and ``--force_base_date`` set
to the base date the v5 run used, so the only changes are the excluded frames and the
full-window masks. ``--num_samples`` becomes the number of kept frames. Jobs run one per GPU.
"""
from __future__ import annotations

import argparse
import json
import queue
import shlex
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V5_MANIFEST = ROOT / "paper/results/run_manifests/confirmatory_v5_boa__run_manifest.json"
V5_NS = "confirmatory_v5_boa"


def replace_arg(cmd: list[str], flag: str, value: str) -> list[str]:
    i = cmd.index(flag)
    return cmd[: i + 1] + [value] + cmd[i + 2:]


def build_jobs(screen_path: Path, namespace: str, stack_root: Path | None = None) -> list[dict]:
    """With ``stack_root``, fit the rebuilt stacks there (kept + replacement frames, masks set in
    their meta.json) instead of screening the v5 stacks at load time."""
    screen = json.loads(screen_path.read_text())["stacks"]
    jobs = []
    for job in json.loads(V5_MANIFEST.read_text())["jobs"]:
        v5_metrics = json.loads(Path(job["expected_metrics_path"]).read_text())
        base_date = v5_metrics["base_frame"]["date"]
        name = Path(job["s2_dir"]).name
        entry = screen[name]
        run_name = job["run_name"].replace(V5_NS, namespace)
        cmd = shlex.split(job["command_shell"])
        cmd = replace_arg(cmd, "--sample_id", namespace)
        cmd = replace_arg(cmd, "--run_name", run_name)
        if stack_root is None:
            n_frames = entry["n_frames"] - entry["n_excluded"]
            cmd += ["--frame_screen", str(screen_path)]
            s2_dir = job["s2_dir"]
        else:
            s2_dir = str(stack_root / name)
            n_frames = len(json.loads((stack_root / name / "meta.json").read_text())["frames"])
            cmd = replace_arg(cmd, "--s2-dir", s2_dir)
        cmd = replace_arg(cmd, "--num_samples", str(n_frames))
        cmd += ["--force_base_date", base_date]
        jobs.append({
            "job_id": run_name, "run_name": run_name, "city": job["city"], "seed": job["seed"],
            "v5_run_name": job["run_name"], "base_date": base_date, "s2_dir": s2_dir,
            "n_frames_v5": job["requested_frames"], "n_frames": n_frames, "excluded": entry["exclude"],
            "command": cmd,
            "expected_metrics_path": str(ROOT / "single_samples" / job["city"] / namespace / run_name / "metrics.json"),
        })
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--screen", type=Path, default=ROOT / "paper/results/frame_screen_v1_named.json")
    ap.add_argument("--namespace", default="confirmatory_v6_screen")
    ap.add_argument("--gpus", default="0,1,2,3,5,6")
    ap.add_argument("--stack_root", type=Path, default=None,
                    help="fit rebuilt stacks under this root (scripts/fetch_screened_replacements.py)")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    jobs = build_jobs(args.screen.resolve(), args.namespace, args.stack_root.resolve() if args.stack_root else None)
    log_dir = ROOT / "logs" / args.namespace
    log_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / f"paper/results/run_manifests/{args.namespace}__run_manifest.json"
    manifest = {"schema": "screened_refits_v1", "namespace": args.namespace, "parent": V5_NS,
                "screen": str(args.screen), "stack_root": str(args.stack_root) if args.stack_root else None,
                "created_utc": datetime.now(timezone.utc).isoformat(), "jobs": jobs}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    todo = [j for j in jobs if not Path(j["expected_metrics_path"]).is_file()]
    print(f"{len(jobs)} jobs, {len(todo)} to run; manifest {manifest_path}")
    if args.dry_run:
        print(shlex.join(todo[0]["command"]) if todo else "nothing to run")
        return

    q: queue.Queue = queue.Queue()
    for j in todo:
        q.put(j)
    lock = threading.Lock()

    def worker(gpu: str) -> None:
        while True:
            try:
                j = q.get_nowait()
            except queue.Empty:
                return
            cmd = replace_arg(j["command"], "--device", "0")
            env = {**__import__("os").environ, "CUDA_VISIBLE_DEVICES": gpu}
            t0 = time.time()
            with open(log_dir / f"{j['run_name']}.log", "w") as log:
                rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, env=env)
            ok = rc == 0 and Path(j["expected_metrics_path"]).is_file()
            with lock:
                print(f"[gpu{gpu}] {'DONE' if ok else 'FAIL'} {j['run_name']} rc={rc} {time.time() - t0:.0f}s "
                      f"left={q.qsize()}", flush=True)

    threads = [threading.Thread(target=worker, args=(g,)) for g in args.gpus.split(",")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    missing = [j["run_name"] for j in jobs if not Path(j["expected_metrics_path"]).is_file()]
    print(f"finished; missing {len(missing)}: {missing}")


if __name__ == "__main__":
    main()
