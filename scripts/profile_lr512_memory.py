#!/usr/bin/env python3
"""Attribute LR512 process-level GPU memory to training, periodic HR evaluation, and final render.

Runs short Asker LR512 fits with the production command, varying one memory-relevant
setting at a time, and samples per-process GPU memory through nvidia-smi once a second.
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "paper" / "results" / "run_manifests" / "confirmatory_v3_halo__run_manifest.json"
OUT = ROOT / "paper" / "results" / "lr512_memory_profile.json"
ITERS = 600


def base_command(city: str = "asker") -> list[str]:
    jobs = json.loads(MANIFEST.read_text())["jobs"]
    job = next(j for j in jobs if j["city"] == city and j["seed"] == 6)
    return list(job["command"])


def set_arg(cmd: list[str], flag: str, value: str | None) -> list[str]:
    cmd = list(cmd)
    if flag in cmd:
        i = cmd.index(flag)
        takes_value = i + 1 < len(cmd) and not cmd[i + 1].startswith("--")
        del cmd[i:i + (2 if takes_value else 1)]
    if value is not None:
        cmd += [flag] if value == "" else [flag, value]
    return cmd


VARIANTS = {
    "production_k4": {},
    "k4_no_periodic_hr_eval": {"--force_hr_eval": None},
    "k4_no_periodic_hr_eval_tiled_render": {"--force_hr_eval": None, "--hr_render_tile": "512"},
    "full_field_no_periodic_hr_eval_tiled_render": {"--force_hr_eval": None, "--hr_render_tile": "512",
                                                     "--lr_tiles_per_step": "16"},
}


def peak_memory(proc: subprocess.Popen, samples: list[int]) -> None:
    while proc.poll() is None:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
        for line in out.splitlines():
            pid, mem = (s.strip() for s in line.split(","))
            if pid == str(proc.pid):
                samples.append(int(mem))
        time.sleep(1.0)


def run(name: str, overrides: dict, gpu: int) -> dict:
    cmd = base_command()
    cmd = set_arg(cmd, "--iters", str(ITERS))
    cmd = set_arg(cmd, "--device", "0")
    cmd = set_arg(cmd, "--sample_id", "memprofile_v1")
    cmd = set_arg(cmd, "--run_name", f"memprofile_v1__{name}")
    for flag, value in overrides.items():
        cmd = set_arg(cmd, flag, value)
    log = ROOT / "logs" / f"memprofile_v1__{name}.log"
    samples: list[int] = []
    with log.open("w") as fh:
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": str(gpu)})
        watcher = threading.Thread(target=peak_memory, args=(proc, samples), daemon=True)
        watcher.start()
        proc.wait()
        watcher.join(timeout=5)
    metrics = next((ROOT / "single_samples").glob(f"asker/memprofile_v1/memprofile_v1__{name}/metrics.json"), None)
    torch_peaks = {}
    if metrics is not None:
        m = json.loads(metrics.read_text())

        def walk(d):
            for k, v in d.items():
                if isinstance(v, dict):
                    walk(v)
                elif k in ("torch_peak_allocated_gb", "torch_peak_reserved_gb"):
                    torch_peaks[k] = v
        walk(m)
    return {"variant": name, "overrides": overrides, "returncode": proc.returncode,
            "process_peak_gib": max(samples) / 1024 if samples else None,
            "process_median_gib": sorted(samples)[len(samples) // 2] / 1024 if samples else None,
            "n_samples": len(samples), **torch_peaks}


def main() -> None:
    gpus = [int(g) for g in sys.argv[1].split(",")] if len(sys.argv) > 1 else [1, 5, 3, 2]
    results: list[dict] = []
    threads = []
    for (name, overrides), gpu in zip(VARIANTS.items(), gpus):
        t = threading.Thread(target=lambda n=name, o=overrides, g=gpu: results.append(run(n, o, g)))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    results.sort(key=lambda r: list(VARIANTS).index(r["variant"]))
    OUT.write_text(json.dumps({"iters": ITERS, "site": "asker", "lr_side": 512, "results": results}, indent=1) + "\n")
    for r in results:
        print(json.dumps(r))


if __name__ == "__main__":
    main()
