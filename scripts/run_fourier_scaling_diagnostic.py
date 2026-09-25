#!/usr/bin/env python3
"""Focused diagnostic: Fourier degradation with increasing field size.

Sites Asker, Bergen, Rana; LR sides 64/256/512; seeds 6/7/8; full-field updates.
Every run goes to a common 5,000-step cap with no early termination (patience 0),
while the LR hold-out loss is still logged every 200 steps so the existing stopping
rule can be replayed afterwards. HR metrics are logged every 200 steps on the full
field and on the central LR64 footprint shared by all three sizes.

Encodings per size L:
  grid          the frozen grid configuration (tcnn, dense at these sizes)
  fourier_s{2,5,10}   existing settings: fixed scale in normalized [0,1) coordinates
  fourier_bw10  fixed physical bandwidth anchored at LR64 s=10: s(L) = 10 * L / 64
                (identical to fourier_s10 at LR64, so not rerun there)

The Fourier frequency matrix is drawn from a dedicated generator seeded with the run
seed, so each seed uses the same matrix at every size and scale.

The protocol otherwise matches the frozen 17-site runs (v2 alignment manifest, loss
cloud masks, whole-frame standardization).
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
NAMESPACE = "fourier_diag_v1"
SITES = ("asker", "bergen", "rana")
SIDES = (64, 256, 512)
SEEDS = (6, 7, 8)
ANCHOR_SIDE = 64
ANCHOR_SCALE = 10.0
EVAL_MANIFEST = ROOT / "eval/confirmatory_eval_manifest.v2.json"
RUN_MANIFEST = ROOT / "single_samples/sweep_results" / NAMESPACE / "run_manifest.json"
LOG_DIR = ROOT / "logs" / NAMESPACE
RUN_MANIFEST_DEFAULT = RUN_MANIFEST


def set_namespace(name: str) -> None:
    global NAMESPACE, RUN_MANIFEST, LOG_DIR
    NAMESPACE = name
    RUN_MANIFEST = ROOT / "single_samples/sweep_results" / name / "run_manifest.json"
    LOG_DIR = ROOT / "logs" / name

COMMON = [
    "--df", "4", "--scale_factor", "4",
    "--lr_degradation", "s2_psf_m",
    "--recon_loss", "charbonnier", "--charbonnier_eps", "0.01",
    "--lr_tile", "0", "--lr_tiles_per_step", "1",
    "--early_stop_metric", "holdout_mse",
    "--early_stop_patience", "0", "--early_stop_min_iters", "1000",
    "--eval_every", "200", "--force_hr_eval",
    "--spatial_holdout", "0.1", "--holdout_block", "0",
    "--hr_render_tile", "2048",
    "--lr_stats_pixels", "all",
    "--no_qgis_export",
]


def encodings(side: int) -> list[tuple[str, list[str]]]:
    out = [("grid", ["--input_projection", "hashgrid_tcnn"])]
    for s in (2, 5, 10):
        out.append((f"fourier_s{s}", ["--input_projection", "fourier", "--fourier_scale", str(s)]))
    if side != ANCHOR_SIDE:
        bw = ANCHOR_SCALE * side / ANCHOR_SIDE
        out.append(("fourier_bw10", ["--input_projection", "fourier", "--fourier_scale", f"{bw:g}"]))
    return out


def jobs(sites, sides, seeds, iters, only_encodings=None) -> list[dict]:
    out = []
    for side in sorted(sides, reverse=True):
        for site in sites:
            s2_dir = ROOT / "data/s2_revisits" / f"{site}_lr{side}"
            frames = len(json.loads((s2_dir / "meta.json").read_text())["frames"])
            off = (side - ANCHOR_SIDE) // 2
            for seed in seeds:
                for enc, flags in encodings(side):
                    if only_encodings and enc not in only_encodings:
                        continue
                    tag = "" if iters == 5000 else f"_it{iters}"
                    run = f"{NAMESPACE}__lr{side}_{enc}{tag}__{site}__seed{seed}"
                    cmd = [
                        sys.executable, str(ROOT / "optimize.py"),
                        "--dataset", site, "--sample_id", NAMESPACE,
                        "--s2-dir", str(s2_dir), "--run_name", run,
                        "--seed", str(seed), "--fourier_matrix_seed", str(seed),
                        "--iters", str(iters), "--num_samples", str(frames),
                        "--spatial_alignment_path", str(EVAL_MANIFEST),
                        "--eval_subwindow_lr", f"{off},{off},{ANCHOR_SIDE}",
                        "--eval_reference_s2_dir",
                        str(ROOT / "data" / "s2_revisits" / f"{site}_lr512"),
                        *COMMON, *flags,
                    ]
                    out.append({
                        "run_name": run, "site": site, "side": side, "seed": seed,
                        "encoding": enc, "command": cmd,
                        "metrics": str(ROOT / "single_samples" / site / NAMESPACE / run / "metrics.json"),
                    })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpus", type=int, nargs="+", default=[1, 2, 3, 5, 6])
    ap.add_argument("--sites", nargs="+", default=list(SITES))
    ap.add_argument("--sides", type=int, nargs="+", default=list(SIDES))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--encodings", nargs="+", default=None)
    ap.add_argument("--manifest", type=Path, default=RUN_MANIFEST)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--namespace", default=NAMESPACE)
    ap.add_argument("--s2-boa-offset", choices=("keep", "remove"), default="keep")
    args = ap.parse_args()
    set_namespace(args.namespace)
    if args.manifest == RUN_MANIFEST_DEFAULT:
        args.manifest = RUN_MANIFEST

    todo = jobs(args.sites, args.sides, args.seeds, args.iters, args.encodings)
    if args.s2_boa_offset != "keep":
        for j in todo:
            j["command"] += ["--s2_boa_offset", args.s2_boa_offset]
    run_manifest = args.manifest
    run_manifest.parent.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    status: dict[str, str] = {}

    def write_manifest() -> None:
        run_manifest.write_text(json.dumps({
            "namespace": NAMESPACE,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "eval_manifest": str(EVAL_MANIFEST),
            "jobs": [{**j, "status": status.get(j["run_name"], "planned")} for j in todo],
        }, indent=2))

    write_manifest()
    print(f"{len(todo)} jobs -> {run_manifest}")
    if args.dry_run:
        for j in todo[:3]:
            print(" ".join(j["command"]))
        return

    q: queue.Queue = queue.Queue()
    for j in todo:
        q.put(j)
    lock = threading.Lock()

    def worker(gpu: int) -> None:
        while True:
            try:
                j = q.get_nowait()
            except queue.Empty:
                return
            if args.skip_existing and Path(j["metrics"]).is_file():
                result = "skipped_existing"
            else:
                cmd = [*j["command"], "--device", "0"]
                env = {**__import__("os").environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
                with open(LOG_DIR / f"{j['run_name']}.log", "w") as log:
                    rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
                result = "completed" if rc == 0 and Path(j["metrics"]).is_file() else f"failed_rc{rc}"
            with lock:
                status[j["run_name"]] = result
                write_manifest()
                done = len(status)
            print(f"[{done}/{len(todo)}] gpu{gpu} {j['run_name']}: {result}", flush=True)

    threads = [threading.Thread(target=worker, args=(g,), daemon=True) for g in args.gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    failed = [k for k, v in status.items() if v.startswith("failed")]
    print(f"done; {len(failed)} failed")


if __name__ == "__main__":
    main()
