#!/usr/bin/env python3
"""K=4 partial updates vs full coverage at LR512, separated from the stopping rule.

Runs K=4 LR128 windows on Asker/Bergen/Rana, seeds 6/7/8, to the common 5,000-step cap
with no early termination and HR metrics every 200 steps, under exactly the protocol of
``run_fourier_scaling_diagnostic.py``. The full-coverage arm is that diagnostic's
``lr512_grid`` runs (same inputs, masks, flags; ``--lr_tile 0``).

Arms: ``k4`` (windows with the PSF blur halo, the corrected default) and
``k4_nohalo`` (``--lr_tile_halo 0``, the window forward used by all earlier K=4 runs).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from run_fourier_scaling_diagnostic import COMMON, EVAL_MANIFEST  # noqa: E402

NAMESPACE = "update_eff_v1"
LOG_DIR = ROOT / "logs" / NAMESPACE
ARMS = {
    "k4": ["--lr_tile_halo", "-1"],
    "k4_nohalo": ["--lr_tile_halo", "0"],
}


def command(site: str, seed: int, arm: str, iters: int) -> tuple[str, list[str], Path]:
    s2_dir = ROOT / "data/s2_revisits" / f"{site}_lr512"
    frames = len(json.loads((s2_dir / "meta.json").read_text())["frames"])
    run = f"{NAMESPACE}__lr512_{arm}__{site}__seed{seed}"
    flags = [f for f in COMMON]
    i = flags.index("--lr_tile")
    flags[i + 1] = "128"
    j = flags.index("--lr_tiles_per_step")
    flags[j + 1] = "4"
    cmd = [
        sys.executable, str(ROOT / "optimize.py"),
        "--dataset", site, "--sample_id", NAMESPACE, "--s2-dir", str(s2_dir),
        "--run_name", run, "--seed", str(seed), "--iters", str(iters),
        "--num_samples", str(frames), "--spatial_alignment_path", str(EVAL_MANIFEST),
        "--eval_subwindow_lr", "224,224,64", "--input_projection", "hashgrid_tcnn",
        "--lr_tile_mix", "within", *flags, *ARMS[arm], "--device", "0",
    ]
    return run, cmd, ROOT / "single_samples" / site / NAMESPACE / run / "metrics.json"


def main() -> None:
    global NAMESPACE, LOG_DIR
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--sites", nargs="+", default=["asker", "bergen", "rana"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[6, 7, 8])
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--namespace", default=NAMESPACE)
    ap.add_argument("--s2-boa-offset", choices=("keep", "remove"), default="keep")
    args = ap.parse_args()
    NAMESPACE, LOG_DIR = args.namespace, ROOT / "logs" / args.namespace
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(args.gpu)}
    todo = [(s, seed, a) for s in args.sites for seed in args.seeds for a in args.arms]
    for n, (site, seed, arm) in enumerate(todo, 1):
        run, cmd, metrics = command(site, seed, arm, args.iters)
        if args.s2_boa_offset != "keep":
            cmd += ["--s2_boa_offset", args.s2_boa_offset]
        if args.skip_existing and metrics.is_file():
            print(f"[{n}/{len(todo)}] {run}: skipped_existing", flush=True)
            continue
        with open(LOG_DIR / f"{run}.log", "w") as log:
            rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        ok = rc == 0 and metrics.is_file()
        print(f"[{n}/{len(todo)}] {run}: {'completed' if ok else f'failed_rc{rc}'}", flush=True)


if __name__ == "__main__":
    main()
