#!/usr/bin/env python3
"""Patch size × within-frame K on the current LR512 recipe.

Windows are non-overlapping and must divide the 512 field. K is 1, 2, 4, and 8
when that many windows exist, plus one step that covers every window. The
128-pixel K=4 cell is the current recipe; an existing within-frame bench is
reused when its metrics are already on disk.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER7 = ("asker", "bergen", "rana", "tromso", "amli", "vennesla", "trondheim")
FIELD = 512
K_LADDER = (1, 2, 4, 8)


def grid() -> list[tuple[int, int, int]]:
    """Return (patch, K, n_windows) for every valid cell."""
    cells = []
    for patch in (64, 128, 256, 512):
        n = (FIELD // patch) ** 2
        ks = [k for k in K_LADDER if k <= n]
        if n not in ks:
            ks.append(n)
        for k in ks:
            cells.append((patch, k, n))
    return cells


def _run_name(patch: int, k: int, iters: int) -> str:
    return f"bench_p{patch}_k{k}_holdout_{iters // 1000}k"


def _metrics_path(city: str, patch: int, k: int, iters: int) -> Path:
    return (
        ROOT
        / "single_samples"
        / city
        / "sample"
        / _run_name(patch, k, iters)
        / "metrics.json"
    )


def _alias_path(city: str, patch: int, k: int, iters: int) -> Path | None:
    if patch == 128 and k == 4:
        return (
            ROOT
            / "single_samples"
            / city
            / "sample"
            / f"bench_within_k4_holdout_{iters // 1000}k"
            / "metrics.json"
        )
    return None


def _cmd(city: str, patch: int, k: int, iters: int) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "optimize.py"),
        "--dataset",
        city,
        "--s2-dir",
        str(ROOT / "data" / "s2_revisits" / f"{city}_lr512"),
        "--run_name",
        _run_name(patch, k, iters),
        "--recon_loss",
        "charbonnier",
        "--charbonnier_eps",
        "0.01",
        "--early_stop_metric",
        "holdout_mse",
        "--early_stop_patience",
        "8",
        "--early_stop_min_iters",
        "1000",
        "--eval_every",
        "200",
        "--iters",
        str(iters),
        "--device",
        "0",
        "--seed",
        "6",
        "--spatial_holdout",
        "0.1",
        "--holdout_block",
        "0",
        "--hr_render_tile",
        "2048",
        "--lr_stats_pixels",
        "all",
        "--lr_tile_halo",
        "-1",
        "--lr_degradation",
        "s2_psf_m",
        "--lr_tile_mix",
        "within",
        "--lr_tile",
        str(patch),
        "--lr_tiles_per_step",
        str(k),
        "--spatial_alignment_path",
        str(ROOT / "eval" / "spatial_alignment.json"),
        "--no_qgis_export",
    ]


def _summarize(city: str, patch: int, k: int, n_windows: int, mpath: Path, *, skipped: bool) -> dict:
    m = json.loads(mpath.read_text())
    es = m.get("early_stop") or {}
    return {
        "city": city,
        "patch": patch,
        "k": k,
        "n_windows": n_windows,
        "coverage": k / n_windows,
        "skipped": skipped,
        "lpips": m["lpips"]["model"],
        "psnr": m["psnr"]["model"],
        "ssim": m["ssim"]["model"],
        "completed_iters": m.get("completed_iters"),
        "best_iter": es.get("best_val_iter"),
        "stopped": es.get("stopped"),
        "training_time_s": m.get("training_time_seconds"),
        "peak_memory_gb": m.get("peak_memory_gb"),
        "lr_tile": m.get("lr_tile"),
        "lr_tiles_per_step": m.get("lr_tiles_per_step"),
        "lr_tile_mix": m.get("lr_tile_mix"),
        "metrics_path": str(mpath.relative_to(ROOT)),
    }


def _run_one(
    city: str,
    patch: int,
    k: int,
    n_windows: int,
    gpu: int,
    iters: int,
    skip_existing: bool,
) -> dict:
    mpath = _metrics_path(city, patch, k, iters)
    alias = _alias_path(city, patch, k, iters)
    if skip_existing and mpath.is_file():
        return _summarize(city, patch, k, n_windows, mpath, skipped=True)
    if skip_existing and alias is not None and alias.is_file():
        row = _summarize(city, patch, k, n_windows, alias, skipped=True)
        row["metrics_path"] = str(alias.relative_to(ROOT))
        return row
    print(f"[gpu{gpu}] {city} patch={patch} K={k}", flush=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    subprocess.run(_cmd(city, patch, k, iters), cwd=ROOT, check=True, env=env)
    return _summarize(city, patch, k, n_windows, mpath, skipped=False)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cities", nargs="+", default=["paper7"])
    p.add_argument("--iters", type=int, default=5000)
    p.add_argument("--gpus", nargs="+", type=int, required=True)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "single_samples" / "sweep_results" / "bench_patch_and_k.json",
    )
    args = p.parse_args()
    cities: list[str] = []
    for name in args.cities:
        cities.extend(PAPER7 if name == "paper7" else (name,))

    jobs = [
        (city, patch, k, n)
        for city in cities
        for patch, k, n in grid()
    ]
    rows: list[dict] = []
    failures: list[dict] = []
    # One queue per GPU. A shared pool was starting the next fit on a card
    # that still held the previous one, and Exclusive_Process rejected it.
    queues: dict[int, list[tuple[str, int, int, int]]] = {gpu: [] for gpu in args.gpus}
    for i, job in enumerate(jobs):
        queues[args.gpus[i % len(args.gpus)]].append(job)

    def _run_queue(gpu: int, gpu_jobs: list[tuple[str, int, int, int]]) -> tuple[list[dict], list[dict]]:
        q_rows: list[dict] = []
        q_failures: list[dict] = []
        for city, patch, k, n in gpu_jobs:
            try:
                row = _run_one(city, patch, k, n, gpu, args.iters, args.skip_existing)
                q_rows.append(row)
                print(
                    f"DONE {city} p{patch} k{k} lpips={row['lpips']:.4f} "
                    f"time={row['training_time_s']:.1f}s gpu={gpu}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001 — keep the rest of this GPU's queue running
                q_failures.append(
                    {"city": city, "patch": patch, "k": k, "gpu": gpu, "error": str(exc)}
                )
                print(f"FAIL {city} p{patch} k{k} gpu={gpu}: {exc}", flush=True)
        return q_rows, q_failures

    with ThreadPoolExecutor(max_workers=len(args.gpus)) as ex:
        futs = [ex.submit(_run_queue, gpu, gpu_jobs) for gpu, gpu_jobs in queues.items()]
        for fut in as_completed(futs):
            q_rows, q_failures = fut.result()
            rows.extend(q_rows)
            failures.extend(q_failures)

    rows.sort(key=lambda r: (r["patch"], r["k"], r["city"]))
    by_cell: dict[str, list[dict]] = {}
    for r in rows:
        by_cell.setdefault(f"p{r['patch']}_k{r['k']}", []).append(r)
    means = []
    for key, cell_rows in by_cell.items():
        n = len(cell_rows)
        means.append(
            {
                "cell": key,
                "patch": cell_rows[0]["patch"],
                "k": cell_rows[0]["k"],
                "n_windows": cell_rows[0]["n_windows"],
                "coverage": cell_rows[0]["coverage"],
                "n": n,
                "lpips": sum(r["lpips"] for r in cell_rows) / n,
                "psnr": sum(r["psnr"] for r in cell_rows) / n,
                "ssim": sum(r["ssim"] for r in cell_rows) / n,
                "time_s": sum(r["training_time_s"] or 0 for r in cell_rows) / n,
            }
        )
    means.sort(key=lambda r: (r["patch"], r["k"]))
    out = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cities": cities,
        "seed": 6,
        "iters_budget": args.iters,
        "recipe": "current within-frame K=4 flags; only patch and K change",
        "grid": [{"patch": a, "k": b, "n_windows": c} for a, b, c in grid()],
        "means": means,
        "rows": rows,
        "failures": failures,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(means, indent=2))
    print(f"Wrote {args.out}  failures={len(failures)}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
