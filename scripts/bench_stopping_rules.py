#!/usr/bin/env python3
"""Compare LR-only early-stop rules against oracle LPIPS (offline + optional runs).

Offline mode replays holdout curves from ``early_stop_validation/*_analysis.json``
(or raw ``metrics.json`` with aligned val + LPIPS) under patience / EMA /
regression grids.

Optional live mode launches production-knob k4 runs with ``--force_hr_eval`` and
``patience=0`` so new trajectories can be scored the same way.

Example
-------
    python scripts/bench_stopping_rules.py
    python scripts/bench_stopping_rules.py --run --cities asker bergen --gpus 2
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.stopping_analysis import analyze_stopping_trajectory  # noqa: E402

DEFAULT_CITIES = ["asker", "bergen", "rana", "tromso", "trondheim", "amli", "vennesla"]
VAL_DIR = ROOT / "single_samples" / "sweep_results" / "early_stop_validation"


def _load_metrics_from_analysis(path: Path) -> dict:
    """Rebuild a metrics-shaped dict from a saved analysis trajectory."""
    a = json.loads(path.read_text())
    traj = a["trajectory"]
    return {
        "completed_iters": a.get("completed_iters"),
        "training": {
            "history": {
                "iterations": traj["iterations"],
                "val_loss": traj["val_loss"],
                "model_lpips": traj["model_lpips"],
                "psnr": traj.get("psnr") or [],
            }
        },
    }


def _rule_grid() -> list[dict]:
    return [
        {"label": "p3_raw", "patience": 3, "ema_alpha": 0.0, "max_regression": 0.0},
        {"label": "p8_raw", "patience": 8, "ema_alpha": 0.0, "max_regression": 0.0},
        {"label": "p8_ema04", "patience": 8, "ema_alpha": 0.4, "max_regression": 0.0},
        {"label": "p8_ema04_reg01", "patience": 8, "ema_alpha": 0.4, "max_regression": 0.01},
        {"label": "p8_raw_reg01", "patience": 8, "ema_alpha": 0.0, "max_regression": 0.01},
        {"label": "p5_ema04_reg01", "patience": 5, "ema_alpha": 0.4, "max_regression": 0.01},
    ]


def _score_city(city: str, metrics: dict, *, min_iters: int, min_delta: float) -> dict:
    rows = []
    for rule in _rule_grid():
        analysis = analyze_stopping_trajectory(
            metrics,
            patience=rule["patience"],
            min_iters=min_iters,
            min_delta=min_delta,
            ema_alpha=rule["ema_alpha"],
            max_regression=rule["max_regression"],
        )
        stop = analysis["checkpoints"].get("simulated_early_stop")
        val_min = analysis["checkpoints"].get("holdout_val_minimum")
        oracle = analysis["oracle_best_lpips"]
        regret = analysis["regret_lpips_vs_oracle"]
        rows.append(
            {
                "city": city,
                "rule": rule["label"],
                **rule,
                "oracle_lpips": oracle["lpips"],
                "oracle_iter": oracle["iter"],
                "stop_iter": None if stop is None else stop["actual_iter"],
                "stop_lpips": None if stop is None else stop["lpips"],
                "stop_regret": regret.get("simulated_early_stop"),
                "val_min_iter": None if val_min is None else val_min["actual_iter"],
                "val_min_lpips": None if val_min is None else val_min["lpips"],
                "val_min_regret": regret.get("holdout_val_minimum"),
                "final_lpips": analysis["checkpoints"]["final_logged"]["lpips"],
            }
        )
    return {"city": city, "rows": rows}


def _aggregate(city_results: list[dict]) -> dict:
    by_rule: dict[str, list[dict]] = {}
    for cr in city_results:
        for row in cr["rows"]:
            by_rule.setdefault(row["rule"], []).append(row)

    summary = []
    for rule, rows in by_rule.items():
        regrets = [r["stop_regret"] for r in rows if r["stop_regret"] is not None]
        stops = [r["stop_iter"] for r in rows if r["stop_iter"] is not None]
        never = sum(1 for r in rows if r["stop_iter"] is None)
        summary.append(
            {
                "rule": rule,
                "n_cities": len(rows),
                "n_stopped": len(stops),
                "n_never_stopped": never,
                "mean_stop_regret": (sum(regrets) / len(regrets)) if regrets else None,
                "median_stop_regret": (
                    sorted(regrets)[len(regrets) // 2] if regrets else None
                ),
                "mean_stop_iter": (sum(stops) / len(stops)) if stops else None,
            }
        )
    summary.sort(
        key=lambda r: (
            float("inf") if r["mean_stop_regret"] is None else r["mean_stop_regret"]
        )
    )
    return {"by_rule": summary}


def _run_live(city: str, device: int, iters: int) -> Path:
    s2 = ROOT / "data" / "s2_revisits" / f"{city}_lr512"
    if not (s2 / "meta.json").is_file():
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "make_lr_size_variants.py"),
                "--city",
                city,
                "--sizes",
                "512",
            ],
            cwd=ROOT,
            check=True,
        )
    run_name = f"stop_trace_k4_{city}_i{iters}"
    cmd = [
        sys.executable,
        str(ROOT / "optimize.py"),
        "--dataset",
        city,
        "--s2-dir",
        str(s2),
        "--run_name",
        run_name,
        "--lr_degradation",
        "s2_psf_m",
        "--recon_loss",
        "charbonnier",
        "--charbonnier_eps",
        "0.01",
        "--lr_tile",
        "128",
        "--lr_tiles_per_step",
        "4",
        "--lr_tile_mix",
        "within",
        "--early_stop_metric",
        "holdout_mse",
        "--early_stop_patience",
        "0",
        "--early_stop_ema",
        "0",
        "--early_stop_max_regression",
        "0",
        "--early_stop_min_iters",
        "1000",
        "--force_hr_eval",
        "--iters",
        str(iters),
        "--eval_every",
        "200",
        "--hr_render_tile",
        "2048",
        "--spatial_holdout",
        "0.1",
        "--holdout_block",
        "0",
        "--device",
        str(device),
        "--no_qgis_export",
    ]
    align = ROOT / "eval" / "spatial_alignment.json"
    if align.is_file():
        cmd += ["--spatial_alignment_path", str(align)]
    print(f"[gpu{device}] {run_name}", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    return ROOT / "single_samples" / city / "sample" / run_name / "metrics.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cities", nargs="+", default=None)
    ap.add_argument("--min-iters", type=int, default=1000)
    ap.add_argument("--min-delta", type=float, default=0.0005)
    ap.add_argument("--run", action="store_true", help="Launch live k4 traces with HR logging.")
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--gpu-offset", type=int, default=0)
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "single_samples" / "sweep_results" / "bench_stopping_rules.json",
    )
    args = ap.parse_args()

    cities = list(args.cities) if args.cities else list(DEFAULT_CITIES)
    city_results: list[dict] = []

    if args.run:
        import threading

        work = list(cities)
        lock = threading.Lock()

        def worker(gpu: int) -> None:
            while True:
                with lock:
                    if not work:
                        return
                    city = work.pop(0)
                try:
                    mpath = _run_live(city, gpu, args.iters)
                    metrics = json.loads(mpath.read_text())
                    scored = _score_city(
                        city, metrics, min_iters=args.min_iters, min_delta=args.min_delta
                    )
                    scored["metrics_path"] = str(mpath.relative_to(ROOT))
                    with lock:
                        city_results.append(scored)
                except Exception as exc:  # noqa: BLE001
                    with lock:
                        city_results.append({"city": city, "error": str(exc), "rows": []})
                    print(f"FAIL {city}: {exc}", flush=True)

        threads = [
            threading.Thread(target=worker, args=(args.gpu_offset + i,), daemon=True)
            for i in range(max(1, min(args.gpus, len(cities))))
        ]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
    else:
        for city in cities:
            analysis_path = VAL_DIR / f"{city}_analysis.json"
            if not analysis_path.is_file():
                print(f"skip {city}: missing {analysis_path}", flush=True)
                continue
            metrics = _load_metrics_from_analysis(analysis_path)
            city_results.append(
                _score_city(
                    city, metrics, min_iters=args.min_iters, min_delta=args.min_delta
                )
            )

    out = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "live_k4" if args.run else "offline_early_stop_validation",
        "min_iters": args.min_iters,
        "min_delta": args.min_delta,
        "aggregate": _aggregate(city_results),
        "cities": city_results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out["aggregate"], indent=2))
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
