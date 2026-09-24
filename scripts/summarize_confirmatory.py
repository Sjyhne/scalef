#!/usr/bin/env python3
"""Aggregate confirmatory_v1 metrics.json files into a paper-freeze summary.

Writes a results JSON under single_samples/sweep_results/confirmatory_v1/ that
scripts/build_paper_results.py can copy into paper/results/.

Example
-------
    python scripts/summarize_confirmatory.py --family fixed_k
    python scripts/summarize_confirmatory.py --family encoding_size
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
NAMESPACE = "confirmatory_v1"
DEFAULT_ROOT = ROOT / "single_samples"
DEFAULT_OUT_DIR = ROOT / "single_samples" / "sweep_results" / NAMESPACE


def _load_metrics(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _row_from_metrics(path: Path) -> dict[str, Any]:
    payload = _load_metrics(path)
    name = path.parent.name
    parts = name.split("__")
    family = parts[1] if len(parts) > 1 else ""
    config = parts[2] if len(parts) > 2 else ""
    city = parts[3] if len(parts) > 3 else payload.get("dataset")
    seed_token = parts[4] if len(parts) > 4 else ""
    seed = int(seed_token.replace("seed", "")) if seed_token.startswith("seed") else None
    lpips = payload.get("lpips") or {}
    gpu_memory = payload.get("gpu_memory") or {}
    return {
        "run_name": name,
        "family": family,
        "config": config,
        "city": city,
        "seed": seed,
        "lpips": lpips.get("model"),
        "lpips_bilinear": lpips.get("bilinear"),
        "lpips_improvement": lpips.get("improvement"),
        "psnr": payload.get("model_psnr"),
        "psnr_bilinear": payload.get("bilinear_psnr"),
        "training_time_s": payload.get("training_time_seconds"),
        "process_peak_gpu_memory_gb": gpu_memory.get("process_peak_used_gb"),
        "torch_peak_allocated_gpu_memory_gb": gpu_memory.get("torch_peak_allocated_gb"),
        "gpu_memory_source": gpu_memory.get("process_memory_source"),
        "completed_iters": payload.get("completed_iters"),
        "df": payload.get("downsampling_factor"),
        "query_gsd": payload.get("query_gsd"),
        "metrics_path": str(path.relative_to(ROOT)),
    }


def collect(family: str, *, sample_root: Path, namespace: str = NAMESPACE) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pattern = f"*/{namespace}/{namespace}__{family}__*/metrics.json"
    for path in sorted(sample_root.glob(pattern)):
        rows.append(_row_from_metrics(path))
    return rows


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["config"]), []).append(row)
    out = []
    for config, group in grouped.items():
        lpips = [r["lpips"] for r in group if r["lpips"] is not None]
        bilin = [r["lpips_bilinear"] for r in group if r["lpips_bilinear"] is not None]
        times = [r["training_time_s"] for r in group if r["training_time_s"] is not None]
        process_memory = [
            r["process_peak_gpu_memory_gb"]
            for r in group
            if r["process_peak_gpu_memory_gb"] is not None
        ]
        out.append(
            {
                "config": config,
                "n": len(group),
                "n_cities": len({r["city"] for r in group}),
                "n_seeds": len({r["seed"] for r in group}),
                "lpips": mean(lpips) if lpips else None,
                "lpips_bilinear": mean(bilin) if bilin else None,
                "training_time_s": mean(times) if times else None,
                "process_peak_gpu_memory_gb": (
                    mean(process_memory) if process_memory else None
                ),
                "process_peak_gpu_memory_sd_gb": (
                    stdev(process_memory) if len(process_memory) > 1 else None
                ),
                "n_process_peak_gpu_memory": len(process_memory),
            }
        )
    order = {
        "lr128_k1": 0,
        "lr128_k2": 1,
        "lr128_k4": 2,
        "lr128_k8": 3,
        "lr128_full": 4,
        "df2_5m": 0,
        "df4_query": 1,
    }
    return sorted(out, key=lambda r: (order.get(str(r["config"]), 50), str(r["config"])))


def _fixed_k_uncertainty(
    rows: list[dict[str, Any]], *, seed: int = 20260914, n_resamples: int = 200_000
) -> dict[str, Any]:
    """Site-cluster bootstrap after averaging seeds within each site."""
    grouped: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        if row.get("lpips") is None:
            continue
        grouped.setdefault(str(row["config"]), {}).setdefault(str(row["city"]), []).append(
            float(row["lpips"])
        )
    cities = sorted(set.intersection(*(set(sites) for sites in grouped.values())))
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(cities), size=(n_resamples, len(cities)))
    configs: dict[str, Any] = {}
    site_means: dict[str, np.ndarray] = {}
    for config, sites in sorted(grouped.items()):
        values = np.asarray([np.mean(sites[city]) for city in cities], dtype=np.float64)
        site_means[config] = values
        boot = values[indices].mean(axis=1)
        configs[config] = {
            "mean_lpips": float(values.mean()),
            "site_mean_sd": float(values.std(ddof=1)),
            "site_bootstrap_ci95": [float(v) for v in np.quantile(boot, [0.025, 0.975])],
            "mean_within_site_seed_sd": float(
                np.mean([np.std(sites[city], ddof=1) for city in cities])
            ),
        }
    paired = {}
    for other in ("lr128_k1", "lr128_k2", "lr128_k8", "lr128_full"):
        delta = site_means["lr128_k4"] - site_means[other]
        boot = delta[indices].mean(axis=1)
        paired[f"lr128_k4_minus_{other}"] = {
            "mean_lpips_delta": float(delta.mean()),
            "site_delta_sd": float(delta.std(ddof=1)),
            "site_bootstrap_ci95": [float(v) for v in np.quantile(boot, [0.025, 0.975])],
        }
    return {
        "method": "nonparametric site-cluster bootstrap of within-site seed means",
        "n_sites": len(cities),
        "seeds_per_site": 3,
        "n_resamples": n_resamples,
        "seed": seed,
        "configs": configs,
        "paired": paired,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", required=True)
    ap.add_argument("--namespace", default=NAMESPACE)
    ap.add_argument("--sample-root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    rows = collect(args.family, sample_root=args.sample_root, namespace=args.namespace)
    if not rows:
        raise SystemExit(f"no {args.family} metrics under {args.sample_root}")
    summary = {
        "schema": "scalef.confirmatory_summary.v1",
        "family": args.family,
        "namespace": args.namespace,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_runs": len(rows),
        "results": rows,
        "by_config": _aggregate(rows),
    }
    if args.family == "fixed_k":
        summary["uncertainty"] = _fixed_k_uncertainty(rows)
    out = args.out
    if out is None:
        out = DEFAULT_OUT_DIR / f"{args.family}_summary.json"
    elif not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {out} ({len(rows)} runs)")
    for row in summary["by_config"]:
        print(
            f"  {row['config']:16} n={row['n']:3}  "
            f"LPIPS {row['lpips']:.3f}  bil {row['lpips_bilinear']:.3f}  "
            f"{row['training_time_s']:.1f}s  "
            f"GPU {row['process_peak_gpu_memory_gb']:.2f} GiB"
            if row["process_peak_gpu_memory_gb"] is not None
            else (
                f"  {row['config']:16} n={row['n']:3}  "
                f"LPIPS {row['lpips']:.3f}  bil {row['lpips_bilinear']:.3f}  "
                f"{row['training_time_s']:.1f}s  GPU n/a"
            )
        )


if __name__ == "__main__":
    main()
