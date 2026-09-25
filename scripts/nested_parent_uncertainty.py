#!/usr/bin/env python3
"""Per-parent quality differences, cost ratios, and bootstrap intervals for the nested ladder.

Each of the 11 float-validated parents is the single parent of its NIB project, so parent means
are also project means and the parent is the resampling unit. Window scores are averaged within
each parent; intervals are nonparametric percentile bootstraps over parents with the fixed seed
used for the 17-site intervals.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_nested_float_validation import MANIFEST as RUN_MANIFEST  # noqa: E402

RESULTS = ROOT / "paper" / "results"
SOURCES = {
    "window": (RESULTS / "nested_parent_reference.json", "lpips_child_ref", "psnr_child_ref"),
    "parent": (RESULTS / "nested_parent_reference.json", "lpips_parent_ref", "psnr_parent_ref"),
    "window_offset_corrected": (RESULTS / "boa_offset_assessment.json", "lpips_corrected", "psnr_corrected"),
    "parent_offset_corrected": (RESULTS / "boa_offset_assessment_parent_ref.json", "lpips_corrected",
                                "psnr_corrected"),
    "v4_window": (RESULTS / "nested_v4_scores.json", "lpips_window", "psnr_window"),
    "v4_parent": (RESULTS / "nested_v4_scores.json", "lpips_parent", "psnr_parent"),
}
SIDES = (64, 128, 256, 512)
N_BOOT = 200_000
SEED = 20260914
PARENT_KM2 = 26.2144


def parent_means(path: Path, key: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in json.loads(path.read_text())["rows"]:
        out.setdefault(r["parent"], {}).setdefault(r["side"], []).append(r[key])
    return {p: {s: statistics.fmean(v) for s, v in d.items()} for p, d in out.items()}


def parent_costs() -> dict[str, dict[int, float]]:
    cost: dict[str, dict[int, float]] = {}
    for j in json.loads(RUN_MANIFEST.read_text())["jobs"]:
        m = json.loads(Path(j["metrics"]).read_text())
        d = cost.setdefault(j["parent_tile_id"], {})
        d[int(j["side"])] = d.get(int(j["side"]), 0.0) + float(m["training_time_seconds"])
    return cost


def bootstrap(values: np.ndarray, stat=np.mean) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(values), size=(N_BOOT, len(values)))
    draws = stat(values[idx], axis=1) if stat is np.mean else np.array([stat(values[i]) for i in idx])
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def summarize(per_parent: dict[str, float]) -> dict:
    v = np.array(list(per_parent.values()))
    lo, hi = bootstrap(v)
    return {"mean": float(v.mean()), "ci95": [lo, hi], "min": float(v.min()), "max": float(v.max()),
            "n_positive": int((v > 0).sum()), "n": int(len(v)), "per_parent": per_parent}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=sorted(SOURCES), default="window")
    args = ap.parse_args()
    path, lp_key, ps_key = SOURCES[args.source]
    lp, ps = parent_means(path, lp_key), parent_means(path, ps_key)
    cost = parent_costs()
    parents = sorted(lp)

    contrasts = {
        "lpips_512_minus_64": {p: lp[p][512] - lp[p][64] for p in parents},
        "lpips_512_minus_256": {p: lp[p][512] - lp[p][256] for p in parents},
        "lpips_256_minus_64": {p: lp[p][256] - lp[p][64] for p in parents},
        "psnr_512_minus_64": {p: ps[p][512] - ps[p][64] for p in parents},
        **{f"lpips_gain_vs_bilinear_{s}": {p: lp[p]["bilinear"] - lp[p][s] for p in parents} for s in SIDES},
    }
    out = {"source": args.source, "scores_file": str(path.relative_to(ROOT)), "n_boot": N_BOOT, "seed": SEED,
           "resampling_unit": "parent (one per NIB project)",
           "contrasts": {k: summarize(v) for k, v in contrasts.items()}}

    ratio = {p: cost[p][64] / cost[p][512] for p in parents}
    c64 = np.array([cost[p][64] for p in parents])
    c512 = np.array([cost[p][512] for p in parents])
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(parents), size=(N_BOOT, len(parents)))
    pooled = c64[idx].mean(axis=1) / c512[idx].mean(axis=1)
    out["throughput_512_over_64"] = {
        "ratio_of_means": float(c64.mean() / c512.mean()),
        "ci95": [float(np.percentile(pooled, 2.5)), float(np.percentile(pooled, 97.5))],
        "per_parent_min": float(min(ratio.values())), "per_parent_max": float(max(ratio.values())),
        "per_parent": ratio,
        "gpu_s_per_km2": {str(s): float(np.mean([cost[p][s] for p in parents]) / PARENT_KM2) for s in SIDES},
    }
    dest = RESULTS / f"nested_parent_uncertainty_{args.source}.json"
    dest.write_text(json.dumps(out, indent=1) + "\n")
    for k, v in out["contrasts"].items():
        print(f"{k:28s} mean {v['mean']:+.4f}  CI [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]  "
              f"range [{v['min']:+.4f}, {v['max']:+.4f}]  positive {v['n_positive']}/{v['n']}")
    t = out["throughput_512_over_64"]
    print(f"throughput 512/64: {t['ratio_of_means']:.1f}x CI [{t['ci95'][0]:.1f}, {t['ci95'][1]:.1f}] "
          f"per-parent [{t['per_parent_min']:.1f}, {t['per_parent_max']:.1f}]")


if __name__ == "__main__":
    main()
