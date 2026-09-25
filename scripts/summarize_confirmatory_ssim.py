#!/usr/bin/env python3
"""Per-site masked SSIM and paired site-bootstrap interval for a confirmatory namespace."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BOOT = dict(seed=20260914, n=200_000)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--namespace", default="confirmatory_v4_boa")
    ap.add_argument("--config", default="lr512_b0")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    runs: dict[str, list[tuple[float, float]]] = collections.defaultdict(list)
    pattern = f"*/{args.namespace}/{args.namespace}__b0_17__{args.config}__*__seed*/metrics.json"
    for f in (ROOT / "single_samples").glob(pattern):
        m = json.loads(f.read_text())
        runs[f.parts[-4]].append((float(m["ssim"]["model"]), float(m["ssim"]["bilinear"])))
    sites = sorted(runs)
    model = np.array([np.mean([r[0] for r in runs[c]]) for c in sites])
    bil = np.array([np.mean([r[1] for r in runs[c]]) for c in sites])
    gain = model - bil
    rng = np.random.default_rng(BOOT["seed"])
    idx = rng.integers(0, len(sites), size=(BOOT["n"], len(sites)))
    ci = np.quantile(gain[idx].mean(axis=1), [0.025, 0.975])
    out = {
        "namespace": args.namespace, "metric": "masked SSIM (eval.masked_metrics.masked_ssim)",
        "n_sites": len(sites), "n_runs": sum(len(v) for v in runs.values()), "bootstrap": BOOT,
        "ssim_mean": float(model.mean()), "ssim_bilinear_mean": float(bil.mean()),
        "ssim_gain": {"mean": float(gain.mean()), "site_sd": float(gain.std(ddof=1)),
                      "ci95": [float(ci[0]), float(ci[1])], "n_sites_improved": int((gain > 0).sum())},
        "ssim_within_site_seed_sd": float(np.mean([np.std([r[0] for r in runs[c]], ddof=1) for c in sites])),
        "per_site_model": dict(zip(sites, map(float, model))),
        "per_site_bilinear": dict(zip(sites, map(float, bil))),
    }
    dest = args.out or ROOT / "paper" / "results" / f"b0_17_{args.namespace.removeprefix('confirmatory_')}_ssim.json"
    dest.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if not k.startswith("per_site")}, indent=1))


if __name__ == "__main__":
    main()
