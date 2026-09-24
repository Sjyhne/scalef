#!/usr/bin/env python3
"""Compare training criteria and test a ground-truth-free stopping rule.

All four reconstruction losses reach the same peak perceptual quality, so the
interesting difference is what happens after the peak: L2-style objectives decay
roughly twice as fast as L1-style ones. The last panel checks whether holding out
the anchored reference revisit can locate the stopping point without ever looking
at the HR ground truth.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CITIES = ["stavanger", "sandvika", "tromso", "rana", "bergen", "kristiansand", "trondheim"]
LOSSES = ["mse", "mae", "gnll", "laplace"]
LOSS_COLOR = {"mse": "#d62728", "mae": "#1f77b4", "gnll": "#ff7f0e", "laplace": "#2ca02c"}


def load(pattern: str, key) -> dict:
    rows = {}
    for path in glob.glob(pattern):
        text = Path(path).read_text().strip()
        if text:
            rec = json.loads(text.split("\n")[0])
            rows[key(rec)] = rec
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--losses", default="logs/losses/*.jsonl")
    ap.add_argument("--holdout", default="logs/holdout/*.jsonl")
    ap.add_argument("--out", type=Path, default=Path("figures/loss_comparison.png"))
    a = ap.parse_args()

    rows = load(a.losses, lambda r: (r["city"], r["loss"]))
    hold = load(a.holdout, lambda r: (r["city"], r["loss"]))
    cities = [c for c in CITIES if (c, "mse") in rows]

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9))
    ax_peak, ax_decay, ax_traj, ax_stop = axes.ravel()

    # Peak quality: essentially identical across criteria.
    xs = range(len(cities))
    w = 0.2
    for j, L in enumerate(LOSSES):
        ax_peak.bar([x + (j - 1.5) * w for x in xs],
                    [rows[(c, L)]["best_lpips"] for c in cities],
                    w, label=L, color=LOSS_COLOR[L])
    ax_peak.plot(list(xs), [rows[(c, "mse")]["bilinear_lpips"] for c in cities],
                 "k_", ms=26, mew=2.2, label="bilinear")
    ax_peak.set_xticks(list(xs))
    ax_peak.set_xticklabels(cities, rotation=30, ha="right", fontsize=8)
    ax_peak.set_ylabel("best LPIPS (lower is better)")
    ax_peak.set_title("Peak quality is the same for every criterion\n"
                      "all four beat bilinear by a similar margin", fontsize=10)
    ax_peak.legend(fontsize=8, ncol=5)
    ax_peak.grid(alpha=0.3, axis="y")

    # Robustness to overtraining: the real difference between the criteria.
    for j, L in enumerate(LOSSES):
        ax_decay.bar([x + (j - 1.5) * w for x in xs],
                     [rows[(c, L)]["final_lpips"] - rows[(c, L)]["best_lpips"] for c in cities],
                     w, label=L, color=LOSS_COLOR[L])
    ax_decay.set_xticks(list(xs))
    ax_decay.set_xticklabels(cities, rotation=30, ha="right", fontsize=8)
    ax_decay.set_ylabel("LPIPS lost by running to iter 6000")
    ax_decay.set_title("L1-style losses are about twice as forgiving of overtraining\n"
                       "(MAE and Laplace vs MSE and GNLL)", fontsize=10)
    ax_decay.legend(fontsize=8, ncol=4)
    ax_decay.grid(alpha=0.3, axis="y")

    # Trajectory shape, averaged over cities.
    for L in LOSSES:
        iters = [t["iter"] for t in rows[(cities[0], L)]["trajectory"]]
        curves = []
        for c in cities:
            t = {x["iter"]: x["lpips"] for x in rows[(c, L)]["trajectory"]}
            curves.append([t[i] for i in iters])
        mean = [sum(col) / len(col) for col in zip(*curves)]
        ax_traj.plot(iters, mean, "-o", color=LOSS_COLOR[L], lw=2, ms=4, label=L)
    ax_traj.axhline(sum(rows[(c, "mse")]["bilinear_lpips"] for c in cities) / len(cities),
                    color="k", ls=":", lw=1.4, label="bilinear")
    ax_traj.set_xscale("log")
    ax_traj.set_xlabel("iteration")
    ax_traj.set_ylabel("LPIPS, mean over cities")
    ax_traj.set_title("Same optimum near 1500 iterations, different tails", fontsize=10)
    ax_traj.legend(fontsize=8)
    ax_traj.grid(alpha=0.3)

    # A stopping rule that needs no ground truth.
    if hold:
        hc = [c for c in cities if (c, "mse") in hold]
        xs2 = range(len(hc))
        for j, (label, get, color) in enumerate([
            ("oracle stop (needs HR GT)", lambda r: 0.0, "#2ca02c"),
            ("held-out revisit stop", lambda r: r["val_stop_regret"], "#1f77b4"),
            ("no stop, run to 6000", lambda r: r["final_lpips"] - r["best_lpips"], "#d62728"),
        ]):
            ax_stop.bar([x + (j - 1) * 0.27 for x in xs2],
                        [get(hold[(c, "mse")]) for c in hc], 0.27, label=label, color=color)
        ax_stop.set_xticks(list(xs2))
        ax_stop.set_xticklabels(hc, rotation=30, ha="right", fontsize=8)
        ax_stop.set_ylabel("LPIPS worse than the oracle stop")
        ax_stop.set_title("Holding out the anchored reference revisit recovers most of\n"
                          "the gain, with no ground truth (MSE runs)", fontsize=10)
        ax_stop.legend(fontsize=8)
        ax_stop.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=140, bbox_inches="tight")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
