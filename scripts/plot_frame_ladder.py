#!/usr/bin/env python3
"""Plot the 8/12/16-frame ladder produced by scripts/diagnose_divergence.py.

Frame count was expected to be the binding constraint on high-frequency
recovery. These panels test that directly: whether adding revisits raises the
correlation between predicted and true above-cutoff detail, and what the
prediction does to that detail over the course of training.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CITY_ORDER = ["stavanger", "sandvika", "tromso", "rana", "bergen", "kristiansand", "trondheim"]
FRAME_COUNTS = (8, 12, 16)


def load(results_dir: Path) -> dict[tuple[str, int], dict]:
    rows: dict[tuple[str, int], dict] = {}
    for path in glob.glob(str(results_dir / "*.jsonl")):
        text = Path(path).read_text().strip()
        if not text:
            continue
        rec = json.loads(text.split("\n")[0])
        rows[(rec["city"], rec["frames"])] = rec
    if not rows:
        raise SystemExit(f"no result files in {results_dir}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=Path("logs/ladder3"))
    ap.add_argument("--out", type=Path, default=Path("figures/frame_ladder.png"))
    a = ap.parse_args()

    rows = load(a.results)
    cities = [c for c in CITY_ORDER if (c, 8) in rows]
    colors = dict(zip(cities, plt.cm.tab10.colors))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax_corr, ax_psnr, ax_traj, ax_span = axes.ravel()

    # Perceptual quality against the bilinear baseline, as a function of frame count.
    for c in cities:
        ax_corr.plot(FRAME_COUNTS, [rows[(c, n)]["best_lpips"] for n in FRAME_COUNTS],
                     "o-", color=colors[c], label=c, lw=1.8, ms=5)
        ax_corr.axhline(rows[(c, 8)]["bilinear_lpips"], color=colors[c],
                        ls=":", lw=1.0, alpha=0.55)
    mean = [sum(rows[(c, n)]["best_lpips"] for c in cities) / len(cities) for n in FRAME_COUNTS]
    ax_corr.plot(FRAME_COUNTS, mean, "k--s", lw=2.6, ms=7, label="mean", zorder=5)
    ax_corr.set_xlabel("LR frames used")
    ax_corr.set_ylabel("LPIPS (lower is better)")
    ax_corr.set_title("Model beats bilinear everywhere, but frame count does not matter\n"
                      "(dotted = that city's bilinear baseline)", fontsize=10)
    ax_corr.set_xticks(FRAME_COUNTS)
    ax_corr.legend(fontsize=7, ncol=2)
    ax_corr.grid(alpha=0.3)

    # Why the metric choice mattered: PSNR prefers the blurrier image.
    width = 0.35
    xs = range(len(cities))
    ax_psnr.bar([x - width / 2 for x in xs],
                [rows[(c, 16)]["bilinear_lpips"] - rows[(c, 16)]["best_lpips"] for c in cities],
                width, label="LPIPS gain over bilinear", color="#2c7fb8")
    ax_psnr.bar([x + width / 2 for x in xs],
                [rows[(c, 16)]["above_vs_bilinear"] for c in cities],
                width, label="above-cutoff PSNR gain (dB)", color="#d95f0e")
    ax_psnr.axhline(0, color="k", lw=1.4)
    ax_psnr.set_xticks(list(xs))
    ax_psnr.set_xticklabels(cities, rotation=30, ha="right", fontsize=8)
    ax_psnr.set_title("The two metrics disagree: LPIPS sees a real gain,\n"
                      "above-cutoff PSNR sees nothing because blur maximizes it", fontsize=10)
    ax_psnr.legend(fontsize=8)
    ax_psnr.grid(alpha=0.3, axis="y")

    # Training trajectory: where the perceptual optimum actually sits.
    for c in cities:
        traj = rows[(c, 16)]["trajectory"]
        ax_traj.plot([t["iter"] for t in traj], [t["lpips"] for t in traj],
                     "-", color=colors[c], lw=1.6, label=c)
        ax_traj.axhline(rows[(c, 16)]["bilinear_lpips"], color=colors[c], ls=":", lw=0.9, alpha=0.5)
        b = rows[(c, 16)]
        ax_traj.plot([b["best_lpips_iter"]], [b["best_lpips"]], "o", color=colors[c], ms=7,
                     mec="k", mew=0.8, zorder=5)
    ax_traj.set_xscale("log")
    ax_traj.set_xlabel("iteration")
    ax_traj.set_ylabel("LPIPS (lower is better)")
    ax_traj.set_title("Perceptual optimum is at 1500-4500 iterations, not 500\n"
                      "(circles = optimum, dotted = bilinear)", fontsize=10)
    ax_traj.grid(alpha=0.3)

    # Frame count buys date span; does the span cost anything?
    for c in cities:
        spans = [rows[(c, n)]["date_span_days"] for n in FRAME_COUNTS]
        deltas = [rows[(c, n)]["best_lpips"] - rows[(c, 8)]["best_lpips"] for n in FRAME_COUNTS]
        ax_span.plot(spans, deltas, "o-", color=colors[c], lw=1.8, ms=5, label=c)
        ax_span.annotate(c, (spans[-1], deltas[-1]), fontsize=7,
                         xytext=(4, 0), textcoords="offset points", va="center")
    ax_span.axhline(0, color="k", lw=1.4)
    ax_span.set_xlabel("acquisition date span of the frames used (days)")
    ax_span.set_ylabel("LPIPS change vs that city at 8 frames\n(negative = wider window helped)")
    ax_span.set_title("Cost of the wider date window\n"
                      "small either way; rana gains, bergen and kristiansand lose", fontsize=10)
    ax_span.grid(alpha=0.3)

    fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=140, bbox_inches="tight")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
