#!/usr/bin/env python3
"""Compare ways of choosing when to stop training, without HR ground truth.

Holding out the anchored reference revisit works, but it removes the very frame
that pins the geometric and radiometric gauge, so its validation signal decays for
reasons unrelated to image quality. Holding out pixel blocks independently per
frame keeps every revisit (and the anchor) in training while still leaving each
individual observation unseen.
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


def load(pattern: str) -> dict:
    rows = {}
    for path in glob.glob(pattern):
        text = Path(path).read_text().strip()
        if text:
            rec = json.loads(text.split("\n")[0])
            rows[(rec["city"], rec.get("loss", "mse"))] = rec
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--loss", default="laplace")
    ap.add_argument("--city", default="bergen", help="City for the trajectory panel.")
    ap.add_argument("--out", type=Path, default=Path("figures/stopping.png"))
    a = ap.parse_args()

    frame = load("logs/holdout/*.jsonl")
    spatial = load("logs/spatial/*.jsonl")
    drift = {}
    for path in glob.glob("logs/drift/*.jsonl"):
        rec = json.loads(Path(path).read_text().strip())
        drift[(rec["city"], "hold" if rec.get("holdout") else "anchor")] = rec

    cities = [c for c in CITIES if (c, a.loss) in spatial]
    fig, (ax_reg, ax_drift, ax_traj) = plt.subplots(1, 3, figsize=(16.5, 5))

    # How much quality each stopping rule leaves on the table.
    xs = range(len(cities))
    w = 0.27
    for j, (label, vals, color) in enumerate([
        ("held-out revisit (frame 0)", [frame[(c, a.loss)]["val_stop_regret"] for c in cities], "#1f77b4"),
        ("held-out pixel blocks", [spatial[(c, a.loss)]["val_stop_regret"] for c in cities], "#2ca02c"),
        ("no stop, run to 6000",
         [spatial[(c, a.loss)]["final_lpips"] - spatial[(c, a.loss)]["best_lpips"] for c in cities],
         "#d62728"),
    ]):
        ax_reg.bar([x + (j - 1) * w for x in xs], vals, w, label=label, color=color)
    ax_reg.set_xticks(list(xs))
    ax_reg.set_xticklabels(cities, rotation=30, ha="right", fontsize=8)
    ax_reg.set_ylabel("LPIPS worse than the oracle stop")
    ax_reg.set_title(f"Cost of each stopping rule ({a.loss})\nlower is better, 0 = matched the oracle",
                     fontsize=10)
    ax_reg.legend(fontsize=8)
    ax_reg.grid(alpha=0.3, axis="y")

    # Whether the holdout design disturbs the gauge that frame 0 anchors.
    dc = [c for c in cities if (c, "anchor") in drift]
    xs2 = range(len(dc))
    for j, (label, get, color) in enumerate([
        ("no holdout (anchor trains)", lambda c: drift[(c, "anchor")], "#7f7f7f"),
        ("frame 0 held out", lambda c: drift[(c, "hold")], "#1f77b4"),
        ("pixel blocks held out", lambda c: spatial[(c, "mse")], "#2ca02c"),
    ]):
        ax_drift.bar([x + (j - 1) * w for x in xs2],
                     [get(c)["trajectory"][-1]["drift_scale"] for c in dc],
                     w, label=label, color=color)
    ax_drift.axhline(1.0, color="k", lw=1.2, ls=":")
    ax_drift.set_ylim(bottom=1.0)
    ax_drift.set_xticks(list(xs2))
    ax_drift.set_xticklabels(dc, rotation=30, ha="right", fontsize=8)
    ax_drift.set_ylabel("mean colour scale at iter 6000")
    ax_drift.set_title("Radiometric gauge drift\nremoving the anchor inflates it; pixel blocks do not",
                       fontsize=10)
    ax_drift.legend(fontsize=8)
    ax_drift.grid(alpha=0.3, axis="y")

    # Does the validation curve actually track image quality?
    rec = spatial[(a.city, a.loss)]
    traj = rec["trajectory"]
    iters = [t["iter"] for t in traj]
    ax_traj.plot(iters, [t["lpips"] for t in traj], "-o", color="#d62728", lw=2, ms=4,
                 label="LPIPS vs HR ground truth")
    ax_traj.axvline(rec["best_lpips_iter"], color="#d62728", ls="--", lw=1.4)
    ax_traj.set_xscale("log")
    ax_traj.set_xlabel("iteration")
    ax_traj.set_ylabel("LPIPS", color="#d62728")
    ax_traj.tick_params(axis="y", labelcolor="#d62728")

    ax2 = ax_traj.twinx()
    ax2.plot(iters, [t["val_loss"] for t in traj], "-s", color="#2ca02c", lw=2, ms=4,
             label="held-out pixel MSE")
    ax2.axvline(rec["val_stop_iter"], color="#2ca02c", ls="--", lw=1.4)
    ax2.set_ylabel("held-out pixel MSE", color="#2ca02c")
    ax2.tick_params(axis="y", labelcolor="#2ca02c")
    ax_traj.set_title(f"{a.city}: the held-out signal needs no ground truth\n"
                      f"dashed lines mark each curve's minimum", fontsize=10)
    lines = ax_traj.get_lines()[:1] + ax2.get_lines()[:1]
    ax_traj.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="upper center")
    ax_traj.grid(alpha=0.3)

    fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=140, bbox_inches="tight")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
