#!/usr/bin/env python3
"""Render the two-scale ScaleF method/system diagram used by the paper."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent.parent


def _box(ax, x, y, w, h, text, *, face="#EEF3F8", edge="#355C7D", size=9):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=1.2,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size)
    return patch


def _arrow(ax, x0, y0, x1, y1, *, text=None):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.2,
            color="#37474F",
        )
    )
    if text:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.025, text, ha="center", fontsize=8)


def render(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(13.2, 6.1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.015, 0.955, "A  One LR512 field: reference-free test-time optimization", weight="bold", fontsize=12)

    xs = [0.02, 0.17, 0.32, 0.47, 0.62, 0.77]
    widths = [0.115, 0.115, 0.11, 0.11, 0.11, 0.105]
    labels = [
        "Sentinel-2\nrevisits",
        "frame affine +\nband gain/bias",
        "HR coordinates\n(partial tiles)",
        "multiresolution\nhash + fused MLP",
        "shared HR\nreflectance field",
        "band-wise MTF\n+ area downsample",
    ]
    for x, w, label in zip(xs, widths, labels):
        _box(ax, x, 0.68, w, 0.16, label)
    for i in range(len(xs) - 1):
        _arrow(ax, xs[i] + widths[i], 0.76, xs[i + 1], 0.76)

    _box(ax, 0.90, 0.68, 0.085, 0.16, "predicted\nLR", face="#FFF4D6", edge="#B07D00")
    _arrow(ax, xs[-1] + widths[-1], 0.76, 0.90, 0.76)

    _box(ax, 0.77, 0.49, 0.105, 0.10, "observed LR\ntarget", face="#F2F2F2", edge="#666666")
    _box(ax, 0.90, 0.49, 0.085, 0.10, "masked loss", face="#FDECEC", edge="#A94442")
    _arrow(ax, 0.822, 0.59, 0.93, 0.68)
    _arrow(ax, 0.942, 0.68, 0.942, 0.59)

    _box(ax, 0.47, 0.49, 0.18, 0.10, "90% train blocks", face="#EAF6EA", edge="#3C7D3C")
    _box(ax, 0.66, 0.49, 0.09, 0.10, "10% hold-out", face="#FFF4D6", edge="#B07D00")
    _arrow(ax, 0.65, 0.54, 0.66, 0.54)
    ax.text(0.56, 0.445, "EMA validation → patience → restore best checkpoint", ha="center", fontsize=9)

    ax.plot([0.015, 0.985], [0.39, 0.39], color="#B0BEC5", linewidth=1)
    ax.text(0.015, 0.345, "B  Geographic system: season-aware scheduling and mosaicking", weight="bold", fontsize=12)

    sx = [0.03, 0.21, 0.40, 0.59, 0.78]
    sw = [0.135, 0.135, 0.14, 0.14, 0.17]
    sl = [
        "MGRS inventory",
        "shared season\ncloud + snow + NDVI",
        "mainland LR512\njob manifest",
        "parallel independent\nScaleF fits",
        "georeferenced overlap\n+ feathered mosaic",
    ]
    for x, w, label in zip(sx, sw, sl):
        _box(ax, x, 0.12, w, 0.14, label, face="#F2F0FA", edge="#615192")
    for i in range(len(sx) - 1):
        _arrow(ax, sx[i] + sw[i], 0.19, sx[i + 1], 0.19)

    ax.text(
        0.5,
        0.035,
        "Every stage records accepted cells, exact inputs, failures, and wall-clock time.",
        ha="center",
        fontsize=9,
        color="#455A64",
    )

    fig.tight_layout(pad=0.5)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", dpi=220)
    if output.suffix.lower() != ".png":
        fig.savefig(output.with_suffix(".png"), bbox_inches="tight", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "ScaleF_Overleaf" / "figures" / "scalef_method.pdf",
    )
    args = parser.parse_args()
    render(args.out if args.out.is_absolute() else ROOT / args.out)


if __name__ == "__main__":
    main()
