#!/usr/bin/env python3
"""Render an overview of two georeferenced delivery mosaics under one display mapping."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from eval.display import highlight_compress  # noqa: E402


def _read_overview(path: Path, side: int) -> np.ndarray:
    with rasterio.open(path) as src:
        scale = min(side / src.width, side / src.height, 1.0)
        height = max(1, int(round(src.height * scale)))
        width = max(1, int(round(src.width * scale)))
        arr = src.read(
            indexes=(1, 2, 3),
            out_shape=(3, height, width),
            resampling=Resampling.average,
        )
    return np.transpose(arr, (1, 2, 0)).astype(np.float32)


def render(
    before_path: Path,
    after_path: Path,
    out: Path,
    *,
    side: int,
    before_title: str = "independent 2% identity",
    after_title: str = "ICM identity + date-cut ramp",
    figure_title: str | None = None,
    tone: str = "highlight",
) -> None:
    before = _read_overview(before_path, side)
    after = _read_overview(after_path, side)
    if tone == "stretch":
        values = np.concatenate(
            [
                before[np.any(before > 0, axis=2)],
                after[np.any(after > 0, axis=2)],
            ],
            axis=0,
        )
        lo, hi = np.percentile(values, [2, 98], axis=0)

        def display(rgb: np.ndarray) -> np.ndarray:
            return np.clip((rgb - lo) / np.maximum(hi - lo, 1e-6), 0.0, 1.0)
    else:
        display = highlight_compress

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 6.2))
    for ax, rgb, title in (
        (axes[0], before, before_title),
        (axes[1], after, after_title),
    ):
        ax.imshow(display(rgb), interpolation="nearest")
        ax.set_title(title)
        ax.set_axis_off()
    if figure_title:
        fig.suptitle(figure_title)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--side", type=int, default=1400)
    parser.add_argument("--before-title", default="independent 2% identity")
    parser.add_argument("--after-title", default="ICM identity + date-cut ramp")
    parser.add_argument("--figure-title", default=None)
    parser.add_argument("--tone", choices=("highlight", "stretch"), default="highlight")
    args = parser.parse_args()
    render(
        args.before,
        args.after,
        args.out,
        side=args.side,
        before_title=args.before_title,
        after_title=args.after_title,
        figure_title=args.figure_title,
        tone=args.tone,
    )


if __name__ == "__main__":
    main()
