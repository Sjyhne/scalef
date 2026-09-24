#!/usr/bin/env python3
"""Render a national-cell identity-date and radiometric-correction QA map."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from rasterio.enums import Resampling


def _overview(path: Path, side: int) -> np.ndarray:
    with rasterio.open(path) as src:
        scale = min(side / src.width, side / src.height, 1.0)
        shape = (max(1, round(src.height * scale)), max(1, round(src.width * scale)))
        array = src.read(
            (1, 2, 3),
            out_shape=(3, *shape),
            resampling=Resampling.average,
        )
    return np.moveaxis(array, 0, -1).astype(np.float32)


def _stretch(rgb: np.ndarray) -> np.ndarray:
    valid = np.any(rgb > 0, axis=2)
    values = rgb[valid]
    lo, hi = np.percentile(values, [2, 98], axis=0)
    out = np.clip((rgb - lo) / np.maximum(hi - lo, 1e-6), 0, 1)
    out[~valid] = 0
    return out


def render(
    mosaic: Path,
    manifest_path: Path,
    identity_path: Path,
    out: Path,
    *,
    sidecar_path: Path | None = None,
    side: int = 1200,
) -> None:
    manifest = json.loads(manifest_path.read_text())
    identity = json.loads(identity_path.read_text())
    assignment = identity["assignment"]
    tiles = manifest["tiles"]
    by_xy = {
        (int(tile["iy"]), int(tile["ix"])): tile["tile_id"] for tile in tiles
    }
    n_y = 1 + max(y for y, _ in by_xy)
    n_x = 1 + max(x for _, x in by_xy)
    dates = sorted({assignment[tile_id] for tile_id in by_xy.values()})
    date_index = {value: index for index, value in enumerate(dates)}

    date_grid = np.full((n_y, n_x), np.nan, dtype=np.float32)
    for xy, tile_id in by_xy.items():
        date_grid[xy] = date_index[assignment[tile_id]]

    corrections = {}
    if sidecar_path is not None and sidecar_path.is_file():
        sidecar = json.loads(sidecar_path.read_text())
        corrections = ((sidecar.get("harmonization") or {}).get("corrections") or {})
    correction_grid = np.full((n_y, n_x), np.nan, dtype=np.float32)
    for xy, tile_id in by_xy.items():
        value = corrections.get(tile_id)
        if value is not None:
            correction_grid[xy] = float(np.max(np.abs(value)))

    rgb = _overview(mosaic, side)
    height, width = rgb.shape[:2]
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, max(len(dates), 1)))
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(len(dates) + 1) - 0.5, len(dates))
    panels = 3 if corrections else 2
    fig, axes = plt.subplots(1, panels, figsize=(6.2 * panels, 7.2))
    axes = np.atleast_1d(axes)

    axes[0].imshow(_stretch(rgb), interpolation="nearest")
    axes[0].set_title("RGB with identity-date cuts")
    for (iy, ix), tile_id in by_xy.items():
        day = assignment[tile_id]
        for neighbour, points in (
            (
                (iy, ix + 1),
                ((ix + 1) * width / n_x, iy * height / n_y,
                 (ix + 1) * width / n_x, (iy + 1) * height / n_y),
            ),
            (
                (iy + 1, ix),
                (ix * width / n_x, (iy + 1) * height / n_y,
                 (ix + 1) * width / n_x, (iy + 1) * height / n_y),
            ),
        ):
            other = by_xy.get(neighbour)
            if other is None:
                continue
            different = day != assignment[other]
            axes[0].plot(
                (points[0], points[2]),
                (points[1], points[3]),
                color="red" if different else "white",
                linewidth=1.0 if different else 0.15,
                alpha=0.9 if different else 0.16,
            )
    axes[0].set_axis_off()

    axes[1].imshow(date_grid, cmap=cmap, norm=norm, interpolation="nearest")
    axes[1].set_title("Assigned base acquisition")
    axes[1].set_axis_off()
    axes[1].legend(
        handles=[Patch(color=colors[i], label=day) for i, day in enumerate(dates)],
        loc="lower left",
        fontsize=7,
        framealpha=0.85,
    )

    if corrections:
        image = axes[2].imshow(
            correction_grid,
            cmap="magma",
            vmin=0,
            vmax=max(0.02, float(np.nanpercentile(correction_grid, 99))),
            interpolation="nearest",
        )
        axes[2].set_title("Maximum absolute RGB correction")
        axes[2].set_axis_off()
        fig.colorbar(image, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(f"{manifest['parent']} identity and radiometric QA")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mosaic", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--identity-plan", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--side", type=int, default=1200)
    args = parser.parse_args()
    render(
        args.mosaic,
        args.manifest,
        args.identity_plan,
        args.out,
        sidecar_path=args.sidecar,
        side=args.side,
    )


if __name__ == "__main__":
    main()
