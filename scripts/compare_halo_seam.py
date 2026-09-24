#!/usr/bin/env python3
"""Visual A|B seam: independent mosaic vs halo-retrained east/south tile.

Crops a strip across the original LR512 cut. Writes a PNG + a small JSON of
edge Δμ / MAE.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _read_rgb(path: Path) -> tuple[np.ndarray, object, object]:
    import rasterio

    with rasterio.open(path) as src:
        rgb = np.transpose(src.read(out_shape=(3, src.height, src.width)), (1, 2, 0))
        return np.clip(rgb.astype(np.float32), 0.0, 1.0), src.transform, src.crs


def _cut_x_from_west(west_transform, west_width: int) -> float:
    return float(west_transform.c) + float(west_width) * float(west_transform.a)


def _read_window(path: Path, bounds, width: int, height: int) -> np.ndarray:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_bounds
    from rasterio.warp import reproject

    dst = np.zeros((3, height, width), dtype=np.float32)
    dst_t = from_bounds(*bounds, width, height)
    with rasterio.open(path) as src:
        for i in range(3):
            reproject(
                source=src.read(i + 1),
                destination=dst[i],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_t,
                dst_crs=src.crs,
                resampling=Resampling.bilinear,
            )
    return np.clip(np.transpose(dst, (1, 2, 0)), 0.0, 1.0)


def _edge_stats(left: np.ndarray, right: np.ndarray) -> dict:
    """left is west tile's last column crop; right is east tile's first column."""
    a = left[:, -1, :]
    b = right[:, 0, :]
    d = np.abs(a - b)
    return {
        "edge_mae": float(d.mean()),
        "edge_p95": float(np.quantile(d, 0.95)),
        "delta_mean": float((a.mean() - b.mean())),
    }


def _strip_panel(left: np.ndarray, right: np.ndarray, *, mark_cut: bool = True) -> np.ndarray:
    panel = np.concatenate([left, right], axis=1)
    if mark_cut:
        c = left.shape[1]
        panel[:, max(0, c - 1) : c + 1, :] = np.array([1.0, 0.2, 0.1], dtype=np.float32)
    return panel


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--west", type=Path, required=True, help="West/north sr_pred.tif (frozen A).")
    ap.add_argument("--east-old", type=Path, required=True, help="Independent east/south sr_pred.tif.")
    ap.add_argument("--east-halo", type=Path, default=None, help="Halo-retrained east/south sr_pred.tif.")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pad-px", type=int, default=160, help="HR pixels each side of the cut.")
    ap.add_argument("--row-frac", type=float, default=0.45, help="Vertical crop center as fraction of height.")
    ap.add_argument("--row-h", type=int, default=420, help="Crop height in HR pixels.")
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rasterio

    west_rgb, west_t, _ = _read_rgb(args.west)
    cut_x = _cut_x_from_west(west_t, west_rgb.shape[1])
    px = float(west_t.a)
    py = float(abs(west_t.e))
    pad = int(args.pad_px)
    row_h = int(args.row_h)
    with rasterio.open(args.west) as src:
        top = float(src.bounds.top)
        bottom = float(src.bounds.bottom)
        height_m = top - bottom
    cy = top - float(args.row_frac) * height_m
    half_h = 0.5 * row_h * py
    bounds = (cut_x - pad * px, cy - half_h, cut_x + pad * px, cy + half_h)
    width = 2 * pad
    height = row_h

    left = _read_window(args.west, (bounds[0], bounds[1], cut_x, bounds[3]), pad, height)
    right_old = _read_window(args.east_old, (cut_x, bounds[1], bounds[2], bounds[3]), pad, height)
    before = _strip_panel(left, right_old)
    stats = {"before": _edge_stats(left, right_old)}

    panels = [before]
    titles = [f"independent  edge MAE {stats['before']['edge_mae']:.4f}"]
    if args.east_halo and Path(args.east_halo).is_file():
        right_h = _read_window(args.east_halo, (cut_x, bounds[1], bounds[2], bounds[3]), pad, height)
        after = _strip_panel(left, right_h)
        stats["after"] = _edge_stats(left, right_h)
        panels.append(after)
        titles.append(f"halo  edge MAE {stats['after']['edge_mae']:.4f}")
        diff = np.abs(before - after)
        panels.append(np.clip(diff * 4.0, 0, 1))
        titles.append("|independent − halo| ×4")

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.6))
    if n == 1:
        axes = [axes]
    for ax, im, title in zip(axes, panels, titles):
        ax.imshow(im)
        ax.set_title(title, fontsize=10)
        ax.set_axis_off()
    fig.suptitle("32VNM y04 x17 | x18  vertical cut  (red = original tile edge)", fontsize=11)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    stats_path = args.out.with_suffix(".json")
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")
    print(f"wrote {args.out}  {stats}", flush=True)


if __name__ == "__main__":
    main()
