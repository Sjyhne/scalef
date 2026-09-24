#!/usr/bin/env python3
"""Compare date-boundary seams across training variants from ``run_boundary_variants.py``.

For each tile pair and variant this reads the untouched per-tile ``sr_pred.tif``
files (no feather, no colour correction) and reports, over their shared overlap:

* ``overlap_mae``  mean |A - B| over RGB (total disagreement)
* ``color_bias``   mean over channels of |mean(A) - mean(B)| (brightness/colour shift)
* ``detail_mae``   overlap MAE after removing that per-channel mean offset

It also renders a grid of 2 x 2 km midpoint hard cuts (rows = pairs, cols = variants)
with Sentinel Hub HighlightCompress(0, 0.4).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from rasterio.windows import Window  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_boundary_variants import (  # noqa: E402
    DEFAULT_MANIFEST,
    DEFAULT_PAIRS,
    DEFAULT_PLAN,
    DEFAULT_PREFIX,
    swapped_dates,
    tile_id,
)

SCALE = 4
CROP_LR = 200
OUT_DIR = ROOT / "production/national_2025/qa/boundary_variants"


def sr_path(parent: str, run_prefix: str, tid: str) -> Path:
    return ROOT / "single_samples" / parent / "sample" / f"{run_prefix}_{tid}" / "qgis/sr_pred.tif"


def read_lr_window(path: Path, tile: dict, r0: int, c0: int, r1: int, c1: int) -> np.ndarray:
    """Read parent-LR window [r0:r1, c0:c1] from a tile's SR GeoTIFF as HWC."""
    tr, tc = int(tile["row_off"]), int(tile["col_off"])
    window = Window((c0 - tc) * SCALE, (r0 - tr) * SCALE, (c1 - c0) * SCALE, (r1 - r0) * SCALE)
    with rasterio.open(path) as src:
        data = src.read((1, 2, 3), window=window, boundless=True, fill_value=0)
    return np.moveaxis(data.astype(np.float32), 0, -1)


def overlap_bounds(a: dict, b: dict) -> tuple[int, int, int, int]:
    r0 = max(a["row_off"], b["row_off"])
    c0 = max(a["col_off"], b["col_off"])
    r1 = min(a["row_off"] + a["side"], b["row_off"] + b["side"])
    c1 = min(a["col_off"] + a["side"], b["col_off"] + b["side"])
    return int(r0), int(c0), int(r1), int(c1)


def seam_metrics(a_img: np.ndarray, b_img: np.ndarray) -> dict:
    valid = np.any(a_img != 0, axis=-1) & np.any(b_img != 0, axis=-1)
    a, b = a_img[valid], b_img[valid]
    offset = a.mean(axis=0) - b.mean(axis=0)
    return {
        "overlap_mae": float(np.mean(np.abs(a - b))),
        "color_bias": float(np.mean(np.abs(offset))),
        "color_offset_rgb": [float(v) for v in offset],
        "detail_mae": float(np.mean(np.abs(a - b - offset))),
        "n_pixels": int(valid.sum()),
    }


def tone(rgb: np.ndarray) -> np.ndarray:
    x = np.clip(rgb / 0.4, 0.0, None)
    return np.clip(np.where(x <= 0.92, x, 0.92 + (x - 0.92) * (0.08 / 1.08)), 0.0, 1.0)


def hard_cut(a: dict, b: dict, path_a: Path, path_b: Path) -> tuple[np.ndarray, str]:
    if a["iy"] != b["iy"]:
        cy = (a["row_off"] + a["side"] + b["row_off"]) / 2
        cx = a["col_off"] + a["side"] / 2
        orientation = "h"
    else:
        cy = a["row_off"] + a["side"] / 2
        cx = (a["col_off"] + a["side"] + b["col_off"]) / 2
        orientation = "v"
    r0, c0 = int(round(cy - CROP_LR / 2)), int(round(cx - CROP_LR / 2))
    img_a = read_lr_window(path_a, a, r0, c0, r0 + CROP_LR, c0 + CROP_LR)
    img_b = read_lr_window(path_b, b, r0, c0, r0 + CROP_LR, c0 + CROP_LR)
    size = CROP_LR * SCALE
    yy, xx = np.indices((size, size))
    first = yy < size / 2 if orientation == "h" else xx < size / 2
    return np.where(first[..., None], img_a, img_b), orientation


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variants", nargs="+", default=["production", "baseline", "clear", "clear_laplace"])
    ap.add_argument("--prefix", default=DEFAULT_PREFIX)
    ap.add_argument("--production-prefix", default="prod_k4_ovl12_icm")
    ap.add_argument("--tag", default="v1")
    args = ap.parse_args()

    manifest = json.loads(DEFAULT_MANIFEST.read_text())
    assignment = json.loads(DEFAULT_PLAN.read_text())["assignment"]
    parent = manifest["parent"]
    tiles = {(int(t["iy"]), int(t["ix"])): t for t in manifest["tiles"]}
    swap = swapped_dates(DEFAULT_PAIRS, parent, assignment)

    def prefix_for(variant: str) -> str:
        if variant == "production":
            return args.production_prefix
        return f"{args.prefix}_{variant.replace('+', '_')}"

    def dates_for(variant: str, xy) -> str:
        return swap[xy] if variant.endswith("+swap") else assignment[tile_id(parent, xy)]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report: dict = {"variants": args.variants, "pairs": []}
    fig, axes = plt.subplots(
        len(DEFAULT_PAIRS), len(args.variants),
        figsize=(4.2 * len(args.variants), 4.4 * len(DEFAULT_PAIRS)), dpi=130, squeeze=False,
    )
    for row, (xa, xb) in enumerate(DEFAULT_PAIRS):
        a, b = tiles[xa], tiles[xb]
        ta, tb = tile_id(parent, xa), tile_id(parent, xb)
        entry = {"a": list(xa), "b": list(xb), "variants": {}}
        for col, variant in enumerate(args.variants):
            pa, pb = sr_path(parent, prefix_for(variant), ta), sr_path(parent, prefix_for(variant), tb)
            ax = axes[row, col]
            ax.set_axis_off()
            if not (pa.is_file() and pb.is_file()):
                ax.set_title(f"{variant}\nmissing", fontsize=9)
                continue
            r0, c0, r1, c1 = overlap_bounds(a, b)
            metrics = seam_metrics(
                read_lr_window(pa, a, r0, c0, r1, c1), read_lr_window(pb, b, r0, c0, r1, c1)
            )
            da, db = dates_for(variant, xa), dates_for(variant, xb)
            metrics["dates"] = [da, db]
            entry["variants"][variant] = metrics

            img, orientation = hard_cut(a, b, pa, pb)
            size = CROP_LR * SCALE
            ax.imshow(tone(img), interpolation="nearest")
            line = ax.axhline(size / 2, color="#00F5FF", lw=1.6) if orientation == "h" else ax.axvline(
                size / 2, color="#00F5FF", lw=1.6
            )
            line.set_path_effects([pe.Stroke(linewidth=3, foreground="black"), pe.Normal()])
            ax.set_title(
                f"{variant} · {xa}↔{xb}\n{da} | {db}\n"
                f"MAE {metrics['overlap_mae']:.4f} · bias {metrics['color_bias']:.4f} · "
                f"detail {metrics['detail_mae']:.4f}",
                fontsize=8,
            )
        report["pairs"].append(entry)

    summary = {}
    for variant in args.variants:
        rows = [p["variants"][variant] for p in report["pairs"] if variant in p["variants"]]
        if rows:
            summary[variant] = {
                key: float(np.mean([r[key] for r in rows]))
                for key in ("overlap_mae", "color_bias", "detail_mae")
            } | {"n_pairs": len(rows)}
    report["mean_over_pairs"] = summary

    fig.suptitle(
        "32VNM date-boundary pairs · raw per-tile sr_pred, midpoint hard cut, no feather/colour correction\n"
        "Overlap metrics in reflectance over the full 640 m shared strip",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    png = OUT_DIR / f"boundary_variants_{args.tag}.png"
    fig.savefig(png, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    out_json = OUT_DIR / f"boundary_variants_{args.tag}.json"
    out_json.write_text(json.dumps(report, indent=2))
    print(png)
    print(out_json)
    for variant, row in summary.items():
        print(
            f"{variant:>16}: MAE {row['overlap_mae']:.4f}  bias {row['color_bias']:.4f}  "
            f"detail {row['detail_mae']:.4f}  (n={row['n_pairs']})"
        )


if __name__ == "__main__":
    main()
