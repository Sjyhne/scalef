#!/usr/bin/env python3
"""Sweep patch-grid rotation angles to maximize complete LR512 tiles.

Rotating the valid mask by -θ is equivalent to angling the 5.12 km grid by θ.
For each city, try angles in [0, 90) and report the best complete-patch count.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import rasterio
from scipy.ndimage import rotate as nd_rotate

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s2_dataset import FOCUS_PROJECT_BY_CITY, resolve_focus_project_dir  # noqa: E402

FOCUS_ROOT = ROOT / "data" / "nib_focus_1m_worldcover"  # default map output root

LR_PATCH = 512
LR_GSD_M = 10.0


def _count_complete(mask: np.ndarray, patch_lr: int, min_valid_frac: float) -> tuple[int, int]:
    """Return (n_complete, n_partial) for an axis-aligned grid on mask."""
    h, w = mask.shape
    n_complete = 0
    n_partial = 0
    for row_off in range(0, h - patch_lr + 1, patch_lr):
        for col_off in range(0, w - patch_lr + 1, patch_lr):
            patch = mask[row_off : row_off + patch_lr, col_off : col_off + patch_lr]
            vf = float(patch.mean())
            if vf >= min_valid_frac:
                n_complete += 1
            elif vf > 0.05:
                n_partial += 1
    return n_complete, n_partial


def _principal_angle_deg(mask: np.ndarray) -> float:
    """Angle of major axis of valid pixels (degrees, CCW from +x), in [0, 90)."""
    ys, xs = np.nonzero(mask > 0.5)
    if xs.size < 100:
        return 0.0
    coords = np.column_stack((xs.astype(np.float64), ys.astype(np.float64)))
    coords -= coords.mean(axis=0)
    # SVD: first right-singular vector = principal direction
    _, _, vt = np.linalg.svd(coords, full_matrices=False)
    ang = float(np.degrees(np.arctan2(vt[0, 1], vt[0, 0]))) % 180.0
    return ang % 90.0  # square tiles are 90°-periodic


def _sweep_city(
    mask: np.ndarray,
    angles: list[float],
    patch_lr: int,
    min_valid_frac: float,
) -> list[dict]:
    rows = []
    for ang in angles:
        if abs(ang) < 1e-9:
            rot = mask
        else:
            # Rotate mask by -ang so an axis-aligned grid matches a grid at +ang on original.
            rot = nd_rotate(mask, angle=-ang, reshape=True, order=0, cval=0.0)
            rot = (rot > 0.5).astype(np.float32)
        n_c, n_p = _count_complete(rot, patch_lr, min_valid_frac)
        rows.append({"angle_deg": float(ang), "n_complete": n_c, "n_partial": n_p})
    return rows


def _plot_best(
    city: str,
    mask: np.ndarray,
    angle_deg: float,
    patch_lr: int,
    min_valid_frac: float,
    out_png: Path,
) -> None:
    if abs(angle_deg) < 1e-9:
        rot = mask
    else:
        rot = nd_rotate(mask, angle=-angle_deg, reshape=True, order=0, cval=0.0)
        rot = (rot > 0.5).astype(np.float32)

    h, w = rot.shape
    stride = max(1, int(np.ceil(max(h, w) / 1800)))
    view = rot[::stride, ::stride]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(view, cmap="gray", vmin=0, vmax=1, extent=(0, w, h, 0), interpolation="nearest")

    n_complete = 0
    n_partial = 0
    for row_off in range(0, h - patch_lr + 1, patch_lr):
        for col_off in range(0, w - patch_lr + 1, patch_lr):
            patch = rot[row_off : row_off + patch_lr, col_off : col_off + patch_lr]
            vf = float(patch.mean())
            if vf >= min_valid_frac:
                color, lw, n_complete = "#2ca02c", 1.6, n_complete + 1
            elif vf > 0.05:
                color, lw, n_partial = "#ff7f0e", 0.7, n_partial + 1
            else:
                continue
            ax.add_patch(
                mpatches.Rectangle(
                    (col_off, row_off),
                    patch_lr,
                    patch_lr,
                    fill=False,
                    edgecolor=color,
                    linewidth=lw,
                )
            )

    ax.set_title(
        f"{city}: best grid angle {angle_deg:.1f}°  |  "
        f"complete {n_complete}  partial {n_partial}\n"
        f"(mask rotated −{angle_deg:.1f}° so grid is axis-aligned in this view)",
        fontsize=11,
    )
    ax.set_xlabel("rotated LR column")
    ax.set_ylabel("rotated LR row")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cities", nargs="+", default=list(FOCUS_PROJECT_BY_CITY))
    ap.add_argument("--patch-lr", type=int, default=LR_PATCH)
    ap.add_argument("--min-valid-frac", type=float, default=1.0)
    ap.add_argument("--angle-step", type=float, default=5.0, help="Sweep step in degrees")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=FOCUS_ROOT / "map" / "patch_grid_rotated_lr512",
    )
    args = ap.parse_args()

    angles = list(np.arange(0.0, 90.0, float(args.angle_step)))
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    city_results = []
    for city in args.cities:
        proj = resolve_focus_project_dir(city)
        if proj is None:
            raise FileNotFoundError(f"No NIB project dir for {city}")
        mask_path = proj / "valid_mask_10m.tif"
        if not mask_path.is_file():
            print(f"SKIP {city}: missing {mask_path}")
            continue
        with rasterio.open(mask_path) as ds:
            mask = (ds.read(1) > 0).astype(np.float32)

        pca_ang = _principal_angle_deg(mask)
        # Include PCA angle in the sweep if not already on the grid
        sweep_angles = sorted(set(angles) | {round(pca_ang, 2)})
        rows = _sweep_city(mask, sweep_angles, args.patch_lr, args.min_valid_frac)
        best = max(rows, key=lambda r: (r["n_complete"], -abs(r["angle_deg"])))
        base = next(r for r in rows if abs(r["angle_deg"]) < 1e-9)

        png = out_dir / f"{city}_best_angle_{best['angle_deg']:.1f}deg.png"
        _plot_best(city, mask, best["angle_deg"], args.patch_lr, args.min_valid_frac, png)

        # Angle-vs-count curve
        fig, ax = plt.subplots(figsize=(8, 3.5))
        xs = [r["angle_deg"] for r in rows]
        ys = [r["n_complete"] for r in rows]
        ax.plot(xs, ys, "-o", ms=4, color="#2ca02c")
        ax.axvline(best["angle_deg"], color="#d62728", ls="--", lw=1.2, label=f"best {best['angle_deg']:.1f}°")
        ax.axvline(pca_ang, color="#1f77b4", ls=":", lw=1.2, label=f"PCA {pca_ang:.1f}°")
        ax.set_xlabel("grid angle (deg)")
        ax.set_ylabel("complete patches")
        ax.set_title(f"{city}: complete {args.patch_lr}² tiles vs grid rotation")
        ax.legend(loc="best", fontsize=8)
        ax.set_xlim(0, 90)
        fig.tight_layout()
        curve_png = out_dir / f"{city}_angle_sweep.png"
        fig.savefig(curve_png, dpi=140)
        plt.close(fig)

        rec = {
            "city": city,
            "pca_angle_deg": round(pca_ang, 2),
            "axis_aligned_complete": base["n_complete"],
            "best_angle_deg": best["angle_deg"],
            "best_complete": best["n_complete"],
            "gain": best["n_complete"] - base["n_complete"],
            "sweep": rows,
            "best_png": str(png.relative_to(ROOT)),
            "curve_png": str(curve_png.relative_to(ROOT)),
        }
        city_results.append(rec)
        print(
            f"{city:12s}  0°={base['n_complete']:3d}  "
            f"best@{best['angle_deg']:5.1f}°={best['n_complete']:3d}  "
            f"gain={best['n_complete'] - base['n_complete']:+3d}  "
            f"PCA={pca_ang:5.1f}°"
        )

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "patch_lr": args.patch_lr,
        "patch_hr": args.patch_lr * 4,
        "patch_span_m": args.patch_lr * LR_GSD_M,
        "min_valid_frac": args.min_valid_frac,
        "angle_step_deg": args.angle_step,
        "total_axis_aligned": sum(r["axis_aligned_complete"] for r in city_results),
        "total_best_rotated": sum(r["best_complete"] for r in city_results),
        "cities": city_results,
    }
    out_json = out_dir / "rotation_sweep_summary.json"
    out_json.write_text(json.dumps(payload, indent=2))
    print(
        json.dumps(
            {
                "total_axis_aligned": payload["total_axis_aligned"],
                "total_best_rotated": payload["total_best_rotated"],
                "gain": payload["total_best_rotated"] - payload["total_axis_aligned"],
            },
            indent=2,
        )
    )
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
