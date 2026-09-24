#!/usr/bin/env python3
"""Visual A/B: unmasked vs cloud-masked SR on selected MGRS cells.

Does not overwrite the unmasked ``prod_k4_*`` trains. Reads
``prod_k4_<tile>/qgis/sr_pred.tif`` vs ``prod_k4_cloudmask_<tile>/qgis/sr_pred.tif``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def _sr(parent: str, tile_id: str, prefix: str) -> Path:
    return (
        ROOT
        / "single_samples"
        / parent
        / "sample"
        / f"{prefix}_{tile_id}"
        / "qgis"
        / "sr_pred.tif"
    )


def _read_rgb(path: Path, out_hw: int = 512) -> np.ndarray:
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(path) as src:
        rgb = src.read(
            indexes=(1, 2, 3),
            out_shape=(3, out_hw, out_hw),
            resampling=Resampling.bilinear,
        )
    return np.clip(np.transpose(rgb, (1, 2, 0)).astype(np.float32), 0.0, 1.0)


def _label(ax, title: str) -> None:
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parent", default="32VNM")
    ap.add_argument("--unmasked-prefix", default="prod_k4")
    ap.add_argument("--masked-prefix", default="prod_k4_cloudmask")
    ap.add_argument(
        "--tiles",
        nargs="+",
        default=[
            "32VNM_t512_y04_x19",
            "32VNM_t512_y04_x18",
            "32VNM_t512_y04_x20",
            "32VNM_t512_y03_x19",
            "32VNM_t512_y05_x19",
        ],
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    import matplotlib.pyplot as plt

    rows = []
    stats = []
    for tid in args.tiles:
        old_p = _sr(args.parent, tid, args.unmasked_prefix)
        new_p = _sr(args.parent, tid, args.masked_prefix)
        if not old_p.is_file() or not new_p.is_file():
            stats.append(
                {
                    "tile_id": tid,
                    "ready": False,
                    "unmasked": old_p.is_file(),
                    "masked": new_p.is_file(),
                }
            )
            continue
        old = _read_rgb(old_p)
        new = _read_rgb(new_p)
        dmu = float(new.mean() - old.mean())
        mae = float(np.abs(new - old).mean())
        rows.append((tid, old, new))
        stats.append(
            {
                "tile_id": tid,
                "ready": True,
                "mean_unmasked": float(old.mean()),
                "mean_masked": float(new.mean()),
                "delta_mean": dmu,
                "mae": mae,
            }
        )

    if not rows:
        raise SystemExit("no paired tiles ready yet")

    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(10.5, 3.4 * n), squeeze=False)
    for i, (tid, old, new) in enumerate(rows):
        stretch_src = np.concatenate([old.reshape(-1, 3), new.reshape(-1, 3)], axis=0)
        lo = np.percentile(stretch_src, 2, axis=0)
        hi = np.percentile(stretch_src, 98, axis=0)
        scale = np.maximum(hi - lo, 1e-6)

        def _show(rgb: np.ndarray) -> np.ndarray:
            return np.clip((rgb - lo) / scale, 0.0, 1.0)

        diff = np.clip(0.5 + (new - old) * 3.0, 0.0, 1.0)
        _label(axes[i, 0], f"{tid}\nunmasked  μ={old.mean():.3f}")
        axes[i, 0].imshow(_show(old))
        _label(axes[i, 1], f"cloud-masked  μ={new.mean():.3f}")
        axes[i, 1].imshow(_show(new))
        _label(axes[i, 2], f"masked−unmasked  Δμ={new.mean()-old.mean():+.3f}")
        axes[i, 2].imshow(diff)

    fig.suptitle(
        f"{args.parent}: reconstruction loss without vs with per-frame cloud mask",
        fontsize=12,
    )
    fig.tight_layout()
    out = args.out
    if out is None:
        out = ROOT / "production" / "cloudmask_ab" / f"{args.parent}_hot_cells.png"
    elif not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    side = out.with_suffix(".json")
    side.write_text(json.dumps({"parent": args.parent, "tiles": stats, "png": str(out)}, indent=2) + "\n")
    print(f"Wrote {out}")
    print(f"Meta {side}")


if __name__ == "__main__":
    main()
