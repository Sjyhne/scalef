#!/usr/bin/env python3
"""Show the LR RGB revisits (and SCL cloudy mask) for one S2 package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def _window(meta: dict):
    from rasterio.windows import Window

    aoi = meta["aoi_window"]
    return Window(int(aoi["col_off"]), int(aoi["row_off"]), int(aoi["width"]), int(aoi["height"]))


def _read_rgb(path: Path, win) -> np.ndarray:
    import rasterio

    with rasterio.open(path) as src:
        if src.width == int(win.width) and src.height == int(win.height):
            arr = src.read(indexes=(1, 2, 3))
        else:
            arr = src.read(indexes=(1, 2, 3), window=win)
    rgb = np.transpose(arr.astype(np.float32), (1, 2, 0))
    if float(np.nanmax(rgb) if rgb.size else 0.0) > 1.5:
        rgb = rgb / 10000.0
    return np.clip(np.nan_to_num(rgb, nan=0.0), 0.0, 1.5)


def _read_mask(path: Path, win) -> np.ndarray:
    import rasterio

    with rasterio.open(path) as src:
        if src.width == int(win.width) and src.height == int(win.height):
            arr = src.read(1)
        else:
            arr = src.read(1, window=win)
    return arr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--s2-dir",
        type=Path,
        default=ROOT / "data/s2_revisits/national_2025/32VNM_t512_y04_x19",
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    s2_dir = args.s2_dir if args.s2_dir.is_absolute() else ROOT / args.s2_dir
    meta = json.loads((s2_dir / "meta.json").read_text())
    win = _window(meta)

    frames = []
    for fr in meta["frames"]:
        rgb = _read_rgb(s2_dir / fr["path"], win)
        mask_name = fr.get("cloud_mask")
        cloudy = None
        if mask_name and (s2_dir / mask_name).is_file():
            cloudy = _read_mask(s2_dir / mask_name, win) > 0
        frames.append((fr, rgb, cloudy))

    import matplotlib.pyplot as plt

    n = len(frames)
    ncols = min(5, n)
    nrows = 2 * int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.6 * (nrows / 2)), squeeze=False)
    for i, (fr, rgb, cloudy) in enumerate(frames):
        col = i % ncols
        block = (i // ncols) * 2
        lo = np.percentile(rgb, 2, axis=(0, 1))
        hi = np.percentile(rgb, 98, axis=(0, 1))
        show = np.clip((rgb - lo) / np.maximum(hi - lo, 1e-6), 0.0, 1.0)
        ax = axes[block, col]
        ax.imshow(show)
        date = str(fr.get("datetime") or fr["path"])[:10]
        scl = float(cloudy.mean()) if cloudy is not None else float("nan")
        stac = fr.get("eo:cloud_cover")
        ax.set_title(
            f"{date}  μ={rgb.mean():.3f}\n"
            f"SCL cloudy {100*scl:.0f}%  scene {stac:.0f}%",
            fontsize=8,
        )
        ax.set_axis_off()
        axm = axes[block + 1, col]
        if cloudy is not None:
            axm.imshow(show)
            overlay = np.zeros((*cloudy.shape, 4), dtype=np.float32)
            overlay[..., 0] = 1.0
            overlay[..., 3] = cloudy.astype(np.float32) * 0.55
            axm.imshow(overlay)
            axm.set_title(f"SCL mask  clear {100*(1-scl):.0f}%", fontsize=8)
        else:
            axm.set_title("no mask", fontsize=8)
        axm.set_axis_off()
    for ax in axes.ravel():
        if not getattr(ax, "has_data", lambda: True)():
            ax.set_axis_off()
    fig.suptitle(
        f"{s2_dir.name}: LR revisits (per-frame 2–98% stretch) + SCL cloudy overlay\n"
        "μ is unstretched mean reflectance — compare those, not the display gain",
        fontsize=11,
    )
    fig.tight_layout()
    out = args.out
    if out is None:
        out = ROOT / "production" / "cloudmask_ab" / f"{s2_dir.name}_lr_frames.png"
    elif not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
