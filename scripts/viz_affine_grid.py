#!/usr/bin/env python3
"""HR mosaics and per-tile warp fields (dx, dy, rotation) for the affine grid.

Reads ``model_output_aligned.png`` and ``affines.json`` from the 1+4+16
partition of the asker LR2048 window. Translations are the centre-point
shift in metres; rotation is the polar-decomposition angle of the 2x2
linear part, in degrees (counter-clockwise, x=east, y=south in INR
coords, then flipped so the map y-axis is north).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "single_samples" / "asker" / "sample"
BLOWUP_M = 50.0
BLOWUP_DEG = 5.0


def _run(name: str) -> Path:
    return RUNS / name


def _load_rgb(name: str) -> np.ndarray | None:
    p = _run(name) / "model_output_aligned.png"
    if not p.is_file():
        return None
    im = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    return im


def _load_aff(name: str) -> dict | None:
    p = _run(name) / "affines.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text())


def _rotation_deg(matrix_2x3: list) -> float:
    """Polar-decomposition rotation of [[a,b],[d,e]], degrees, north-up.

    INR y increases south. A positive INR rotation (CCW in x-east/y-south)
    is clockwise on a north-up map, so the sign is flipped.
    """
    a, b, _tx = matrix_2x3[0]
    d, e, _ty = matrix_2x3[1]
    m = np.array([[a, b], [d, e]], dtype=np.float64)
    u, _s, vt = np.linalg.svd(m)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    alpha_inr = float(np.degrees(np.arctan2(r[1, 0], r[0, 0])))
    return -alpha_inr


def _frame_warp(dump: dict, frame: int) -> tuple[float, float, float]:
    fr = dump["frames"][frame]
    return (
        float(fr["center_shift_east_m"]),
        float(fr["center_shift_north_m"]),
        _rotation_deg(fr["matrix_2x3"]),
    )


def _grid_names(side: int) -> list[list[str]]:
    n = 2048 // side
    return [[f"aff_g{side}_y{iy}_x{ix}" if side < 2048 else "aff_g2048"
             for ix in range(n)] for iy in range(n)]


def _mosaic_hr(side: int, canvas: int = 2048) -> np.ndarray:
    names = _grid_names(side)
    n = len(names)
    cell = canvas // n
    out = np.full((canvas, canvas, 3), 0.15, dtype=np.float32)
    for iy, row in enumerate(names):
        for ix, name in enumerate(row):
            im = _load_rgb(name)
            if im is None:
                continue
            tile = np.asarray(Image.fromarray((im * 255).astype(np.uint8)).resize(
                (cell, cell), Image.Resampling.LANCZOS
            ), dtype=np.float32) / 255.0
            out[iy * cell:(iy + 1) * cell, ix * cell:(ix + 1) * cell] = tile
    return out


def _field(side: int, frame: int, kind: str) -> np.ndarray:
    names = _grid_names(side)
    n = len(names)
    arr = np.full((n, n), np.nan, dtype=np.float32)
    for iy, row in enumerate(names):
        for ix, name in enumerate(row):
            dump = _load_aff(name)
            if dump is None:
                continue
            dx, dy, ang = _frame_warp(dump, frame)
            if abs(dx) > BLOWUP_M or abs(dy) > BLOWUP_M or abs(ang) > BLOWUP_DEG:
                continue
            arr[iy, ix] = {"dx": dx, "dy": dy, "ang": ang}[kind]
    return arr


def _mean_field(side: int, n_frames: int, kind: str) -> np.ndarray:
    acc = []
    for f in range(n_frames):
        acc.append(_field(side, f, kind))
    stacked = np.stack(acc, axis=0)
    with np.errstate(all="ignore"):
        return np.nanmean(stacked, axis=0)


def _stretch(rgb: np.ndarray, pct: float = 2.0) -> np.ndarray:
    lo, hi = np.percentile(rgb, [pct, 100.0 - pct], axis=(0, 1))
    hi = np.maximum(hi, lo + 1e-6)
    return np.clip((rgb - lo) / (hi - lo), 0, 1)


def _diverging(ax, data: np.ndarray, vmax: float, cmap: str, title: str) -> None:
    n = data.shape[0]
    extent = [0, 2048, 2048, 0]  # col, row in LR px of the parent window
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax) if vmax > 0 else None
    im = ax.imshow(data, cmap=cmap, norm=norm, extent=extent, interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    for i in range(1, n):
        ax.axhline(i * (2048 / n), color="white", lw=0.4, alpha=0.5)
        ax.axvline(i * (2048 / n), color="white", lw=0.4, alpha=0.5)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path,
                    default=ROOT / "single_samples" / "sweep_results")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    big = _load_aff("aff_g2048")
    if big is None:
        raise SystemExit("aff_g2048/affines.json missing")
    n_frames = len(big["frames"])

    hr512 = _stretch(_mosaic_hr(512))
    hr1024 = _stretch(_mosaic_hr(1024))
    hr2048 = _stretch(_mosaic_hr(2048))

    kinds = [
        ("dx", "east shift (m)", "RdBu_r"),
        ("dy", "north shift (m)", "RdBu_r"),
        ("ang", "rotation (deg, CCW)", "PiYG"),
    ]
    means = {side: {k: _mean_field(side, n_frames, k) for k, _, _ in kinds}
             for side in (512, 1024, 2048)}
    vmax = {}
    for k, _, _ in kinds:
        vals = np.concatenate([means[s][k].ravel() for s in (512, 1024, 2048)])
        vals = vals[np.isfinite(vals)]
        vmax[k] = float(np.nanpercentile(np.abs(vals), 95)) if vals.size else 1.0
        vmax[k] = max(vmax[k], 0.15)

    fig, axes = plt.subplots(4, 3, figsize=(12.5, 16.2))
    for ax, im, title in zip(
        axes[0],
        [hr512, hr1024, hr2048],
        ["512 mosaic (16 INRs)", "1024 mosaic (4 INRs)", "2048 (1 INR)"],
    ):
        ax.imshow(im, extent=[0, 2048, 2048, 0])
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        if title.startswith("512"):
            for i in range(1, 4):
                ax.axhline(i * 512, color="white", lw=0.5, alpha=0.7)
                ax.axvline(i * 512, color="white", lw=0.5, alpha=0.7)
        if title.startswith("1024"):
            ax.axhline(1024, color="white", lw=0.5, alpha=0.7)
            ax.axvline(1024, color="white", lw=0.5, alpha=0.7)

    for row, (kind, label, cmap) in enumerate(kinds, start=1):
        for col, side in enumerate((512, 1024, 2048)):
            _diverging(
                axes[row, col],
                means[side][kind],
                vmax[kind],
                cmap,
                f"{side}  {label}",
            )

    fig.suptitle(
        "Asker grid: HR decode and mean warp over frames that did not diverge "
        f"(|shift|<{BLOWUP_M:.0f} m, |α|<{BLOWUP_DEG:.0f}°)",
        fontsize=12,
        y=0.995,
    )
    fig.tight_layout()
    hr_path = args.out_dir / "affine_grid_hr_warp.png"
    fig.savefig(hr_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    # Per-frame 512 atlas so a single mean does not hide disagreement.
    fig, axes = plt.subplots(n_frames, 3, figsize=(8.2, 1.15 * n_frames))
    if n_frames == 1:
        axes = np.array([axes])
    for f in range(n_frames):
        frozen = big["frames"][f]["frozen"]
        for col, (kind, label, cmap) in enumerate(kinds):
            data = _field(512, f, kind)
            _diverging(
                axes[f, col],
                data,
                vmax[kind],
                cmap,
                f"f{f:02d}{'*' if frozen else ''}  {label}" if col == 0 else label,
            )
    fig.suptitle("512-tile warp per S2 frame  (* = frozen identity)", fontsize=11)
    fig.tight_layout()
    fr_path = args.out_dir / "affine_grid_frames.png"
    fig.savefig(fr_path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {hr_path}")
    print(f"Wrote {fr_path}")
    missing = [
        f"aff_g512_y{iy}_x{ix}"
        for iy in range(4) for ix in range(4)
        if not (_run(f"aff_g512_y{iy}_x{ix}") / "affines.json").is_file()
    ]
    if missing:
        print("missing tiles (gray / NaN):", ", ".join(missing))


if __name__ == "__main__":
    main()
