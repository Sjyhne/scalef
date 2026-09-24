#!/usr/bin/env python3
"""Visual comparison of aligned degradation variants (area / s2_psf / s2_psf_m / psfm_2x).

Reads saved PNGs + metrics.json from degA_* runs (no retraining).

Example
-------
python scripts/render_degradation_comparison.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.visualize import _crop_hwc, _shared_display_stretch, _upscale_spot

CITIES = ["asker", "vennesla", "trondheim", "bergen", "rana", "tromso", "amli"]
VARIANTS = [
    ("area", "area"),
    ("s2_psf", "s2_psf"),
    ("s2_psf_m", "s2_psf_m"),
    ("psfm_2x", "psfm_2x"),
]
ZOOM_HR_PX = 128


def _run_dir(city: str, variant: str, lr_size: int) -> Path:
    return ROOT / "single_samples" / city / "sample" / f"degA_{variant}_lr{lr_size}"


def _load_rgb(path: Path) -> np.ndarray:
    arr = plt.imread(path)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return np.asarray(arr[..., :3], dtype=np.float32)


def _center_slices(h: int, w: int, size: int) -> tuple[int, int, int, int]:
    size = int(min(max(1, size), h, w))
    y0 = max(0, (h - size) // 2)
    x0 = max(0, (w - size) // 2)
    return y0, y0 + size, x0, x0 + size


def _metrics(city: str, variant: str, lr_size: int) -> dict:
    return json.loads((_run_dir(city, variant, lr_size) / "metrics.json").read_text())


def render_city(city: str, lr_size: int, out_dir: Path) -> Path | None:
    first = _run_dir(city, "area", lr_size)
    if not (first / "model_output_aligned.png").is_file():
        print(f"skip {city}: missing {first}")
        return None

    lr = _load_rgb(first / "lr_original.png")
    bil = _load_rgb(first / "bilinear_baseline.png")
    gt = _load_rgb(first / "ground_truth.png")
    preds = {}
    mets = {}
    for key, label in VARIANTS:
        d = _run_dir(city, key, lr_size)
        if not (d / "model_output_aligned.png").is_file():
            print(f"skip {city}/{key}")
            return None
        preds[key] = _load_rgb(d / "model_output_aligned.png")
        mets[key] = _metrics(city, key, lr_size)

    h, w = gt.shape[:2]
    zoom = _center_slices(h, w, ZOOM_HR_PX)

    panels_full = [lr, bil, *[preds[k] for k, _ in VARIANTS], gt]
    labels_full = [
        "LR input",
        "Bilinear",
        "area",
        "s2_psf",
        "s2_psf_m",
        "psfm_2x",
        "HR GT (aligned)",
    ]
    disp_full = _shared_display_stretch(*panels_full)

    zoom_src = [bil, *[preds[k] for k, _ in VARIANTS], gt]
    zoom_disp = _shared_display_stretch(*[_crop_hwc(p, zoom) for p in zoom_src])
    zoom_disp = [_upscale_spot(z, 256) for z in zoom_disp]
    zoom_labels = ["Bilinear", "area", "s2_psf", "s2_psf_m", "psfm_2x", "HR GT"]

    n_full = len(disp_full)
    fig = plt.figure(figsize=(2.35 * n_full, 8.6))
    gs = fig.add_gridspec(2, n_full, height_ratios=[1.15, 1.0], hspace=0.22, wspace=0.04)

    for i, (img, lab) in enumerate(zip(disp_full, labels_full)):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(np.clip(img, 0, 1))
        ax.set_xticks([])
        ax.set_yticks([])
        title = lab
        if lab in {"area", "s2_psf", "s2_psf_m", "psfm_2x"}:
            m = mets[lab]
            title = (
                f"{lab}\n"
                f"PSNR {m['psnr']['model']:.2f}  "
                f"SSIM {m['ssim']['model']:.3f}\n"
                f"LPIPS {m['lpips']['model']:.3f}"
            )
        elif lab == "Bilinear":
            m = mets["area"]
            title = (
                f"Bilinear\n"
                f"PSNR {m['psnr']['bilinear']:.2f}  "
                f"SSIM {m['ssim']['bilinear']:.3f}\n"
                f"LPIPS {m['lpips']['bilinear']:.3f}"
            )
        ax.set_title(title, fontsize=8)
        y0, y1, x0, x1 = zoom
        ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color="#FFD000", lw=1.2)

    # Zoom row: skip LR (col 0), start bilinear under full-frame bilinear.
    ax0 = fig.add_subplot(gs[1, 0])
    ax0.axis("off")
    ax0.text(0.5, 0.5, f"center\n{ZOOM_HR_PX}×{ZOOM_HR_PX}\nHR crop", ha="center", va="center", fontsize=9)
    for i, (img, lab) in enumerate(zip(zoom_disp, zoom_labels), start=1):
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(np.clip(img, 0, 1))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(lab, fontsize=8)

    fig.suptitle(
        f"{city}  lr{lr_size}  aligned NIB eval  —  HR→LR degradation comparison",
        fontsize=12,
        y=0.995,
    )
    out = out_dir / f"{city}_degradations.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return out


def render_metrics_overview(lr_size: int, out_dir: Path, cities: list[str]) -> Path:
    kept = []
    data = {k: {"psnr": [], "ssim": [], "lpips": []} for k, _ in VARIANTS}
    for city in cities:
        if not (_run_dir(city, "area", lr_size) / "metrics.json").is_file():
            continue
        kept.append(city)
        for key, _ in VARIANTS:
            m = _metrics(city, key, lr_size)
            data[key]["psnr"].append(m["psnr"]["model"])
            data[key]["ssim"].append(m["ssim"]["model"])
            data[key]["lpips"].append(m["lpips"]["model"])

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    x = np.arange(len(kept))
    width = 0.2
    colors = ["#4c4c4c", "#6b8cae", "#2ca02c", "#d62728"]
    for ax, metric, ylabel, invert in zip(
        axes,
        ["psnr", "ssim", "lpips"],
        ["PSNR (dB) ↑", "SSIM ↑", "LPIPS ↓"],
        [False, False, True],
    ):
        for i, (key, _) in enumerate(VARIANTS):
            ax.bar(x + (i - 1.5) * width, data[key][metric], width, label=key, color=colors[i])
        ax.set_xticks(x)
        ax.set_xticklabels(kept, rotation=30, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)
        if invert:
            ax.invert_yaxis()
        ax.legend(fontsize=7, loc="best")
    fig.suptitle(f"Aligned degradation bench  lr{lr_size}  (7 cities)", fontsize=12)
    fig.tight_layout()
    out = out_dir / "metrics_bars.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return out


def render_gallery(lr_size: int, out_dir: Path, cities: list[str]) -> Path:
    """One row per city: bilinear, area, psfm_2x, GT (zoom)."""
    rows = []
    for city in cities:
        d = _run_dir(city, "psfm_2x", lr_size)
        if not (d / "model_output_aligned.png").is_file():
            continue
        bil = _load_rgb(_run_dir(city, "area", lr_size) / "bilinear_baseline.png")
        area = _load_rgb(_run_dir(city, "area", lr_size) / "model_output_aligned.png")
        psf = _load_rgb(d / "model_output_aligned.png")
        gt = _load_rgb(_run_dir(city, "area", lr_size) / "ground_truth.png")
        h, w = gt.shape[:2]
        zoom = _center_slices(h, w, ZOOM_HR_PX)
        crops = [_crop_hwc(p, zoom) for p in (bil, area, psf, gt)]
        disp = _shared_display_stretch(*crops)
        disp = [_upscale_spot(z, 220) for z in disp]
        rows.append((city, disp))

    n = len(rows)
    fig, axes = plt.subplots(n, 4, figsize=(10.5, 2.35 * n))
    if n == 1:
        axes = np.array([axes])
    titles = ["Bilinear", "area", "psfm_2x", "HR GT"]
    for r, (city, imgs) in enumerate(rows):
        for c, img in enumerate(imgs):
            ax = axes[r, c]
            ax.imshow(np.clip(img, 0, 1))
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(titles[c], fontsize=10)
            if c == 0:
                ax.set_ylabel(city, fontsize=10)
    fig.suptitle(f"Center {ZOOM_HR_PX}×{ZOOM_HR_PX}  —  bilinear vs area vs psfm_2x vs GT", fontsize=12)
    fig.tight_layout()
    out = out_dir / "gallery_zoom.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cities", nargs="+", default=CITIES)
    p.add_argument("--lr-size", type=int, default=512)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "single_samples" / "sweep_results" / "degA_lr512_7cities" / "viz",
    )
    args = p.parse_args()

    cities = list(args.cities)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for city in cities:
        render_city(city, args.lr_size, args.out_dir)
    render_metrics_overview(args.lr_size, args.out_dir, cities)
    render_gallery(args.lr_size, args.out_dir, cities)
    print(f"Done → {args.out_dir}")


if __name__ == "__main__":
    main()
