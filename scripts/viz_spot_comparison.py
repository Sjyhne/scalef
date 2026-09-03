#!/usr/bin/env python3
"""Side-by-side HR crop: ground truth vs bilinear vs full-field vs fused-k.

Reads the GeoTIFFs written by --no_qgis_export=off (qgis/hr_gt.tif,
qgis/sr_pred.tif, qgis/s2_bilinear.tif), which are co-registered, so the same
pixel window is directly comparable across runs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import rasterio  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def _read(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)
    arr = np.transpose(arr, (1, 2, 0))
    if arr.max() > 1.5:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def _pick_window(hr: np.ndarray, size: int, stride: int = 64) -> tuple[int, int]:
    """Highest-detail window, so the methods actually differ visibly."""
    gray = hr.mean(axis=2)
    gy, gx = np.gradient(gray)
    energy = gy**2 + gx**2
    h, w = gray.shape
    best, best_rc = -1.0, (0, 0)
    for r in range(0, max(1, h - size), stride):
        for c in range(0, max(1, w - size), stride):
            score = float(energy[r:r + size, c:c + size].mean())
            if score > best:
                best, best_rc = score, (r, c)
    return best_rc


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    return float("inf") if mse == 0 else 10.0 * np.log10(1.0 / mse)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="*", default=[
        "viz_lr512_full", "viz_lr512_k2", "viz_lr512_k4",
    ], help="Run names, in panel order after HR/bilinear.")
    ap.add_argument("--labels", nargs="*", default=["full frame", "k2", "k4"])
    ap.add_argument("--sample-dir", default="single_samples/asker/sample")
    ap.add_argument("--size", type=int, default=256, help="HR crop side.")
    ap.add_argument("--row", type=int, default=-1)
    ap.add_argument("--col", type=int, default=-1)
    ap.add_argument("--stretch", type=float, default=2.0,
                    help="Percentile clip per side for display only.")
    ap.add_argument("--out", default="single_samples/sweep_results/spot_256_comparison.png")
    args = ap.parse_args()

    base = ROOT / args.sample_dir
    qgis = [base / r / "qgis" for r in args.runs]
    missing = [str(q) for q in qgis if not (q / "sr_pred.tif").is_file()]
    if missing:
        raise SystemExit("Missing sr_pred.tif for: " + ", ".join(missing))

    hr = _read(qgis[0] / "hr_gt.tif")
    bil = _read(qgis[0] / "s2_bilinear.tif")
    preds = [_read(q / "sr_pred.tif") for q in qgis]

    size = int(args.size)
    if args.row >= 0 and args.col >= 0:
        r0, c0 = args.row, args.col
    else:
        r0, c0 = _pick_window(hr, size)
    sl = (slice(r0, r0 + size), slice(c0, c0 + size))
    print(f"crop = rows {r0}:{r0 + size}, cols {c0}:{c0 + size} of {hr.shape[:2]}")

    hr_c = hr[sl]
    panels = [("HR ground truth", hr_c, None), ("bilinear", bil[sl], _psnr(bil[sl], hr_c))]
    for label, p in zip(args.labels, preds):
        panels.append((label, p[sl], _psnr(p[sl], hr_c)))

    # S2 scenes sit in a narrow dark band; stretch every panel by the same
    # HR-derived bounds so differences are visible and still comparable.
    lo, hi = np.percentile(hr_c, [float(args.stretch), 100.0 - float(args.stretch)])
    span = max(float(hi - lo), 1e-6)

    def show(img: np.ndarray) -> np.ndarray:
        return np.clip((img - lo) / span, 0.0, 1.0)

    fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 4.6))
    for ax, (label, img, psnr) in zip(axes, panels):
        ax.imshow(show(img), interpolation="nearest")
        title = label if psnr is None else f"{label}\n{psnr:.2f} dB"
        ax.set_title(title, fontsize=13)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        f"asker LR512 — {size}x{size} HR crop at row {r0}, col {c0} "
        f"(2.5 m GSD, {float(args.stretch):.0f}-{100 - float(args.stretch):.0f}% stretch)",
        fontsize=14, y=1.02,
    )
    fig.tight_layout()
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
