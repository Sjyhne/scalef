"""SR evaluation figures: full-frame comparisons, fixed-spot crops, multi-sample summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.gridspec import GridSpec


def _clip01(arr: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(arr, dtype=np.float32), 0.0, 1.0)


def _upscale_spot(arr_hwc: np.ndarray, target_px: int = 256) -> np.ndarray:
    h, w = arr_hwc.shape[:2]
    if h <= 0 or w <= 0:
        return arr_hwc
    scale = max(1, int(round(float(target_px) / max(h, w))))
    if scale <= 1:
        return arr_hwc
    out_w, out_h = w * scale, h * scale
    return cv2.resize(arr_hwc, (out_w, out_h), interpolation=cv2.INTER_NEAREST)


def _crop_hwc(arr_hwc: np.ndarray, slices: list[int] | tuple[int, ...]) -> np.ndarray:
    y0, y1, x0, x1 = (int(v) for v in slices)
    return arr_hwc[y0:y1, x0:x1]


def _draw_spot_rect(ax, slices: list[int] | tuple[int, ...], *, color: str = "#FFD000") -> None:
    y0, y1, x0, x1 = (int(v) for v in slices)
    rect = patches.Rectangle(
        (x0, y0),
        x1 - x0,
        y1 - y0,
        linewidth=2,
        edgecolor=color,
        facecolor="none",
    )
    ax.add_patch(rect)


def save_eval_visualizations(
    output_dir: Path,
    *,
    lr_hwc: np.ndarray,
    bilinear_hwc: np.ndarray,
    pred_hwc: np.ndarray,
    gt_hwc: np.ndarray,
    image_metrics: dict[str, Any],
    fixed_spot: dict[str, Any] | None = None,
    sample_label: str = "",
) -> None:
    """Write full SR comparison + fixed-spot crop figures into ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lr = _clip01(lr_hwc)
    bil = _clip01(bilinear_hwc)
    pred = _clip01(pred_hwc)
    gt = _clip01(gt_hwc)

    model_psnr = float(image_metrics.get("model_psnr", 0.0))
    bilinear_psnr = float(image_metrics.get("bilinear_psnr", 0.0))
    spot_slices = fixed_spot.get("slices_hr") if fixed_spot else None

    prefix = f"{sample_label} — " if sample_label else ""

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes[0, 0].imshow(lr)
    axes[0, 0].set_title(f"{prefix}LR input", fontsize=13, fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(bil)
    axes[0, 1].set_title(
        f"Bilinear\nfull PSNR {bilinear_psnr:.2f} dB",
        fontsize=13,
        fontweight="bold",
    )
    axes[0, 1].axis("off")
    if spot_slices:
        _draw_spot_rect(axes[0, 1], spot_slices)

    axes[1, 0].imshow(pred)
    spot_psnr = ""
    if fixed_spot:
        spot_psnr = f"\nspot PSNR {fixed_spot['model_psnr']:.2f} dB"
    axes[1, 0].set_title(
        f"Model SR\nfull PSNR {model_psnr:.2f} dB{spot_psnr}",
        fontsize=13,
        fontweight="bold",
    )
    axes[1, 0].axis("off")
    if spot_slices:
        _draw_spot_rect(axes[1, 0], spot_slices)

    axes[1, 1].imshow(gt)
    axes[1, 1].set_title(f"{prefix}HR ground truth", fontsize=13, fontweight="bold")
    axes[1, 1].axis("off")
    if spot_slices:
        _draw_spot_rect(axes[1, 1], spot_slices)

    plt.tight_layout(pad=2.0)
    plt.savefig(output_dir / "comparison.png", bbox_inches="tight", pad_inches=0.1, dpi=300)
    plt.close()

    for name, arr in (
        ("model_output_aligned.png", pred),
        ("ground_truth.png", gt),
        ("bilinear_baseline.png", bil),
        ("lr_original.png", lr),
    ):
        plt.figure(figsize=(8, 8))
        plt.imshow(arr)
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(output_dir / name, bbox_inches="tight", pad_inches=0, dpi=300)
        plt.close()

    if not fixed_spot or not spot_slices:
        return

    bil_s = _upscale_spot(_crop_hwc(bil, spot_slices))
    pred_s = _upscale_spot(_crop_hwc(pred, spot_slices))
    gt_s = _upscale_spot(_crop_hwc(gt, spot_slices))
    err_bil = np.abs(bil_s - gt_s).mean(axis=-1)
    err_pred = np.abs(pred_s - gt_s).mean(axis=-1)

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 4, figure=fig, height_ratios=[1.0, 0.85])

    panels = [
        (gs[0, 0], bil_s, f"Bilinear spot\nPSNR {fixed_spot['bilinear_psnr']:.2f} dB"),
        (gs[0, 1], pred_s, f"Model spot\nPSNR {fixed_spot['model_psnr']:.2f} dB"),
        (gs[0, 2], gt_s, f"GT spot\n{fixed_spot['hr_pixels']}×{fixed_spot['hr_pixels']} HR px"),
        (gs[0, 3], np.clip(err_pred - err_bil, 0, 1), f"Δ|err| (bil−model)\nSSIM {fixed_spot['model_ssim']:.3f}"),
    ]
    for spec, img, title in panels:
        ax = fig.add_subplot(spec)
        cmap = "magma" if img.ndim == 2 else None
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")

    ax_b = fig.add_subplot(gs[1, 0:2])
    ax_b.imshow(err_bil, cmap="magma", vmin=0, vmax=max(0.05, err_bil.max()))
    ax_b.set_title(f"Bilinear |error|  MAE {fixed_spot['bilinear_mae']:.4f}", fontweight="bold")
    ax_b.axis("off")

    ax_m = fig.add_subplot(gs[1, 2:4])
    ax_m.imshow(err_pred, cmap="magma", vmin=0, vmax=max(0.05, err_pred.max()))
    ax_m.set_title(
        f"Model |error|  MAE {fixed_spot['model_mae']:.4f}  "
        f"LPIPS {fixed_spot['model_lpips']:.3f}",
        fontweight="bold",
    )
    ax_m.axis("off")

    fig.suptitle(
        f"{prefix}Fixed-spot eval (center {fixed_spot['hr_pixels']}×{fixed_spot['hr_pixels']} HR px)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(pad=1.5)
    plt.savefig(output_dir / "spot_comparison.png", bbox_inches="tight", pad_inches=0.1, dpi=300)
    plt.close()

    metrics_path = output_dir / "spot_metrics.json"
    metrics_path.write_text(json.dumps(fixed_spot, indent=2), encoding="utf-8")


def create_spot_summary_visualization(all_results: list[dict], output_dir: Path) -> None:
    """Bar charts of fixed-spot metrics across samples."""
    rows = []
    for r in all_results:
        spot = (r.get("image_metrics") or {}).get("fixed_spot") or {}
        if not spot:
            continue
        rows.append(
            {
                "idx": r.get("sample_idx", len(rows)),
                "label": str((r.get("sample_info") or {}).get("sample_id", r.get("sample_idx"))),
                **spot,
            }
        )
    if not rows:
        return

    output_dir = Path(output_dir)
    idx = [row["idx"] for row in rows]
    labels = [row["label"] for row in rows]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    x = np.arange(len(rows))

    axes[0, 0].bar(x - 0.15, [r["model_psnr"] for r in rows], 0.3, label="Model", color="#2563eb")
    axes[0, 0].bar(x + 0.15, [r["bilinear_psnr"] for r in rows], 0.3, label="Bilinear", color="#f97316")
    axes[0, 0].set_xticks(x, labels, rotation=45, ha="right")
    axes[0, 0].set_ylabel("PSNR (dB)")
    axes[0, 0].set_title("Fixed-spot PSNR")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    imp = [r["psnr_improvement"] for r in rows]
    axes[0, 1].bar(x, imp, color=["#16a34a" if v >= 0 else "#dc2626" for v in imp])
    axes[0, 1].axhline(0, color="black", linewidth=0.8)
    axes[0, 1].set_xticks(x, labels, rotation=45, ha="right")
    axes[0, 1].set_ylabel("Δ PSNR (dB)")
    axes[0, 1].set_title("Spot PSNR improvement (model − bilinear)")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].bar(x - 0.15, [r["model_ssim"] for r in rows], 0.3, label="Model", color="#7c3aed")
    axes[1, 0].bar(x + 0.15, [r["bilinear_ssim"] for r in rows], 0.3, label="Bilinear", color="#f97316")
    axes[1, 0].set_xticks(x, labels, rotation=45, ha="right")
    axes[1, 0].set_ylabel("SSIM")
    axes[1, 0].set_title("Fixed-spot SSIM")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].bar(x - 0.15, [r["model_lpips"] for r in rows], 0.3, label="Model", color="#92400e")
    axes[1, 1].bar(x + 0.15, [r["bilinear_lpips"] for r in rows], 0.3, label="Bilinear", color="#f97316")
    axes[1, 1].set_xticks(x, labels, rotation=45, ha="right")
    axes[1, 1].set_ylabel("LPIPS")
    axes[1, 1].set_title("Fixed-spot LPIPS (lower is better)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "spot_summary_metrics.png", bbox_inches="tight", pad_inches=0.1, dpi=300)
    plt.close()


def create_sr_sample_grid(output_dir: Path, all_results: list[dict], *, thumb_px: int = 256) -> None:
    """Mosaic per-sample SR outputs (spot crop if available, else center crop)."""
    tiles: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    for r in all_results:
        idx = int(r.get("sample_idx", len(tiles)))
        sample_dir = Path(output_dir) / f"sample_{idx:03d}"
        pred_path = sample_dir / "model_output_aligned.png"
        gt_path = sample_dir / "ground_truth.png"
        if not pred_path.is_file() or not gt_path.is_file():
            continue
        pred = cv2.cvtColor(cv2.imread(str(pred_path)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        gt = cv2.cvtColor(cv2.imread(str(gt_path)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        spot = (r.get("image_metrics") or {}).get("fixed_spot") or {}
        if spot.get("slices_hr"):
            pred = _crop_hwc(pred, spot["slices_hr"])
            gt = _crop_hwc(gt, spot["slices_hr"])
        else:
            cy, cx = pred.shape[0] // 2, pred.shape[1] // 2
            half = min(pred.shape[0], pred.shape[1]) // 8
            pred = pred[cy - half : cy + half, cx - half : cx + half]
            gt = gt[cy - half : cy + half, cx - half : cx + half]
        pred = _upscale_spot(pred, thumb_px)
        gt = _upscale_spot(gt, thumb_px)
        label = str((r.get("sample_info") or {}).get("sample_id", idx))
        tiles.append((label, pred, gt, np.abs(pred - gt).mean(axis=-1)))

    if not tiles:
        return

    n = len(tiles)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols * 3, figsize=(cols * 4.5, rows * 3.5))
    if rows == 1 and cols * 3 == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols * 3 == 1:
        axes = axes.reshape(-1, 1)

    for i, (label, pred, gt, err) in enumerate(tiles):
        r, c0 = divmod(i, cols)
        base = c0 * 3
        for j, (img, title) in enumerate(
            (
                (pred, f"{label}\nmodel"),
                (gt, "GT"),
                (err, "|err|"),
            )
        ):
            ax = axes[r, base + j]
            ax.imshow(img if img.ndim == 3 else img, cmap=None if img.ndim == 3 else "magma")
            ax.set_title(title, fontsize=9)
            ax.axis("off")

    total_axes = rows * cols * 3
    for k in range(len(tiles) * 3, total_axes):
        r, c = divmod(k, cols * 3)
        axes[r, c].axis("off")

    fig.suptitle("SR spot outputs across samples", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "sr_spot_grid.png", bbox_inches="tight", pad_inches=0.1, dpi=200)
    plt.close()


def visualize_benchmark_runs(summary_json: Path, out_dir: Path | None = None) -> Path:
    """Plot fixed-spot metrics vs LR size from ``benchmark_psf_datasets`` summary.json."""
    summary_json = Path(summary_json)
    data = json.loads(summary_json.read_text(encoding="utf-8"))
    results = data.get("results") or []
    rows = []
    for row in results:
        spot = row.get("fixed_spot") or {}
        if not spot and row.get("spot_model_psnr") is not None:
            spot = {
                "model_psnr": row.get("spot_model_psnr"),
                "bilinear_psnr": row.get("spot_bilinear_psnr"),
                "psnr_improvement": row.get("spot_psnr_improvement"),
                "model_ssim": row.get("spot_model_ssim"),
                "model_lpips": row.get("spot_model_lpips"),
            }
        if not spot:
            continue
        rows.append({**row, "spot": spot})
    if not rows:
        raise ValueError(f"No fixed_spot metrics in {summary_json}")

    out_dir = Path(out_dir or summary_json.parent)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_lr: dict[int, list[dict]] = {}
    for row in rows:
        lr = int(row.get("lr_size") or 0)
        by_lr.setdefault(lr, []).append(row)

    lr_sizes = sorted(by_lr)
    model_psnr = [np.mean([r["spot"]["model_psnr"] for r in by_lr[s]]) for s in lr_sizes]
    bil_psnr = [np.mean([r["spot"]["bilinear_psnr"] for r in by_lr[s]]) for s in lr_sizes]
    full_psnr = [np.mean([r.get("model_psnr") or np.nan for r in by_lr[s]]) for s in lr_sizes]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(lr_sizes, model_psnr, "o-", label="Model spot PSNR", linewidth=2)
    axes[0].plot(lr_sizes, bil_psnr, "s--", label="Bilinear spot PSNR", linewidth=2)
    axes[0].set_xlabel("LR size (px)")
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].set_title("Fixed-spot PSNR vs LR context size")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(lr_sizes, model_psnr, "o-", label="Spot PSNR", linewidth=2)
    axes[1].plot(lr_sizes, full_psnr, "^--", label="Full-frame PSNR", linewidth=2)
    axes[1].set_xlabel("LR size (px)")
    axes[1].set_ylabel("PSNR (dB)")
    axes[1].set_title("Spot vs full-frame model PSNR")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    out_path = out_dir / "spot_vs_lr_size.png"
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.1, dpi=300)
    plt.close()
    return out_path
