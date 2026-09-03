"""SR evaluation figures: full-frame comparisons and fixed-spot crops."""

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


def _shared_display_stretch(*arrays: np.ndarray, p: float = 99.5) -> list[np.ndarray]:
    """Stretch panels with one vmax so dark S2 reflectance is visible without per-image color shifts."""
    vmax = 0.0
    for arr in arrays:
        if arr.size == 0:
            continue
        vmax = max(vmax, float(np.percentile(arr, p)))
    vmax = max(vmax, 1e-6)
    return [_clip01(np.asarray(arr, dtype=np.float32) / vmax) for arr in arrays]


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


def _metric_title(
    label: str,
    *,
    psnr: float | None = None,
    ssim: float | None = None,
    lpips: float | None = None,
    extra: str = "",
) -> str:
    lines = [label]
    if psnr is not None:
        lines.append(f"PSNR {psnr:.2f} dB")
    if ssim is not None:
        lines.append(f"SSIM {ssim:.3f}")
    if lpips is not None:
        lines.append(f"LPIPS {lpips:.3f}")
    if extra:
        lines.append(extra)
    return "\n".join(lines)


def _metrics_table_text(
    image_metrics: dict[str, Any],
    fixed_spot: dict[str, Any] | None = None,
) -> str:
    rows = [
        ("Metric", "Bilinear", "Model", "Δ"),
        (
            "PSNR (dB)",
            f"{image_metrics['bilinear_psnr']:.2f}",
            f"{image_metrics['model_psnr']:.2f}",
            f"{image_metrics['model_psnr'] - image_metrics['bilinear_psnr']:+.2f}",
        ),
        (
            "SSIM",
            f"{image_metrics['bilinear_ssim']:.3f}",
            f"{image_metrics['model_ssim']:.3f}",
            f"{image_metrics['model_ssim'] - image_metrics['bilinear_ssim']:+.3f}",
        ),
        (
            "LPIPS ↓",
            f"{image_metrics['bilinear_lpips']:.3f}",
            f"{image_metrics['model_lpips']:.3f}",
            f"{image_metrics['bilinear_lpips'] - image_metrics['model_lpips']:+.3f}",
        ),
    ]
    if fixed_spot:
        rows.append(("", "", "", ""))
        rows.append(
            (
                f"Spot PSNR ({fixed_spot['hr_pixels']}×{fixed_spot['hr_pixels']})",
                f"{fixed_spot['bilinear_psnr']:.2f}",
                f"{fixed_spot['model_psnr']:.2f}",
                f"{fixed_spot['psnr_improvement']:+.2f}",
            )
        )
        rows.append(
            (
                "Spot SSIM",
                f"{fixed_spot['bilinear_ssim']:.3f}",
                f"{fixed_spot['model_ssim']:.3f}",
                f"{fixed_spot['ssim_improvement']:+.3f}",
            )
        )
        rows.append(
            (
                "Spot LPIPS ↓",
                f"{fixed_spot['bilinear_lpips']:.3f}",
                f"{fixed_spot['model_lpips']:.3f}",
                f"{fixed_spot['lpips_improvement']:+.3f}",
            )
        )

    col_w = [18, 10, 10, 8]
    lines = []
    for i, row in enumerate(rows):
        if not any(row):
            lines.append("")
            continue
        if i == 0:
            lines.append("".join(v.ljust(col_w[j]) for j, v in enumerate(row)))
            lines.append("-" * sum(col_w))
        else:
            lines.append("".join(str(v).ljust(col_w[j]) for j, v in enumerate(row)))
    return "\n".join(lines)


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
    """Write SR comparison figures and a metrics summary into ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lr = _clip01(lr_hwc)
    bil = _clip01(bilinear_hwc)
    pred = _clip01(pred_hwc)
    gt = _clip01(gt_hwc)
    lr, bil, pred, gt = _shared_display_stretch(lr, bil, pred, gt)

    spot_slices = fixed_spot.get("slices_hr") if fixed_spot else None
    prefix = f"{sample_label} — " if sample_label else ""
    metrics_text = _metrics_table_text(image_metrics, fixed_spot)

    fig = plt.figure(figsize=(18, 5.5))
    gs = GridSpec(1, 5, figure=fig, width_ratios=[1.0, 1.0, 1.0, 1.0, 0.72], wspace=0.08)

    panels = [
        (gs[0, 0], lr, f"{prefix}LR input", None, None, None),
        (
            gs[0, 1],
            bil,
            "Bilinear upsample",
            image_metrics.get("bilinear_psnr"),
            image_metrics.get("bilinear_ssim"),
            image_metrics.get("bilinear_lpips"),
        ),
        (
            gs[0, 2],
            pred,
            "Model SR",
            image_metrics.get("model_psnr"),
            image_metrics.get("model_ssim"),
            image_metrics.get("model_lpips"),
        ),
        (gs[0, 3], gt, f"{prefix}HR ground truth", None, None, None),
    ]
    for spec, img, label, psnr, ssim, lpips in panels:
        ax = fig.add_subplot(spec)
        ax.imshow(img)
        ax.set_title(_metric_title(label, psnr=psnr, ssim=ssim, lpips=lpips), fontsize=11, fontweight="bold")
        ax.axis("off")
        if spot_slices and label != f"{prefix}LR input":
            _draw_spot_rect(ax, spot_slices)

    ax_tbl = fig.add_subplot(gs[0, 4])
    ax_tbl.axis("off")
    ax_tbl.text(
        0.0,
        1.0,
        "Full-frame metrics\n\n" + metrics_text,
        transform=ax_tbl.transAxes,
        va="top",
        ha="left",
        fontsize=9.5,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#f7f7f7", "edgecolor": "#cccccc"},
    )

    fig.suptitle(f"{prefix}Super-resolution comparison", fontsize=14, fontweight="bold", y=1.02)
    plt.savefig(output_dir / "comparison.png", bbox_inches="tight", pad_inches=0.12, dpi=300)
    plt.close()

    (output_dir / "metrics_table.txt").write_text(metrics_text + "\n", encoding="utf-8")

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
        _save_metrics_card(output_dir, sample_label, image_metrics, fixed_spot=None)
        return

    bil_s = _upscale_spot(_crop_hwc(bil, spot_slices))
    pred_s = _upscale_spot(_crop_hwc(pred, spot_slices))
    gt_s = _upscale_spot(_crop_hwc(gt, spot_slices))
    err_bil = np.abs(bil_s - gt_s).mean(axis=-1)
    err_pred = np.abs(pred_s - gt_s).mean(axis=-1)

    fig = plt.figure(figsize=(16, 9))
    gs = GridSpec(2, 4, figure=fig, height_ratios=[1.0, 0.85])

    spot_panels = [
        (
            gs[0, 0],
            bil_s,
            _metric_title(
                "Bilinear spot",
                psnr=fixed_spot["bilinear_psnr"],
                ssim=fixed_spot["bilinear_ssim"],
                lpips=fixed_spot["bilinear_lpips"],
            ),
        ),
        (
            gs[0, 1],
            pred_s,
            _metric_title(
                "Model spot",
                psnr=fixed_spot["model_psnr"],
                ssim=fixed_spot["model_ssim"],
                lpips=fixed_spot["model_lpips"],
            ),
        ),
        (
            gs[0, 2],
            gt_s,
            f"GT spot\n{fixed_spot['hr_pixels']}×{fixed_spot['hr_pixels']} HR px",
        ),
        (
            gs[0, 3],
            np.clip(err_pred - err_bil, 0, 1),
            "Δ|err| (model better → brighter)",
        ),
    ]
    for spec, img, title in spot_panels:
        ax = fig.add_subplot(spec)
        cmap = "magma" if img.ndim == 2 else None
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")

    ax_b = fig.add_subplot(gs[1, 0:2])
    ax_b.imshow(err_bil, cmap="magma", vmin=0, vmax=max(0.05, err_bil.max()))
    ax_b.set_title(
        _metric_title("Bilinear |error|", extra=f"MAE {fixed_spot['bilinear_mae']:.4f}"),
        fontweight="bold",
    )
    ax_b.axis("off")

    ax_m = fig.add_subplot(gs[1, 2:4])
    ax_m.imshow(err_pred, cmap="magma", vmin=0, vmax=max(0.05, err_pred.max()))
    ax_m.set_title(
        _metric_title("Model |error|", extra=f"MAE {fixed_spot['model_mae']:.4f}"),
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

    (output_dir / "spot_metrics.json").write_text(json.dumps(fixed_spot, indent=2), encoding="utf-8")

    _save_results_summary(
        output_dir,
        lr=lr,
        bil=bil,
        pred=pred,
        gt=gt,
        bil_s=bil_s,
        pred_s=pred_s,
        gt_s=gt_s,
        err_bil=err_bil,
        err_pred=err_pred,
        image_metrics=image_metrics,
        fixed_spot=fixed_spot,
        spot_slices=spot_slices,
        sample_label=sample_label,
        metrics_text=metrics_text,
    )
    _save_metrics_card(output_dir, sample_label, image_metrics, fixed_spot)


def _save_metrics_card(
    output_dir: Path,
    sample_label: str,
    image_metrics: dict[str, Any],
    fixed_spot: dict[str, Any] | None,
) -> None:
    prefix = f"{sample_label} — " if sample_label else ""
    metrics_text = _metrics_table_text(image_metrics, fixed_spot)
    fig, ax = plt.subplots(figsize=(6.5, 4.5 if fixed_spot else 3.2))
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        metrics_text,
        transform=ax.transAxes,
        va="center",
        ha="center",
        fontsize=11,
        family="monospace",
    )
    ax.set_title(f"{prefix}Evaluation metrics", fontsize=13, fontweight="bold", pad=12)
    plt.savefig(output_dir / "metrics_card.png", bbox_inches="tight", pad_inches=0.3, dpi=200)
    plt.close()


def _save_results_summary(
    output_dir: Path,
    *,
    lr: np.ndarray,
    bil: np.ndarray,
    pred: np.ndarray,
    gt: np.ndarray,
    bil_s: np.ndarray,
    pred_s: np.ndarray,
    gt_s: np.ndarray,
    err_bil: np.ndarray,
    err_pred: np.ndarray,
    image_metrics: dict[str, Any],
    fixed_spot: dict[str, Any],
    spot_slices: list[int] | tuple[int, ...],
    sample_label: str,
    metrics_text: str,
) -> None:
    prefix = f"{sample_label} — " if sample_label else ""
    fig = plt.figure(figsize=(20, 11))
    gs = GridSpec(
        3,
        5,
        figure=fig,
        height_ratios=[1.35, 1.0, 0.55],
        width_ratios=[1.0, 1.0, 1.0, 1.0, 0.75],
        hspace=0.28,
        wspace=0.06,
    )

    full_panels = [
        (gs[0, 0], lr, f"{prefix}LR input", None, None, None),
        (
            gs[0, 1],
            bil,
            "Bilinear",
            image_metrics["bilinear_psnr"],
            image_metrics["bilinear_ssim"],
            image_metrics["bilinear_lpips"],
        ),
        (
            gs[0, 2],
            pred,
            "Model SR",
            image_metrics["model_psnr"],
            image_metrics["model_ssim"],
            image_metrics["model_lpips"],
        ),
        (gs[0, 3], gt, f"{prefix}HR GT", None, None, None),
    ]
    for spec, img, label, psnr, ssim, lpips in full_panels:
        ax = fig.add_subplot(spec)
        ax.imshow(img)
        ax.set_title(_metric_title(label, psnr=psnr, ssim=ssim, lpips=lpips), fontsize=10, fontweight="bold")
        ax.axis("off")
        if label != f"{prefix}LR input":
            _draw_spot_rect(ax, spot_slices)

    spot_panels = [
        (gs[1, 0], bil_s, "Spot bilinear", fixed_spot["bilinear_psnr"], fixed_spot["bilinear_ssim"], fixed_spot["bilinear_lpips"]),
        (gs[1, 1], pred_s, "Spot model", fixed_spot["model_psnr"], fixed_spot["model_ssim"], fixed_spot["model_lpips"]),
        (gs[1, 2], gt_s, f"Spot GT ({fixed_spot['hr_pixels']}×{fixed_spot['hr_pixels']})", None, None, None),
        (gs[1, 3], err_pred, "Model |error|", None, None, None),
    ]
    for spec, img, label, psnr, ssim, lpips in spot_panels:
        ax = fig.add_subplot(spec)
        cmap = "magma" if img.ndim == 2 else None
        vmax = max(0.05, err_pred.max()) if img.ndim == 2 else None
        ax.imshow(img, cmap=cmap, vmin=0 if img.ndim == 2 else None, vmax=vmax)
        if psnr is not None:
            ax.set_title(_metric_title(label, psnr=psnr, ssim=ssim, lpips=lpips), fontsize=10, fontweight="bold")
        else:
            ax.set_title(label, fontsize=10, fontweight="bold")
        ax.axis("off")

    ax_err = fig.add_subplot(gs[1, 4])
    ax_err.imshow(err_bil, cmap="magma", vmin=0, vmax=max(0.05, err_bil.max()))
    ax_err.set_title("Bilinear |error|", fontsize=10, fontweight="bold")
    ax_err.axis("off")

    ax_tbl = fig.add_subplot(gs[2, :])
    ax_tbl.axis("off")
    ax_tbl.text(
        0.5,
        0.85,
        metrics_text,
        transform=ax_tbl.transAxes,
        va="top",
        ha="center",
        fontsize=11,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#f7f7f7", "edgecolor": "#cccccc"},
    )

    fig.suptitle(
        f"{prefix}Results overview — full frame + center spot",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    plt.savefig(output_dir / "results_summary.png", bbox_inches="tight", pad_inches=0.12, dpi=200)
    plt.close()


def save_all_cities_metrics_summary(
    city_metrics: list[tuple[str, dict[str, Any]]],
    output_path: Path,
) -> None:
    if not city_metrics:
        return

    headers = ["City", "PSNR", "SSIM", "LPIPS", "Spot PSNR", "Spot SSIM", "Spot LPIPS"]
    rows = []
    for city, m in city_metrics:
        spot = m.get("fixed_spot") or {}
        rows.append(
            [
                city,
                f"{m['psnr']['model']:.2f}",
                f"{m['ssim']['model']:.3f}",
                f"{m['lpips']['model']:.3f}",
                f"{spot.get('model_psnr', float('nan')):.2f}" if spot else "—",
                f"{spot.get('model_ssim', float('nan')):.3f}" if spot else "—",
                f"{spot.get('model_lpips', float('nan')):.3f}" if spot else "—",
            ]
        )

    fig, ax = plt.subplots(figsize=(12, 0.55 * (len(rows) + 2)))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#4472C4")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f2f2f2")
    ax.set_title(
        "All cities — model metrics (full frame + center spot)",
        fontsize=14,
        fontweight="bold",
        pad=16,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.2, dpi=200)
    plt.close()


def load_metrics_from_json(metrics_path: Path) -> dict[str, Any]:
    data = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    image_metrics = {
        "model_psnr": float(data["psnr"]["model"]),
        "bilinear_psnr": float(data["psnr"]["bilinear"]),
        "model_ssim": float(data["ssim"]["model"]),
        "bilinear_ssim": float(data["ssim"]["bilinear"]),
        "model_lpips": float(data["lpips"]["model"]),
        "bilinear_lpips": float(data["lpips"]["bilinear"]),
    }
    return {"image_metrics": image_metrics, "fixed_spot": data.get("fixed_spot"), "raw": data}


def regenerate_visualizations_from_dir(result_dir: Path, *, sample_label: str = "") -> None:
    """Rebuild comparison figures from saved PNGs + metrics.json (no retraining)."""
    result_dir = Path(result_dir)
    lr = plt.imread(result_dir / "lr_original.png")[..., :3]
    bil = plt.imread(result_dir / "bilinear_baseline.png")[..., :3]
    pred = plt.imread(result_dir / "model_output_aligned.png")[..., :3]
    gt = plt.imread(result_dir / "ground_truth.png")[..., :3]
    loaded = load_metrics_from_json(result_dir / "metrics.json")
    label = sample_label or loaded["raw"].get("sample_id", "")
    save_eval_visualizations(
        result_dir,
        lr_hwc=lr.astype(np.float32),
        bilinear_hwc=bil.astype(np.float32),
        pred_hwc=pred.astype(np.float32),
        gt_hwc=gt.astype(np.float32),
        image_metrics=loaded["image_metrics"],
        fixed_spot=loaded["fixed_spot"],
        sample_label=str(label),
    )


def save_hr_lr_revisit_panel(
    ds,
    output_path: Path,
    *,
    city_name: str | None = None,
) -> None:
    """Show harmonized HR GT and every S2 LR revisit for one city."""
    n_lr = ds.num_samples
    hr = ds.get_original_hr().detach().cpu().numpy()
    if hr.ndim == 3 and hr.shape[0] in (1, 3, 4) and hr.shape[0] != hr.shape[-1]:
        hr = hr.transpose(1, 2, 0)
    lr_list = [ds.get_lr_sample(i).detach().cpu().numpy().transpose(1, 2, 0) for i in range(n_lr)]

    hr_disp, *lr_disp = _shared_display_stretch(hr, *lr_list)

    frame_order = sorted(
        range(n_lr),
        key=lambda i: ds.frames[i].get("datetime", ""),
    )
    base_idx = ds.base_frame_index

    n_cols = 4
    n_lr_rows = int(np.ceil(n_lr / n_cols))
    fig_h = 2.8 + 2.2 * n_lr_rows
    fig = plt.figure(figsize=(4 * n_cols, fig_h))
    gs = GridSpec(
        1 + n_lr_rows,
        n_cols,
        figure=fig,
        height_ratios=[2.2] + [1.0] * n_lr_rows,
    )

    title_city = city_name or getattr(ds, "s2_dir", Path("city")).name
    fig.suptitle(
        f"{title_city}: harmonized HR GT and all {n_lr} S2 LR revisits",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    hr_ax = fig.add_subplot(gs[0, :])
    hr_ax.imshow(hr_disp)
    hr_ax.set_title(
        f"HR GT (harmonized)\nNIB {getattr(ds, 'nib_acquisition_date', '?')}",
        fontsize=11,
        fontweight="bold",
    )
    hr_ax.axis("off")
    hr_ax.add_patch(
        patches.Rectangle(
            (0, 0),
            hr.shape[1] - 1,
            hr.shape[0] - 1,
            fill=False,
            edgecolor="#2ecc71",
            linewidth=3,
        )
    )

    for plot_idx, frame_idx in enumerate(frame_order):
        row = 1 + plot_idx // n_cols
        col = plot_idx % n_cols
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(lr_disp[frame_idx])
        frame = ds.frames[frame_idx]
        dt = frame.get("datetime", "")[:10]
        is_base = frame_idx == base_idx
        label = f"LR {frame_idx}: {dt}"
        if is_base:
            label += "\n(base / harmonize ref)"
        ax.set_title(label, fontsize=9, fontweight="bold" if is_base else "normal")
        ax.axis("off")
        if is_base:
            ax.add_patch(
                patches.Rectangle(
                    (0, 0),
                    lr_list[frame_idx].shape[1] - 1,
                    lr_list[frame_idx].shape[0] - 1,
                    fill=False,
                    edgecolor="#e74c3c",
                    linewidth=3,
                )
            )

    plt.tight_layout(pad=1.2)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.15, dpi=200)
    plt.close()
