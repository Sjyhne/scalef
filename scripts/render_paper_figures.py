#!/usr/bin/env python3
"""Paper RGB figures rendered from float reflectance under the fixed display mapping.

``qualitative``  renders LR / bilinear / ScaleF / NIB rows with central zooms from the reported
                 seed-6 runs (confirmatory_v7_clear: clear-pixel statistics and histogram match,
                 offset removed, float GeoTIFF export).
``nested``       renders the matched LR64 window of the Asker parent whose LR64 LPIPS gain over
                 bilinear (nested_v4_scores_clear_v4.json, window-matched, clear_v4) is the
                 median of that parent (not the best). Every panel is shifted by the same
                 exact -offset before display.

All panels use a linear 0–0.25 reflectance stretch, the same range used to view the production GeoTIFFs.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import rasterio  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from eval.display import linear_stretch  # noqa: E402

FIG_DIR = ROOT / "ScaleF_Overleaf" / "generated" / "figures"
FIG_NAMESPACE = "confirmatory_nocut"
NESTED_V4_SCORES = ROOT / "paper" / "results" / "nested_v4_scores_clear_v4.json"
QUALITATIVE_SITES = ("asker", "rafsbotn", "flekkefjord", "melhus", "bergen")
DIFFICULT_SITE = "bergen"
ZOOM_FRAC = 0.25

plt.rcParams.update({"font.family": "serif", "font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42})


def read_rgb(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return np.transpose(src.read((1, 2, 3)).astype(np.float32), (1, 2, 0))


def centre_crop(img: np.ndarray, frac: float) -> np.ndarray:
    h, w = img.shape[:2]
    ch, cw = int(round(h * frac)), int(round(w * frac))
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    return img[y0:y0 + ch, x0:x0 + cw]


def qualitative_run_dir(city: str) -> Path:
    return ROOT / "single_samples" / city / FIG_NAMESPACE / f"{FIG_NAMESPACE}__b0_17__lr512_b0__{city}__seed6"


def remove_offset(img: np.ndarray, offset: float) -> np.ndarray:
    return np.where(img > 0, np.clip(img - offset, 0.0, None), img) if offset else img


def render_qualitative(sites: tuple[str, ...], out: Path) -> None:
    cols = ("LR (base frame)", "Bilinear", "SuperF ext.", "NIB", "SuperF ext. (zoom)", "NIB (zoom)")
    fig, axes = plt.subplots(len(sites), len(cols), figsize=(7.2, 1.3 * len(sites)))
    meta = []
    for r, city in enumerate(sites):
        run = qualitative_run_dir(city)
        q = run / "qgis"
        lr, bil, sr, gt = (read_rgb(q / f) for f in ("s2_lr.tif", "s2_bilinear.tif", "sr_pred.tif", "hr_gt.tif"))
        m = json.loads((run / "metrics.json").read_text())
        panels = (lr, bil, sr, gt, centre_crop(sr, ZOOM_FRAC), centre_crop(gt, ZOOM_FRAC))
        for c, img in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(linear_stretch(img), interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(cols[c], fontsize=8, pad=3)
        lp, lpb = m["lpips"]["model"], m["lpips"]["bilinear"]
        label = city.replace("tromso", "troms\u00f8").title()
        if city == DIFFICULT_SITE:
            label = f"{label}*"
        axes[r, 0].set_ylabel(label, fontsize=8)
        meta.append({"site": city, "difficult_case": city == DIFFICULT_SITE,
                     "run_dir": str(run.relative_to(ROOT)), "lpips": lp, "bilinear_lpips": lpb,
                     "psnr": m["psnr"]["model"], "bilinear_psnr": m["psnr"]["bilinear"]})
    fig.tight_layout(w_pad=0.1, h_pad=0.2)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    out.with_suffix(".json").write_text(json.dumps({
        "display": "linear stretch 0-0.25",
        "zoom_fraction": ZOOM_FRAC,
        "difficult_case": DIFFICULT_SITE,
        "difficult_case_reason": "Bergen is marked when included. Among the float-export sites, Flekkefjord is the SSIM deficit and Melhus has the smallest LPIPS gain.",
        "rows": meta,
    }, indent=1) + "\n")
    print(f"wrote {out.with_suffix('.pdf')}")


def render_failures(out: Path) -> None:
    """Named-site PSNR exceptions and the two complete-field LPIPS exceptions."""
    rows = (
        {"label": "Alta", "kind": "named", "city": "alta"},
        {"label": "Trondheim", "kind": "named", "city": "trondheim"},
        {"label": "\u00c5lg\u00e5rd p05", "kind": "complete", "city": "algard", "tile": "algard_p05_02", "offset": 0.1},
        {"label": "\u00c5lg\u00e5rd p04", "kind": "complete", "city": "algard", "tile": "algard_p04_03", "offset": 0.1},
    )
    cols = ("LR (base frame)", "Bilinear", "SuperF ext.", "NIB", "SuperF ext. (zoom)", "NIB (zoom)")
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(7.2, 1.3 * len(rows)))
    meta = []
    for r, spec in enumerate(rows):
        if spec["kind"] == "named":
            run = qualitative_run_dir(spec["city"])
            offset = 0.0
        else:
            run = ROOT / "single_samples" / spec["city"] / "sample" / f"prod_k4_clear_v4_{spec['tile']}"
            offset = float(spec["offset"])
        q = run / "qgis"
        panels_full = tuple(
            remove_offset(read_rgb(q / name), offset)
            for name in ("s2_lr.tif", "s2_bilinear.tif", "sr_pred.tif", "hr_gt.tif")
        )
        panels = panels_full + (centre_crop(panels_full[2], ZOOM_FRAC), centre_crop(panels_full[3], ZOOM_FRAC))
        m = json.loads((run / "metrics.json").read_text())
        for c, img in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(linear_stretch(img), interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if r == 0:
                ax.set_title(cols[c], fontsize=8, pad=3)
        axes[r, 0].set_ylabel(spec["label"], fontsize=8)
        row = {
            "label": spec["label"],
            "kind": spec["kind"],
            "run_dir": str(run.relative_to(ROOT)),
            "offset_removed": offset,
            "fit_lpips_model": m["lpips"]["model"],
            "fit_lpips_bilinear": m["lpips"]["bilinear"],
            "psnr_model": m["psnr"]["model"],
            "psnr_bilinear": m["psnr"]["bilinear"],
            "ssim_model": m["ssim"]["model"],
            "ssim_bilinear": m["ssim"]["bilinear"],
            "stopped": m["early_stop"]["stopped"],
            "stopped_iter": m["early_stop"]["stopped_iter"],
            "best_val_iter": m["early_stop"]["best_val_iter"],
            "training_time_seconds": m["training_time_seconds"],
            "git": m["provenance"]["git"],
        }
        if spec["kind"] == "complete":
            scored = json.loads((ROOT / "paper/results/complete_fields_clear_v4.json").read_text())
            rec = next(item for item in scored["rows"] if item["tile_id"] == spec["tile"])
            row["reported_lpips_model"] = rec["lpips_model"]
            row["reported_lpips_bilinear"] = rec["lpips_bilinear"]
            row["reported_score_file"] = "paper/results/complete_fields_clear_v4.json"
        meta.append(row)
    fig.tight_layout(w_pad=0.1, h_pad=0.2)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    out.with_suffix(".json").write_text(json.dumps({
        "display": "linear stretch 0-0.25",
        "zoom_fraction": ZOOM_FRAC,
        "named_site_namespace": FIG_NAMESPACE,
        "complete_field_prefix": "prod_k4_clear_v4",
        "rows": meta,
    }, indent=1) + "\n")
    print(f"wrote {out.with_suffix('.pdf')}")


def render_nested(out: Path, parent: str = "asker_p02_02", scores_path: Path | None = None,
                  run_tag: str | None = None) -> None:
    from eval.common_footprint import crop_hwc_frac, nest_window_frac
    from scripts.rescore_nested_common_footprint import SIDES, _covering_tile, _index_manifests, _run_dir
    from scripts.run_nested_float_validation import RUN_TAG
    scores_path = scores_path or NESTED_V4_SCORES
    run_tag = run_tag or RUN_TAG

    def crop_for(side: int, child: dict, image: np.ndarray) -> np.ndarray:
        if side == 64:
            return image
        k = side // 64
        return crop_hwc_frac(image, *nest_window_frac(int(child["nest_iy"]) % k, int(child["nest_ix"]) % k, 64, side))

    scores = json.loads(scores_path.read_text())["rows"]
    by_child: dict[str, dict] = {}
    for r in scores:
        if r["parent"] == parent:
            by_child.setdefault(r["child"], {})[r["side"]] = r
    gain = {c: v["bilinear"]["lpips_window"] - v[64]["lpips_window"] for c, v in by_child.items()}
    ordered = sorted(gain, key=gain.get)
    chosen_id = ordered[len(ordered) // 2]
    median_gain = statistics.median(gain.values())
    offset = float(by_child[chosen_id][64]["boa_offset"])

    from scripts.rescore_nested_common_footprint import NEST_ROOT

    index = _index_manifests()
    children = json.loads((NEST_ROOT / "nested_lr64_manifest.json").read_text())["tiles"]
    child = next(c for c in children if c["tile_id"] == chosen_id)
    covering = {side: _covering_tile(child, side, index) for side in SIDES}
    dirs = {s: _run_dir(t["parent_city"], t["tile_id"], s, run_tag) for s, t in covering.items()}
    gt = remove_offset(read_rgb(dirs[64] / "qgis" / "hr_gt.tif"), offset)
    bil = remove_offset(read_rgb(dirs[64] / "qgis" / "s2_bilinear.tif"), offset)
    preds = {s: remove_offset(crop_for(s, child, read_rgb(dirs[s] / "qgis" / "sr_pred.tif")), offset)
             for s in (64, 128, 256, 512)}

    labels = ["NIB", "Bilinear", *(f"Ext. LR{s}" for s in (64, 128, 256, 512))]
    panels = [gt, bil, *(preds[s] for s in (64, 128, 256, 512))]
    fig, axes = plt.subplots(1, len(panels), figsize=(7.2, 1.55))
    for ax, label, img in zip(axes, labels, panels):
        ax.imshow(linear_stretch(img), interpolation="nearest")
        ax.set_title(label, fontsize=8, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    fig.tight_layout(w_pad=0.15)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    meta = {
        "selection": (f"median LR64 LPIPS gain over bilinear among the {len(gain)} windows of {parent} "
                      f"(window-matched, {scores_path.name}, run_tag={run_tag})"),
        "child_tile_id": chosen_id, "median_gain": median_gain, "boa_offset_removed": offset,
        "scores": {str(s): {k: v for k, v in r.items() if k.endswith(("_window", "_parent"))}
                   for s, r in by_child[chosen_id].items()},
        "run_dirs": {str(s): str(d.relative_to(ROOT)) for s, d in dirs.items()},
        "display": "linear stretch 0-0.25",
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"wrote {out.with_suffix('.pdf')}  child={chosen_id}  gain={median_gain:.3f}  offset={offset}")


def render_qualitative_main(sites: tuple[str, ...], out: Path) -> None:
    """Main-text multi-site figure: full 5.12 km context + zoomed 1.28 km Bilinear / Ext. / NIB."""
    import matplotlib.patches as patches

    cols = (
        "Full 5.12 km field (NIB)",
        "Bilinear (2.5 m, zoom)",
        "SuperF ext. (2.5 m, zoom)",
        "NIB reference (2.5 m, zoom)",
    )
    fig, axes = plt.subplots(len(sites), len(cols), figsize=(7.2, 1.72 * len(sites)))
    meta = []
    for r, city in enumerate(sites):
        run = qualitative_run_dir(city)
        q = run / "qgis"
        bil, sr, gt = (read_rgb(q / f) for f in ("s2_bilinear.tif", "sr_pred.tif", "hr_gt.tif"))
        m = json.loads((run / "metrics.json").read_text())
        h, w = gt.shape[:2]
        ch, cw = int(round(h * ZOOM_FRAC)), int(round(w * ZOOM_FRAC))
        y0, x0 = (h - ch) // 2, (w - cw) // 2

        panels = (
            gt,
            bil[y0:y0 + ch, x0:x0 + cw],
            sr[y0:y0 + ch, x0:x0 + cw],
            gt[y0:y0 + ch, x0:x0 + cw],
        )
        for c, img in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(linear_stretch(img), interpolation="nearest")
            if c == 0:
                rect = patches.Rectangle(
                    (x0, y0), cw, ch, linewidth=1.1, edgecolor="#FFD000", facecolor="none"
                )
                ax.add_patch(rect)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(cols[c], fontsize=8.0, pad=3)
        lp, lpb = m["lpips"]["model"], m["lpips"]["bilinear"]
        psnr, psnrb = m["psnr"]["model"], m["psnr"]["bilinear"]
        label = city.replace("tromso", "troms\u00f8").title()
        axes[r, 0].set_ylabel(
            f"{label}\nLPIPS {lp:.3f} (bil. {lpb:.3f})\nPSNR {psnr:.1f} dB",
            fontsize=7.3,
        )
        meta.append({
            "site": city,
            "run_dir": str(run.relative_to(ROOT)),
            "zoom_box_hr_px": [y0, x0, ch, cw],
            "lpips_model": lp,
            "lpips_bilinear": lpb,
            "psnr_model": psnr,
            "psnr_bilinear": psnrb,
        })
    fig.tight_layout(w_pad=0.12, h_pad=0.2)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    out.with_suffix(".json").write_text(json.dumps({
        "display": "linear stretch 0-0.25 (processing-baseline offset removed)",
        "zoom_fraction": ZOOM_FRAC,
        "zoom_extent_m": 1280.0,
        "rows": meta,
    }, indent=1) + "\n")
    print(f"wrote {out.with_suffix('.pdf')}")


def render_original_superf(out: Path) -> None:
    """Render all 6 shared LR64 windows comparing Original SuperF (s=2, 5, 10) and our extension."""
    from s2_dataset import boa_add_offset

    sf_root = ROOT / "paper" / "results" / "original_superf_norway_lr64"
    scales_json = json.loads((sf_root / "fourier_scale_search.json").read_text())
    sites_by_city = {row["city"]: row for row in scales_json["sites"]}
    cities = ("asker", "bergen", "flekkefjord", "tromso", "trondheim", "vennesla")
    cols = (
        "Bilinear (2.5 m)",
        "Orig. SuperF (s=2)",
        "Orig. SuperF (s=5)",
        "Orig. SuperF (s=10)",
        "SuperF ext. (grid)",
        "NIB (2.5 m)",
    )
    fig, axes = plt.subplots(len(cities), len(cols), figsize=(7.2, 1.24 * len(cities)))
    meta = []
    for r, city in enumerate(cities):
        summary = json.loads((sf_root / city / "summary.json").read_text())
        ext_dir = Path(summary["extension_run"])
        ext_metrics = json.loads((ext_dir / "metrics.json").read_text())
        s2_dir = Path(ext_metrics["provenance"]["args"]["s2_dir"])
        s2_meta = json.loads((s2_dir / "meta.json").read_text())
        offsets = {boa_add_offset(f["stac_id"]) for f in s2_meta["frames"]}
        offset = float(next(iter(offsets))) if len(offsets) == 1 else 0.1

        q = ext_dir / "qgis"
        bil = remove_offset(read_rgb(q / "s2_bilinear.tif"), offset)
        ext = remove_offset(read_rgb(q / "sr_pred.tif"), offset)
        gt = remove_offset(read_rgb(q / "hr_gt.tif"), offset)
        sf_s2 = remove_offset(np.load(sf_root / "fourier_s2" / city / "superf" / "pred_hwc.npy"), offset)
        sf_s5 = remove_offset(np.load(sf_root / "fourier_s5" / city / "superf" / "pred_hwc.npy"), offset)
        sf_s10 = remove_offset(np.load(sf_root / city / "superf" / "pred_hwc.npy"), offset)

        panels = (bil, sf_s2, sf_s5, sf_s10, ext, gt)
        srec = sites_by_city[city]
        lp_s2 = srec["scales"]["2"]["lpips"]
        lp_s5 = srec["scales"]["5"]["lpips"]
        lp_s10 = srec["scales"]["10"]["lpips"]
        lp_ext = srec["extension"]["lpips"]
        lp_bil = srec["bilinear"]["lpips"]
        lp_vals = (lp_bil, lp_s2, lp_s5, lp_s10, lp_ext, None)

        for c, (img, lp_val) in enumerate(zip(panels, lp_vals)):
            ax = axes[r, c]
            ax.imshow(linear_stretch(img), interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(cols[c], fontsize=7.5, pad=3)
            if lp_val is not None:
                ax.text(
                    0.03,
                    0.04,
                    f"{lp_val:.3f}",
                    transform=ax.transAxes,
                    fontsize=6.5,
                    color="white",
                    bbox={"boxstyle": "round,pad=0.12", "facecolor": "black", "alpha": 0.65, "edgecolor": "none"},
                )
        label = city.replace("tromso", "troms\u00f8").title()
        axes[r, 0].set_ylabel(label, fontsize=7.8)
        meta.append({
            "city": city,
            "window": srec["window"],
            "boa_offset_removed": offset,
            "lpips_bilinear": lp_bil,
            "lpips_superf_s2": lp_s2,
            "lpips_superf_s5": lp_s5,
            "lpips_superf_s10": lp_s10,
            "lpips_extension": lp_ext,
        })
    fig.tight_layout(w_pad=0.1, h_pad=0.15)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    out.with_suffix(".json").write_text(json.dumps({
        "display": "linear stretch 0-0.25 (processing-baseline offset removed per site)",
        "window_extent_m": 640.0,
        "overlay_metric": "unmasked full-window LPIPS against float qgis/hr_gt.tif",
        "rows": meta,
    }, indent=1) + "\n")
    print(f"wrote {out.with_suffix('.pdf')}")


def render_denmark_rceil(out: Path) -> None:
    """Render Denmark 32UNG solar-panel glint artifact before and after --lr_reflectance_ceiling 1.0."""
    import matplotlib.patches as patches
    from rasterio.enums import Resampling
    from rasterio.windows import Window

    mosaic_before = ROOT / "production/denmark_national_2025/mosaics/32UNG_ovl5_nocut_sr_2p5m.tif"
    mosaic_after = ROOT / "production/denmark_national_2025/mosaics/32UNG_ovl5_nocut_rceil_sr_2p5m.tif"
    cell_dir = ROOT / "data/s2_revisits/denmark_national_2025/32UNG_t512_ovl5_y13_x19"
    run_before = ROOT / "single_samples/32UNG/sample/prod_k4_ocm_ovl5_nocut_32UNG_t512_ovl5_y13_x19"
    run_after = ROOT / "single_samples/32UNG/sample/prod_k4_ocm_ovl5_nocut_rceil_32UNG_t512_ovl5_y13_x19"

    # Cell y13_x19 starts at LR (row_off=6318, col_off=9234), HR (25272, 36936), size 2048x2048.
    # Show a 3x3-cell neighborhood (approx 6000x6000 HR px = 15x15 km) centered on y13_x19.
    nh_side = 5832  # 3 * 1944 (stride 486 LR * 4)
    nh_y0 = 25272 - 1944
    nh_x0 = 36936 - 1944
    win_nh = Window(nh_x0, nh_y0, nh_side, nh_side)

    with rasterio.open(mosaic_before) as src:
        nh_before = np.transpose(
            src.read((1, 2, 3), window=win_nh, out_shape=(3, 972, 972), resampling=Resampling.average),
            (1, 2, 0),
        ).astype(np.float32)
    with rasterio.open(mosaic_after) as src:
        nh_after = np.transpose(
            src.read((1, 2, 3), window=win_nh, out_shape=(3, 972, 972), resampling=Resampling.average),
            (1, 2, 0),
        ).astype(np.float32)

    # Solar plant inside cell y13_x19: around (y=200..360, x=230..390) in LR pixels (160x160 LR = 1.6x1.6 km)
    lr_y0, lr_x0, lr_s = 192, 224, 160
    hr_y0, hr_x0, hr_s = lr_y0 * 4, lr_x0 * 4, lr_s * 4

    # Read raw base frame DN/10000 for the ceiling mask on 014_20250630.tif
    with rasterio.open(cell_dir / "014_20250630.tif") as src:
        raw_dn = np.transpose(
            src.read((1, 2, 3), window=Window(9234 + lr_x0, 6318 + lr_y0, lr_s, lr_s)),
            (1, 2, 0),
        ).astype(np.float32) / 10000.0
    ceiling_mask = np.any(raw_dn > 1.0, axis=-1)

    lr_base = read_rgb(run_after / "qgis" / "s2_lr.tif")[lr_y0:lr_y0 + lr_s, lr_x0:lr_x0 + lr_s]
    sr_before = read_rgb(run_before / "qgis" / "sr_pred.tif")[hr_y0:hr_y0 + hr_s, hr_x0:hr_x0 + hr_s]
    sr_after = read_rgb(run_after / "qgis" / "sr_pred.tif")[hr_y0:hr_y0 + hr_s, hr_x0:hr_x0 + hr_s]

    lr_disp = linear_stretch(lr_base)
    mask_overlay = lr_disp.copy()
    mask_overlay[ceiling_mask] = np.array([1.0, 0.1, 0.85], dtype=np.float32)

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.7))
    top_panels = (
        (nh_before, "(a) 32UNG 3×3-cell region (no ceiling)"),
        (nh_after, "(b) 32UNG 3×3-cell region (ceiling ≤ 1.0)"),
        (lr_disp, "(c) 10 m base frame (1.6 km zoom)"),
    )
    bot_panels = (
        (mask_overlay, "(d) Ceiling mask (DN/10000 > 1.0, 2.05%)"),
        (linear_stretch(sr_before), "(e) 2.5 m SR without ceiling (zoom)"),
        (linear_stretch(sr_after), "(f) 2.5 m SR with ceiling ≤ 1.0 (zoom)"),
    )
    scale_nh = 972.0 / nh_side
    cell_box_xy = (1944 * scale_nh, 1944 * scale_nh)
    cell_box_s = 2048 * scale_nh
    zoom_box_xy = ((1944 + hr_x0) * scale_nh, (1944 + hr_y0) * scale_nh)
    zoom_box_s = hr_s * scale_nh

    for c, (img, title) in enumerate(top_panels):
        ax = axes[0, c]
        ax.imshow(linear_stretch(img) if c < 2 else img, interpolation="nearest")
        if c < 2:
            ax.add_patch(patches.Rectangle(
                cell_box_xy, cell_box_s, cell_box_s,
                linewidth=1.0, linestyle="--", edgecolor="white", facecolor="none"
            ))
            ax.add_patch(patches.Rectangle(
                zoom_box_xy, zoom_box_s, zoom_box_s,
                linewidth=1.2, edgecolor="#FFD000", facecolor="none"
            ))
        ax.set_title(title, fontsize=7.3, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    for c, (img, title) in enumerate(bot_panels):
        ax = axes[1, c]
        ax.imshow(img, interpolation="nearest")
        ax.set_title(title, fontsize=7.3, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    fig.tight_layout(w_pad=0.15, h_pad=0.25)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    out.with_suffix(".json").write_text(json.dumps({
        "display": "linear stretch 0-0.25 (processing-baseline offset removed)",
        "mgrs_tile": "32UNG",
        "cell_id": "32UNG_t512_ovl5_y13_x19",
        "base_frame": "014_20250630.tif",
        "neighborhood_window_hr_px": [nh_x0, nh_y0, nh_side, nh_side],
        "zoom_window_in_cell_lr_px": [lr_x0, lr_y0, lr_s, lr_s],
        "ceiling_dn_over_10000": 1.0,
        "ceiling_masked_fraction_in_zoom": float(ceiling_mask.mean()),
    }, indent=1) + "\n")
    print(f"wrote {out.with_suffix('.pdf')}")


def render_granule_multiscale(out: Path) -> None:
    """Multi-scale zoom from full 109.8 km MGRS 32UNG granule -> 5.12 km LR512 cell -> 640 m 10 m vs 2.5 m detail."""
    import matplotlib.patches as patches
    from matplotlib.gridspec import GridSpec
    from rasterio.enums import Resampling

    mosaic_path = ROOT / "production/denmark_national_2025/mosaics/32UNG_ovl5_nocut_rceil_sr_2p5m.tif"
    ov_side = 1200
    with rasterio.open(mosaic_path) as src:
        full_h, full_w = src.height, src.width
        overview = np.transpose(
            src.read((1, 2, 3), out_shape=(3, ov_side, ov_side), resampling=Resampling.average),
            (1, 2, 0),
        ).astype(np.float32)

    # Region A: Odense city/harbor (y11_x17 -> row_off = 11*486 = 5346, col_off = 17*486 = 8262)
    # Region B: Svendborg / Tåsinge bridge & coast (y16_x19 -> row_off = 16*486 = 7776, col_off = 19*486 = 9234)
    specs = (
        {"tag": "A", "label": "Region A (Odense, 5.12 km)", "cell": "32UNG_t512_ovl5_y11_x17",
         "row_lr": 11 * 486, "col_lr": 17 * 486, "zy_lr": 192, "zx_lr": 192, "color": "#FFD000"},
        {"tag": "B", "label": "Region B (Funen coast, 5.12 km)", "cell": "32UNG_t512_ovl5_y13_x18",
         "row_lr": 13 * 486, "col_lr": 18 * 486, "zy_lr": 160, "zx_lr": 220, "color": "#00E5FF"},
    )

    fig = plt.figure(figsize=(7.2, 4.35))
    gs = GridSpec(2, 4, figure=fig, width_ratios=[1.85, 1.0, 1.0, 1.0], wspace=0.08, hspace=0.18)

    ax_ov = fig.add_subplot(gs[:, 0])
    ax_ov.imshow(linear_stretch(overview), interpolation="nearest")
    ax_ov.set_title("MGRS 32UNG mosaic\n(109.8 × 109.8 km, 443 cells)", fontsize=7.5, pad=3)
    ax_ov.set_xticks([])
    ax_ov.set_yticks([])
    for s in ax_ov.spines.values():
        s.set_visible(False)

    scale_ov = ov_side / float(full_w)
    meta_rows = []
    for r, sp in enumerate(specs):
        cell_run = ROOT / "single_samples/32UNG/sample" / f"prod_k4_ocm_ovl5_nocut_rceil_{sp['cell']}"
        q = cell_run / "qgis"
        bil = read_rgb(q / "s2_bilinear.tif")
        sr = read_rgb(q / "sr_pred.tif")

        # Draw 5.12 km box on full granule overview
        bx = sp["col_lr"] * 4 * scale_ov
        by = sp["row_lr"] * 4 * scale_ov
        bs = 2048 * scale_ov
        ax_ov.add_patch(patches.Rectangle(
            (bx, by), bs, bs, linewidth=1.3, edgecolor=sp["color"], facecolor="none"
        ))
        ax_ov.text(
            bx - 6, by - 6, sp["tag"], color=sp["color"], fontsize=7.5, fontweight="bold",
            bbox={"boxstyle": "round,pad=0.1", "facecolor": "black", "alpha": 0.65, "edgecolor": "none"},
        )

        zy, zx = sp["zy_lr"], sp["zx_lr"]
        zs_lr = 64  # 640 m window
        zy_hr, zx_hr, zs_hr = zy * 4, zx * 4, zs_lr * 4

        row_panels = (
            (sr, sp["label"]),
            (bil[zy_hr:zy_hr + zs_hr, zx_hr:zx_hr + zs_hr], f"{sp['tag']}: Bilinear (2.5 m, 640 m)"),
            (sr[zy_hr:zy_hr + zs_hr, zx_hr:zx_hr + zs_hr], f"{sp['tag']}: SuperF ext. (2.5 m, 640 m)"),
        )
        for c, (img, title) in enumerate(row_panels):
            ax = fig.add_subplot(gs[r, c + 1])
            ax.imshow(linear_stretch(img), interpolation="nearest")
            if c == 0:
                ax.add_patch(patches.Rectangle(
                    (zx_hr, zy_hr), zs_hr, zs_hr, linewidth=1.1, edgecolor=sp["color"], facecolor="none"
                ))
            ax.set_title(title, fontsize=7.3, pad=2.5)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)

        meta_rows.append({
            "tag": sp["tag"],
            "cell": sp["cell"],
            "run_dir": str(cell_run.relative_to(ROOT)),
            "zoom_lr_px": [zy, zx, zs_lr, zs_lr],
        })

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    out.with_suffix(".json").write_text(json.dumps({
        "display": "linear stretch 0-0.25 (processing-baseline offset removed)",
        "mosaic": str(mosaic_path.relative_to(ROOT)),
        "granule_extent_km": 109.8,
        "cell_extent_km": 5.12,
        "detail_extent_m": 640.0,
        "regions": meta_rows,
    }, indent=1) + "\n")
    print(f"wrote {out.with_suffix('.pdf')}")


def render_encoding_scale(out: Path) -> None:
    """Visual comparison of Fourier (s=10) vs. multiresolution grid across LR64, LR256, and LR512 on the central 640 m crop."""
    from PIL import Image

    cities = ("asker", "bergen", "trondheim")
    cols = (
        "Bilinear",
        "Fourier s=10 (LR64)",
        "Grid (LR64)",
        "Fourier s=10 (LR256)",
        "Grid (LR256)",
        "Fourier s=10 (LR512)",
        "Grid (LR512)",
        "NIB (2.5 m)",
    )
    fig, axes = plt.subplots(len(cities), len(cols), figsize=(7.2, 1.05 * len(cities)))
    meta = []

    def load_png(p: Path) -> np.ndarray:
        return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0

    for r, city in enumerate(cities):
        base = ROOT / "single_samples" / city / "confirmatory_nocut"
        r64_g = base / f"confirmatory_nocut__encoding_size__lr64_hash__{city}__seed6"
        r64_f = base / f"confirmatory_nocut__encoding_size__lr64_fourier_s10__{city}__seed6"
        r256_g = base / f"confirmatory_nocut__encoding_size__lr256_hash__{city}__seed6"
        r256_f = base / f"confirmatory_nocut__encoding_size__lr256_fourier_s10__{city}__seed6"
        r512_g = base / f"confirmatory_nocut__encoding_size__lr512_hash__{city}__seed6"
        r512_f = base / f"confirmatory_nocut__encoding_size__lr512_fourier_s10__{city}__seed6"

        bil = load_png(r64_g / "bilinear_baseline.png")
        gt = load_png(r64_g / "ground_truth.png")
        p64_f = load_png(r64_f / "model_output_aligned.png")
        p64_g = load_png(r64_g / "model_output_aligned.png")
        p256_f = centre_crop(load_png(r256_f / "model_output_aligned.png"), 64.0 / 256.0)
        p256_g = centre_crop(load_png(r256_g / "model_output_aligned.png"), 64.0 / 256.0)
        p512_f = centre_crop(load_png(r512_f / "model_output_aligned.png"), 64.0 / 512.0)
        p512_g = centre_crop(load_png(r512_g / "model_output_aligned.png"), 64.0 / 512.0)

        runs = (r64_g, r64_f, r64_g, r256_f, r256_g, r512_f, r512_g, None)
        panels = (bil, p64_f, p64_g, p256_f, p256_g, p512_f, p512_g, gt)
        row_scores = {}
        for c, (img, rd) in enumerate(zip(panels, runs)):
            ax = axes[r, c]
            ax.imshow(img, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(cols[c], fontsize=6.4, pad=3)
            if rd is not None:
                m = json.loads((rd / "metrics.json").read_text())
                lp = m["lpips"]["bilinear"] if c == 0 else m["lpips"]["model"]
                row_scores[cols[c]] = lp
                ax.text(
                    0.03,
                    0.04,
                    f"{lp:.3f}",
                    transform=ax.transAxes,
                    fontsize=6.0,
                    color="white",
                    bbox={"boxstyle": "round,pad=0.12", "facecolor": "black", "alpha": 0.65, "edgecolor": "none"},
                )
        axes[r, 0].set_ylabel(city.title(), fontsize=7.5)
        meta.append({"city": city, "field_lpips": row_scores})

    fig.tight_layout(w_pad=0.08, h_pad=0.15)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    out.with_suffix(".json").write_text(json.dumps({
        "display": "shared 99.5th-percentile stretch from confirmatory_nocut encoding_size seed 6",
        "central_crop_extent_m": 640.0,
        "overlay_metric": "full-field clear-pixel LPIPS from metrics.json",
        "rows": meta,
    }, indent=1) + "\n")
    print(f"wrote {out.with_suffix('.pdf')}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("qualitative")
    q.add_argument("--sites", nargs="+", default=list(QUALITATIVE_SITES))
    q.add_argument("--out", type=Path, default=FIG_DIR / "qualitative_tiles_hc")
    qm = sub.add_parser("qualitative-main")
    qm.add_argument("--sites", nargs="+", default=["asker", "rafsbotn", "flekkefjord", "melhus"])
    qm.add_argument("--out", type=Path, default=FIG_DIR / "qualitative_main_hc")
    sf = sub.add_parser("original-superf")
    sf.add_argument("--out", type=Path, default=FIG_DIR / "original_superf_comparison_hc")
    es = sub.add_parser("encoding-scale")
    es.add_argument("--out", type=Path, default=FIG_DIR / "fourier_vs_grid_extent_hc")
    dr = sub.add_parser("denmark-rceil")
    dr.add_argument("--out", type=Path, default=FIG_DIR / "denmark_32ung_rceil_hc")
    gm = sub.add_parser("granule-multiscale")
    gm.add_argument("--out", type=Path, default=FIG_DIR / "granule_multiscale_zoom_hc")
    f = sub.add_parser("failures")
    f.add_argument("--out", type=Path, default=FIG_DIR / "qualitative_failures")
    n = sub.add_parser("nested")
    n.add_argument("--out", type=Path, default=FIG_DIR / "nested_spot_hc")
    n.add_argument("--scores", type=Path, default=NESTED_V4_SCORES)
    n.add_argument("--run-tag", default="clear_v4")
    args = ap.parse_args()
    if args.cmd == "qualitative":
        render_qualitative(tuple(args.sites), args.out)
    elif args.cmd == "qualitative-main":
        render_qualitative_main(tuple(args.sites), args.out)
    elif args.cmd == "original-superf":
        render_original_superf(args.out)
    elif args.cmd == "encoding-scale":
        render_encoding_scale(args.out)
    elif args.cmd == "denmark-rceil":
        render_denmark_rceil(args.out)
    elif args.cmd == "granule-multiscale":
        render_granule_multiscale(args.out)
    elif args.cmd == "failures":
        render_failures(args.out)
    else:
        render_nested(args.out, scores_path=args.scores, run_tag=args.run_tag)


if __name__ == "__main__":
    main()
