#!/usr/bin/env python3
"""Paper RGB figures rendered from float reflectance under the fixed display mapping.

``qualitative``  refits the listed named sites with the frozen 17-site command
                 (confirmatory_v3_halo, seed 6) plus float GeoTIFF export, under a separate
                 run name, and renders LR / bilinear / ScaleF / NIB rows with central zooms.
``nested``       renders the matched LR64 window of the float-validated Asker parent whose
                 LR64 LPIPS gain over bilinear is the median of that parent (not the best).

All panels use eval.display.highlight_compress (Sentinel Hub HighlightCompress, 0-0.4).
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import statistics
import subprocess
import sys
import threading
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import rasterio  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from eval.display import highlight_compress  # noqa: E402

FIG_DIR = ROOT / "ScaleF_Overleaf" / "generated" / "figures"
HALO_MANIFEST = ROOT / "single_samples" / "sweep_results" / "confirmatory_v3_halo" / "run_manifest.json"
FIG_TAG = "figure_float_v3"
QUALITATIVE_SITES = ("asker", "rafsbotn", "flekkefjord", "melhus")
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
    return ROOT / "single_samples" / city / FIG_TAG / f"{FIG_TAG}__{city}__seed6"


def qualitative_command(city: str, device: int) -> list[str]:
    jobs = json.loads(HALO_MANIFEST.read_text())["jobs"]
    job = next(j for j in jobs if j["city"] == city and int(j["seed"]) == 6)
    cmd = shlex.split(job["command_shell"])
    out = []
    skip = False
    for tok in cmd:
        if skip:
            skip = False
            continue
        if tok == "--no_qgis_export":
            continue
        if tok in ("--sample_id", "--run_name", "--device"):
            skip = True
            continue
        out.append(tok)
    return [*out, "--sample_id", FIG_TAG, "--run_name", f"{FIG_TAG}__{city}__seed6", "--device", str(device)]


def fit_qualitative(sites: tuple[str, ...], gpus: list[int]) -> None:
    todo = [c for c in sites if not (qualitative_run_dir(c) / "qgis" / "sr_pred.tif").is_file()]
    log_dir = ROOT / "logs" / FIG_TAG
    log_dir.mkdir(parents=True, exist_ok=True)

    def run(city: str, gpu: int) -> None:
        with open(log_dir / f"{city}.log", "w") as log:
            subprocess.run(qualitative_command(city, 0), cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                           env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}, check=True)

    threads = [threading.Thread(target=run, args=(c, gpus[i % len(gpus)])) for i, c in enumerate(todo)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def render_qualitative(sites: tuple[str, ...], out: Path) -> None:
    cols = ("LR (base frame)", "Bilinear", "ScaleF", "NIB", "ScaleF (zoom)", "NIB (zoom)")
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
            ax.imshow(highlight_compress(img), interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(cols[c], fontsize=8, pad=3)
        lp, lpb = m["lpips"], m.get("bilinear_lpips")
        label = city.replace("tromso", "troms\u00f8").title()
        axes[r, 0].set_ylabel(label, fontsize=8)
        meta.append({"site": city, "run_dir": str(run.relative_to(ROOT)), "lpips": lp, "bilinear_lpips": lpb,
                     "psnr": m.get("psnr"), "bilinear_psnr": m.get("bilinear_psnr")})
    fig.tight_layout(w_pad=0.1, h_pad=0.2)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    out.with_suffix(".json").write_text(json.dumps({"display": "highlight_compress(0, 0.4)",
                                                    "zoom_fraction": ZOOM_FRAC, "rows": meta}, indent=1) + "\n")
    print(f"wrote {out.with_suffix('.pdf')}")


def render_nested(out: Path, parent: str = "asker_p02_02") -> None:
    from eval.common_footprint import crop_hwc_frac, nest_window_frac
    from scripts.rescore_nested_common_footprint import SIDES, _covering_tile, _index_manifests, _run_dir
    from scripts.run_nested_float_validation import RUN_TAG

    def crop_for(side: int, child: dict, image: np.ndarray) -> np.ndarray:
        if side == 64:
            return image
        k = side // 64
        return crop_hwc_frac(image, *nest_window_frac(int(child["nest_iy"]) % k, int(child["nest_ix"]) % k, 64, side))

    val = json.loads((ROOT / "paper" / "results" / "nested_float_validation.json").read_text())
    rows = [r for r in val["rows"]["float_v3"] if r["side"] == 64 and r["parent_tile_id"] == parent]
    gains = sorted(rows, key=lambda r: r["lpips_bilinear"] - r["lpips"])
    chosen = gains[len(gains) // 2]
    median_gain = statistics.median(r["lpips_bilinear"] - r["lpips"] for r in rows)

    from scripts.rescore_nested_common_footprint import NEST_ROOT

    index = _index_manifests()
    children = json.loads((NEST_ROOT / "nested_lr64_manifest.json").read_text())["tiles"]
    child = next(c for c in children if c["tile_id"] == chosen["child_tile_id"])
    covering = {side: _covering_tile(child, side, index) for side in SIDES}
    dirs = {s: _run_dir(t["parent_city"], t["tile_id"], s, RUN_TAG) for s, t in covering.items()}
    gt = read_rgb(dirs[64] / "qgis" / "hr_gt.tif")
    bil = read_rgb(dirs[64] / "qgis" / "s2_bilinear.tif")
    preds = {s: crop_for(s, child, read_rgb(dirs[s] / "qgis" / "sr_pred.tif")) for s in (64, 128, 256, 512)}
    by_side = {r["side"]: r for r in val["rows"]["float_v3"] if r["child_tile_id"] == chosen["child_tile_id"]}

    labels = ["NIB", "Bilinear", *(f"ScaleF LR{s}" for s in (64, 128, 256, 512))]
    panels = [gt, bil, *(preds[s] for s in (64, 128, 256, 512))]
    fig, axes = plt.subplots(1, len(panels), figsize=(7.2, 1.55))
    for ax, label, img in zip(axes, labels, panels):
        ax.imshow(highlight_compress(img), interpolation="nearest")
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
        "selection": f"median LR64 LPIPS gain over bilinear among the {len(rows)} windows of {parent}",
        "child_tile_id": chosen["child_tile_id"], "median_gain": median_gain,
        "scores": {str(s): {k: by_side[s][k] for k in ("lpips", "lpips_bilinear", "psnr", "psnr_bilinear")}
                   for s in sorted(by_side)},
        "run_dirs": {str(s): str(d.relative_to(ROOT)) for s, d in dirs.items()},
        "display": "highlight_compress(0, 0.4)",
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"wrote {out.with_suffix('.pdf')}  child={chosen['child_tile_id']}  gain={median_gain:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("qualitative")
    q.add_argument("--sites", nargs="+", default=list(QUALITATIVE_SITES))
    q.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    q.add_argument("--render-only", action="store_true")
    q.add_argument("--out", type=Path, default=FIG_DIR / "qualitative_tiles_hc")
    n = sub.add_parser("nested")
    n.add_argument("--out", type=Path, default=FIG_DIR / "nested_spot_hc")
    args = ap.parse_args()
    if args.cmd == "qualitative":
        sites = tuple(args.sites)
        if not args.render_only:
            fit_qualitative(sites, args.gpus)
        render_qualitative(sites, args.out)
    else:
        render_nested(args.out)


if __name__ == "__main__":
    main()
