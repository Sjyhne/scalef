#!/usr/bin/env python3
"""Export the paper's same-geographic nested-spot figure.

The nest ladder runs kept metrics only (PNG/GeoTIFF exports were stripped).
This script selects one complete Asker (else paper7) 64/128/256/512 footprint,
re-fits those four tiles with GeoTIFF export, and crops the shared 640 m window.
Does not overwrite the freeze run directories.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.windows import from_bounds

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.bench_complete_patches import _cmd, _run_name  # noqa: E402
from scripts.run_complete_patch_size_ladder import _train_knobs  # noqa: E402

PAPER7 = ("asker", "bergen", "rana", "tromso", "amli", "vennesla", "trondheim")
NEST_JSON = {
    64: ROOT / "single_samples/sweep_results/bench_complete_patches_nest_lr64.json",
    128: ROOT / "single_samples/sweep_results/bench_complete_patches_nest_lr128.json",
    256: ROOT / "single_samples/sweep_results/bench_complete_patches_nest_lr256.json",
    512: ROOT / "single_samples/sweep_results/bench_complete_patches_nest_lr512.json",
}
PAPER_PREFIX = {
    64: "paper_spot_nest64",
    128: "paper_spot_nest128",
    256: "paper_spot_nest256",
    512: "paper_spot_nest512",
}
FREEZE_PREFIX = {
    64: "prod_full_nest64",
    128: "prod_full_nest128",
    256: "prod_k4_nest256",
    512: "prod_k4",
}


def _nested_index(tile_id: object, axis: str) -> int | None:
    match = re.search(rf"_{axis}(\d+)(?:_|$)", str(tile_id))
    return int(match.group(1)) if match else None


def _rows(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    return [row for row in data["rows"] if not row.get("error") and row.get("lpips") is not None]


def _index(rows: list[dict], side: int) -> dict[tuple, dict]:
    out: dict[tuple, dict] = {}
    for row in rows:
        y = _nested_index(row.get("tile_id"), "y")
        x = _nested_index(row.get("tile_id"), "x")
        if side == 512:
            key = (row.get("parent_city"), row.get("patch_row"), row.get("patch_col"))
        else:
            if y is None or x is None:
                continue
            key = (row.get("parent_city"), row.get("patch_row"), row.get("patch_col"), y, x)
        out[key] = row
    return out


def _select_spot(prefer_city: str | None) -> tuple[dict[int, dict], int, int]:
    by_side = {side: _index(_rows(path), side) for side, path in NEST_JSON.items()}
    ranked = []
    for row64 in by_side[64].values():
        y64 = _nested_index(row64.get("tile_id"), "y")
        x64 = _nested_index(row64.get("tile_id"), "x")
        if y64 is None or x64 is None:
            continue
        if (row64.get("eval_valid_fraction") or 0) < 0.9:
            continue
        base = (row64.get("parent_city"), row64.get("patch_row"), row64.get("patch_col"))
        selected = {64: row64}
        for side in (128, 256):
            key = (*base, (y64 * 64) // side, (x64 * 64) // side)
            if key in by_side[side]:
                selected[side] = by_side[side][key]
        if base in by_side[512]:
            selected[512] = by_side[512][base]
        if set(selected) != {64, 128, 256, 512}:
            continue
        city = str(row64.get("parent_city") or "")
        gain = float(row64.get("lpips_vs_bilinear") or 0.0)
        city_rank = 0 if city == prefer_city else (1 if city in PAPER7 else 2)
        edge = int(y64 in (0, 7) or x64 in (0, 7))
        ranked.append((city_rank, edge, -gain, selected, y64 * 64, x64 * 64))
    if not ranked:
        raise SystemExit("no complete 64/128/256/512 geographic spot in nest JSONs")
    ranked.sort(key=lambda item: item[:3])
    _, _, _, selected, gy, gx = ranked[0]
    return selected, gy, gx


def _run_dir(row: dict, side: int, prefix: str) -> Path:
    name = _run_name(row["tile_id"], prefix)
    return ROOT / "single_samples" / row["parent_city"] / "sample" / name


def _qgis_dir(row: dict, side: int) -> Path:
    return _run_dir(row, side, PAPER_PREFIX[side]) / "qgis"


def _freeze_png(row: dict, side: int, filename: str) -> Path:
    return _run_dir(row, side, FREEZE_PREFIX[side]) / filename


def _load_png(path: Path) -> np.ndarray:
    from PIL import Image

    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return arr


def _crop_frac(img: np.ndarray, fy: float, fx: float, fh: float, fw: float) -> np.ndarray:
    h, w = img.shape[:2]
    y0 = int(round(fy * h))
    x0 = int(round(fx * w))
    y1 = max(y0 + 1, int(round((fy + fh) * h)))
    x1 = max(x0 + 1, int(round((fx + fw) * w)))
    return img[y0:y1, x0:x1]


def _try_existing_pngs(selected: dict[int, dict], global_y: int, global_x: int) -> list[np.ndarray] | None:
    row64 = selected[64]
    required = [
        _freeze_png(row64, 64, "ground_truth.png"),
        _freeze_png(row64, 64, "bilinear_baseline.png"),
        *(_freeze_png(selected[side], side, "model_output_aligned.png") for side in (64, 128, 256, 512)),
    ]
    if not all(path.is_file() for path in required):
        missing = [str(path) for path in required if not path.is_file()]
        print("missing freeze PNGs:\n  " + "\n  ".join(missing[:6]), flush=True)
        return None
    gt = _load_png(required[0])
    bilinear = _load_png(required[1])
    preds = []
    for side in (64, 128, 256, 512):
        img = _load_png(_freeze_png(selected[side], side, "model_output_aligned.png"))
        fy = (global_y % side) / side
        fx = (global_x % side) / side
        preds.append(_crop_frac(img, fy, fx, 64 / side, 64 / side))
    return [gt, bilinear, *preds]


def _fit_if_needed(row: dict, side: int, device: int, iters: int) -> Path:
    qgis = _qgis_dir(row, side)
    if (qgis / "sr_pred.tif").is_file() and (qgis / "hr_gt.tif").is_file():
        print(f"reuse {qgis}", flush=True)
        return qgis
    knobs = _train_knobs(side)
    cmd = _cmd(
        row,
        device,
        iters,
        export_geotiff=True,
        lr_tile=knobs["lr_tile"],
        lr_tiles_per_step=knobs["lr_tiles_per_step"],
        run_prefix=PAPER_PREFIX[side],
    )
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    if not (qgis / "sr_pred.tif").is_file():
        raise SystemExit(f"missing sr_pred.tif after fit: {qgis}")
    return qgis


def _read_window(path: Path, bounds) -> np.ndarray:
    with rasterio.open(path) as src:
        window = from_bounds(*bounds, transform=src.transform)
        arr = src.read(window=window, boundless=True, fill_value=0).astype(np.float32)
    arr = np.transpose(arr, (1, 2, 0))
    if arr.max() > 1.5:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def _stretch(images: list[np.ndarray], ref: np.ndarray, pct: float) -> list[np.ndarray]:
    lo, hi = np.percentile(ref, [pct, 100.0 - pct])
    span = max(float(hi - lo), 1e-6)
    return [np.clip((img - lo) / span, 0.0, 1.0) for img in images]


def _save_panel(
    panels: list[tuple[str, np.ndarray]],
    destination: Path,
    caption: str,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, len(panels), figsize=(7.2, 1.55))
    for ax, (label, img) in zip(axes, panels):
        ax.imshow(img, interpolation="nearest")
        ax.set_title(label, fontsize=8, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.tight_layout(w_pad=0.15)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(destination.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {destination.with_suffix('.pdf')}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--city", default="asker")
    ap.add_argument("--device", type=int, default=7)
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--refit", action="store_true", help="Retrain the four tiles with GeoTIFF export.")
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "ScaleF_Overleaf" / "figures" / "nested_same_geographic_64_spot",
    )
    args = ap.parse_args()

    selected, global_y, global_x = _select_spot(args.city)
    row64 = selected[64]
    print(
        f"spot {row64['tile_id']}  gain={row64.get('lpips_vs_bilinear'):.3f}  "
        f"global_yx=({global_y},{global_x})",
        flush=True,
    )
    source = "freeze_png"
    images = None if args.refit else _try_existing_pngs(selected, global_y, global_x)
    qgis_rel = {}
    if images is None:
        if not args.refit:
            raise SystemExit("freeze PNGs missing; pass --refit when a GPU is free")
        source = "refit_geotiff"
        qgis = {}
        for side in (64, 128, 256, 512):
            qgis[side] = _fit_if_needed(selected[side], side, args.device, args.iters)
        with rasterio.open(qgis[64] / "hr_gt.tif") as src:
            bounds = src.bounds
        gt = _read_window(qgis[64] / "hr_gt.tif", bounds)
        bilinear = _read_window(qgis[64] / "s2_bilinear.tif", bounds)
        preds = [_read_window(qgis[side] / "sr_pred.tif", bounds) for side in (64, 128, 256, 512)]
        images = _stretch([gt, bilinear, *preds], gt, 2.0)
        qgis_rel = {str(side): str(qgis[side].relative_to(ROOT)) for side in qgis}
    shown = images
    labels = [
        "NIB",
        "Bilinear",
        "ScaleF LR64",
        "ScaleF LR128",
        "ScaleF LR256",
        "ScaleF LR512",
    ]
    city = str(row64["parent_city"]).title()
    prow, pcol = row64["patch_row"], row64["patch_col"]
    y64, x64 = _nested_index(row64["tile_id"], "y"), _nested_index(row64["tile_id"], "x")
    caption = (
        f"{city} complete tile p{prow:02d}_{pcol:02d}, child y{y64:02d}_x{x64:02d} "
        "(640 m window; LR64/128 full-field, LR256/512 k4)"
    )
    _save_panel(list(zip(labels, shown)), args.out, caption)
    meta = {
        "tile_id": row64["tile_id"],
        "parent_city": row64["parent_city"],
        "patch_row": prow,
        "patch_col": pcol,
        "y64": y64,
        "x64": x64,
        "global_y": global_y,
        "global_x": global_x,
        "lpips": row64.get("lpips"),
        "lpips_vs_bilinear": row64.get("lpips_vs_bilinear"),
        "source": source,
        "qgis": qgis_rel,
    }
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
