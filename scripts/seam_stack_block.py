#!/usr/bin/env python3
"""Stacked seam stages on a large 32VNM block.

Stage 0  independent 2% identity (existing prod_k4_base2)
Stage 1  + date-ladder ICM identity (prod_k4_icm, new trains)
Stage 2  + coarse S2 reference colour on leftover tiles (appendix only)
Stage 3  + date-cut ramp (appendix stack applies this after colour)
Delivery preview (--stage delivery): ICM + date-cut ramp, no colour.

Each stage writes a shared-stretch overview and the same zoom crops so
additions are visible. Orthophotos unused. No full-tile SR histogram matching.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.s2_cloud_mask import cloudy_from_scl  # noqa: E402
from scripts.mosaic_seam_ramp import _ramp_one_cell, _read_strip  # noqa: E402
from scripts.national_cell_queue import frame_day  # noqa: E402
from scripts.run_production import _sr_geotiff_path  # noqa: E402
from scripts.seam_pilot_3x3 import _fixed_stretch, apply_gain  # noqa: E402

OUT = ROOT / "production/seams/32VNM_stack"
PARENT = "32VNM"
S2_PARENT = ROOT / "data/s2_revisits/national_2025_v2/32VNM"
GRANULE_MANIFEST = S2_PARENT / "granule_tiles_lr512_manifest.json"
CELL_HR = 2048
LR_SIDE = 512
DF = 4
DS = 8  # overview downsample → 20 m


def _load_plan(path: Path) -> dict:
    return json.loads(path.read_text())


def write_block_manifest(plan: dict) -> Path:
    man = json.loads(GRANULE_MANIFEST.read_text())
    keep_ids = {c["tile_id"] for c in plan["cells"]}
    tiles = []
    for t in man["tiles"]:
        if t["tile_id"] not in keep_ids:
            continue
        row = dict(t)
        row["force_base_date"] = plan["assignment"].get(t["tile_id"])
        tiles.append(row)
    tiles.sort(key=lambda t: (t["iy"], t["ix"]))
    out = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "parent": PARENT,
        "side": LR_SIDE,
        "stride": LR_SIDE,
        "overlap_frac": 0.0,
        "overlap_px": 0,
        "pilot": "stack_block",
        "block": plan.get("block"),
        "identity_primary": plan.get("primary"),
        "tiles": tiles,
        "src": man.get("src"),
        "merge_method_hint": "first",
    }
    path = OUT / "manifest_block.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")
    changed = [
        t
        for t in tiles
        if plan["assignment"].get(t["tile_id"]) != plan["independent"].get(t["tile_id"])
    ]
    chg = dict(out)
    chg["tiles"] = changed
    chg["pilot"] = "stack_block_icm_changed"
    chpath = OUT / "manifest_icm_changed.json"
    chpath.write_text(json.dumps(chg, indent=2) + "\n")
    print(f"Wrote {path}  {len(tiles)} tiles", flush=True)
    print(f"Wrote {chpath}  {len(changed)} identity-changed tiles", flush=True)
    return path


def _grid(plan: dict, prefix: str, *, fallback_unchanged: bool = False) -> dict[tuple[int, int], Path]:
    """ICM mosaic: new trains where the date changed, else reuse base2."""
    grid = {}
    for c in plan["cells"]:
        p = _sr_geotiff_path(PARENT, c["tile_id"], prefix)
        if fallback_unchanged and c.get("icm_date") == c.get("independent_date"):
            p = _sr_geotiff_path(PARENT, c["tile_id"], "prod_k4_base2")
        if p.is_file():
            grid[c["iy"], c["ix"]] = p
    return grid


def _read_rgb(path: Path) -> np.ndarray:
    import rasterio

    with rasterio.open(path) as src:
        return np.transpose(src.read(indexes=(1, 2, 3)), (1, 2, 0)).astype(np.float32)


def _block_canvas_meta(grid: dict[tuple[int, int], Path]):
    ys = [iy for iy, _ in grid]
    xs = [ix for _, ix in grid]
    y0, x0 = min(ys), min(xs)
    y1, x1 = max(ys) + 1, max(xs) + 1
    return y0, x0, y1, x1


def stitch_preview(grid: dict[tuple[int, int], Path], *, rgb_at=None) -> np.ndarray:
    """Downsampled block mosaic. ``rgb_at`` overrides a cell's array (HWC)."""
    y0, x0, y1, x1 = _block_canvas_meta(grid)
    th, tw = CELL_HR // DS, CELL_HR // DS
    canvas = np.zeros(((y1 - y0) * th, (x1 - x0) * tw, 3), np.float32)
    for (iy, ix), path in grid.items():
        arr = rgb_at[(iy, ix)] if rgb_at and (iy, ix) in rgb_at else _read_rgb(path)
        small = arr[::DS, ::DS]
        r0, c0 = (iy - y0) * th, (ix - x0) * tw
        h, w = small.shape[:2]
        canvas[r0 : r0 + h, c0 : c0 + w] = small[:th, :tw]
    return canvas


def neighbour_stats(
    grid: dict[tuple[int, int], Path],
    dates: dict[tuple[int, int], str | None],
    *,
    rgb_at=None,
) -> dict:
    """Edge MAE / Δμ on abutting pixels, split by same-date vs date-cut."""
    same, cuts = [], []
    for iy, ix in grid:
        if (iy, ix + 1) in grid:
            a = rgb_at[(iy, ix)] if rgb_at and (iy, ix) in rgb_at else _read_rgb(grid[iy, ix])
            b = (
                rgb_at[(iy, ix + 1)]
                if rgb_at and (iy, ix + 1) in rgb_at
                else _read_rgb(grid[iy, ix + 1])
            )
            mae = float(np.abs(a[:, -1, :] - b[:, 0, :]).mean())
            dmu = float(a[:, -1, :].mean() - b[:, 0, :].mean())
            rec = {"pair": f"y{iy:02d}_x{ix:02d}|x{ix+1:02d}", "mae": mae, "dmu": dmu}
            if dates.get((iy, ix)) == dates.get((iy, ix + 1)):
                same.append(rec)
            else:
                cuts.append(rec)
        if (iy + 1, ix) in grid:
            a = rgb_at[(iy, ix)] if rgb_at and (iy, ix) in rgb_at else _read_rgb(grid[iy, ix])
            b = (
                rgb_at[(iy + 1, ix)]
                if rgb_at and (iy + 1, ix) in rgb_at
                else _read_rgb(grid[iy + 1, ix])
            )
            mae = float(np.abs(a[-1, :, :] - b[0, :, :]).mean())
            dmu = float(a[-1, :, :].mean() - b[0, :, :].mean())
            rec = {"pair": f"y{iy:02d}|y{iy+1:02d}_x{ix:02d}", "mae": mae, "dmu": dmu}
            if dates.get((iy, ix)) == dates.get((iy + 1, ix)):
                same.append(rec)
            else:
                cuts.append(rec)

    def _agg(rows):
        if not rows:
            return {"n": 0, "mae": None, "abs_dmu": None}
        return {
            "n": len(rows),
            "mae": float(np.mean([r["mae"] for r in rows])),
            "abs_dmu": float(np.mean([abs(r["dmu"]) for r in rows])),
            "worst": max(rows, key=lambda r: r["mae"]),
        }

    return {"same_date": _agg(same), "date_cuts": _agg(cuts)}


def fit_tile_to_reference(
    ident_lr: np.ndarray,
    ref_lr: np.ndarray,
    scl: np.ndarray | None,
    *,
    change_thr: float = 0.25,
) -> dict | None:
    import cv2

    if ident_lr.shape != ref_lr.shape:
        ref_lr = cv2.resize(ref_lr, (ident_lr.shape[1], ident_lr.shape[0]), interpolation=cv2.INTER_AREA)
    valid = (ident_lr.sum(2) > 1e-5) & (ref_lr.sum(2) > 1e-5)
    if scl is not None:
        cloudy = cloudy_from_scl(scl, include_shadow=True)
        if cloudy.shape == valid.shape:
            valid &= ~cloudy
    delta = np.abs(ident_lr - ref_lr).mean(axis=2)
    valid &= delta < change_thr
    if int(valid.sum()) < 64:
        return None
    k = 9
    fi = cv2.blur(ident_lr, (k, k))
    fr = cv2.blur(ref_lr, (k, k))
    a, b = [], []
    for c in range(3):
        x = fi[..., c][valid].astype(np.float64)
        y = fr[..., c][valid].astype(np.float64)
        A = np.stack([x, np.ones_like(x)], axis=1)
        # mild regularize toward a=1, b=0
        extra = np.array([[40.0, 0.0], [0.0, 80.0]])
        extra_b = np.array([40.0, 0.0])
        coef, *_ = np.linalg.lstsq(
            np.vstack([A, extra]), np.concatenate([y, extra_b]), rcond=None
        )
        a.append(float(coef[0]))
        b.append(float(coef[1]))
    return {"a": a, "b": b, "n": int(valid.sum())}


def colour_leftovers(
    plan: dict,
    grid: dict[tuple[int, int], Path],
) -> tuple[dict[tuple[int, int], np.ndarray], dict]:
    import rasterio
    from rasterio.windows import Window

    parent_meta = json.loads((S2_PARENT / "meta.json").read_text())
    frames_by_day = {frame_day(fr): fr for fr in parent_meta.get("frames") or []}
    primary = plan["primary"]
    fr = frames_by_day.get(primary)
    if fr is None:
        raise SystemExit(f"primary {primary} not in parent frames")
    rgb_path = S2_PARENT / fr["path"]
    scl_path = S2_PARENT / fr["scl_path"] if fr.get("scl_path") else None
    by_id = {c["tile_id"]: c for c in plan["cells"]}
    rgb_at: dict[tuple[int, int], np.ndarray] = {}
    fits = []
    failures = []
    from contextlib import ExitStack

    with ExitStack() as stack:
        src = stack.enter_context(rasterio.open(rgb_path))
        sscl = (
            stack.enter_context(rasterio.open(scl_path))
            if scl_path is not None and scl_path.is_file()
            else None
        )
        for (iy, ix), path in grid.items():
            rec = by_id.get(f"{PARENT}_t512_y{iy:02d}_x{ix:02d}")
            sr = _read_rgb(path)
            if rec is None or not rec.get("leftover"):
                rgb_at[(iy, ix)] = sr
                continue
            row0, col0 = iy * LR_SIDE, ix * LR_SIDE
            ref = np.transpose(
                src.read(indexes=(1, 2, 3), window=Window(col0, row0, LR_SIDE, LR_SIDE)),
                (1, 2, 0),
            ).astype(np.float32)
            if float(ref.max()) > 1.5:
                ref = ref / 10000.0
            scl = None
            if sscl is not None:
                scl = sscl.read(1, window=Window(col0, row0, LR_SIDE, LR_SIDE))
            lr_path = path.with_name("s2_lr.tif")
            ident = _read_rgb(lr_path) if lr_path.is_file() else sr[::DF, ::DF]
            fit = fit_tile_to_reference(ident, ref, scl)
            if fit is None:
                failures.append({"iy": iy, "ix": ix, "reason": "too_few_pixels"})
                rgb_at[(iy, ix)] = sr
                continue
            rgb_at[(iy, ix)] = apply_gain(sr, fit)
            fits.append({"iy": iy, "ix": ix, **fit})
    return rgb_at, {"fits": fits, "failures": failures, "primary": primary}


def ramp_date_cuts(
    plan: dict,
    grid: dict[tuple[int, int], Path],
    rgb_at: dict[tuple[int, int], np.ndarray],
    *,
    ramp_px: int = 128,
) -> dict[tuple[int, int], np.ndarray]:
    dates = {(c["iy"], c["ix"]): c["icm_date"] for c in plan["cells"]}
    out = {}
    for (iy, ix), path in grid.items():
        arr = rgb_at[(iy, ix)]
        west = east = north = south = None
        if (iy, ix - 1) in grid and dates.get((iy, ix - 1)) != dates.get((iy, ix)):
            west = rgb_at[(iy, ix - 1)][:, -8:, :]
        if (iy, ix + 1) in grid and dates.get((iy, ix + 1)) != dates.get((iy, ix)):
            east = rgb_at[(iy, ix + 1)][:, :8, :]
        if (iy - 1, ix) in grid and dates.get((iy - 1, ix)) != dates.get((iy, ix)):
            north = rgb_at[(iy - 1, ix)][-8:, :, :]
        if (iy + 1, ix) in grid and dates.get((iy + 1, ix)) != dates.get((iy, ix)):
            south = rgb_at[(iy + 1, ix)][:8, :, :]
        out[(iy, ix)] = _ramp_one_cell(
            arr,
            west_strip=west,
            east_strip=east,
            north_strip=north,
            south_strip=south,
            ramp_px=ramp_px,
        )
    return out


def _zoom(canvas, grid, iy, ix, half=160):
    y0, x0, _y1, _x1 = _block_canvas_meta(grid)
    th, tw = CELL_HR // DS, CELL_HR // DS
    cy = (iy - y0) * th + th // 2
    cx = (ix - x0) * tw + tw
    return canvas[
        max(0, cy - half) : cy + half,
        max(0, cx - half) : cx + half,
    ]


def write_stage_figure(
    stages: list[tuple[str, np.ndarray]],
    grid: dict,
    lo,
    hi,
    path: Path,
    zooms: list[tuple[str, int, int]],
    title: str = "32VNM stack  ·  one shared 2–98% stretch  ·  each column adds one technique",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(stages)
    fig, axes = plt.subplots(
        1 + len(zooms),
        n,
        figsize=(4.2 * n, 4.4 * (1 + len(zooms))),
        squeeze=False,
    )
    for j, (name, rgb) in enumerate(stages):
        axes[0, j].imshow(_fixed_stretch(rgb, lo, hi), interpolation="nearest")
        axes[0, j].set_title(name, fontsize=9)
        axes[0, j].set_axis_off()
        for i, (zname, iy, ix) in enumerate(zooms, start=1):
            axes[i, j].imshow(
                _fixed_stretch(_zoom(rgb, grid, iy, ix), lo, hi),
                interpolation="nearest",
            )
            if j == 0:
                axes[i, j].set_ylabel(zname, fontsize=8)
            axes[i, j].set_axis_off()
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def run_previews(plan: dict, *, have_icm: bool) -> dict:
    dates0 = {(c["iy"], c["ix"]): c["independent_date"] for c in plan["cells"]}
    dates1 = {(c["iy"], c["ix"]): c["icm_date"] for c in plan["cells"]}
    g0 = _grid(plan, "prod_k4_base2")
    if len(g0) < 8:
        raise SystemExit(f"base2 grid too small: {len(g0)}")
    prev0 = stitch_preview(g0)
    stats0 = neighbour_stats(g0, dates0)
    stages = [("0  independent 2%", prev0)]
    rgb1 = None
    g1 = {}
    stats1 = stats2 = stats3 = None
    colour_info = None

    if have_icm:
        g1 = _grid(plan, "prod_k4_icm", fallback_unchanged=True)
        if len(g1) < len(plan["cells"]) * 0.9:
            print(f"WARN: ICM tiles {len(g1)}/{len(plan['cells'])}", flush=True)
        prev1 = stitch_preview(g1)
        stats1 = neighbour_stats(g1, dates1)
        stages.append(("1  + ICM identity", prev1))
        rgb_at, colour_info = colour_leftovers(plan, g1)
        prev2 = stitch_preview(g1, rgb_at=rgb_at)
        stats2 = neighbour_stats(g1, dates1, rgb_at=rgb_at)
        stages.append(("2  + S2 ref colour", prev2))
        rgb_r = ramp_date_cuts(plan, g1, rgb_at, ramp_px=128)
        prev3 = stitch_preview(g1, rgb_at=rgb_r)
        stats3 = neighbour_stats(g1, dates1, rgb_at=rgb_r)
        stages.append(("3  + date-cut ramp", prev3))

    stack = np.concatenate([s[1].reshape(-1, 3) for s in stages], axis=0)
    ok = stack.sum(1) > 1e-5
    lo, hi = np.percentile(stack[ok], [2, 98], axis=0)
    zooms = [
        ("y04_x17|x18 date island", 4, 17),
        ("y09_x12|x13 Jul18 edge", 9, 12),
        ("y02_x14 same-date interior", 2, 14),
    ]
    png = OUT / ("stack_0.png" if not have_icm else "stack_0_to_3.png")
    write_stage_figure(stages, g0 if not have_icm else g1 or g0, lo, hi, png, zooms)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_base2": len(g0),
        "n_icm": len(g1),
        "stage0_independent": stats0,
        "stage1_icm": stats1,
        "stage2_colour": stats2,
        "stage3_ramp": stats3,
        "colour": colour_info,
        "png": str(png.relative_to(ROOT)),
        "primary": plan.get("primary"),
        "n_leftover": plan.get("n_leftover"),
        "n_joined_primary": plan.get("n_joined_primary"),
        "boundaries_independent": plan.get("n_boundaries_independent"),
        "boundaries_icm": plan.get("n_boundaries_icm"),
    }
    (OUT / ("stack_0.json" if not have_icm else "stack_0_to_3.json")).write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps({k: report[k] for k in ("png", "stage0_independent", "stage1_icm", "stage2_colour", "stage3_ramp") if report.get(k)}, indent=2))
    return report


def write_delivery_preview(
    plan: dict,
    g0,
    g1,
    dates0,
    dates1,
    lo,
    hi,
    zooms,
) -> dict:
    """Main-text pair: independent 2% vs ICM + date-cut ramp (no S2 colour)."""
    prev0 = stitch_preview(g0)
    rgb_at = {xy: _read_rgb(p) for xy, p in g1.items()}
    rgb_r = ramp_date_cuts(plan, g1, rgb_at, ramp_px=128)
    prev_d = stitch_preview(g1, rgb_at=rgb_r)
    stats_d = neighbour_stats(g1, dates1, rgb_at=rgb_r)
    png = OUT / "delivery_icm_ramp.png"
    write_stage_figure(
        [
            ("independent 2% identity", prev0),
            ("delivery: ICM + date-cut ramp", prev_d),
        ],
        g1,
        lo,
        hi,
        png,
        zooms,
        title="32VNM  ·  shared 2–98% stretch  ·  shipped mosaic = ICM + date-cut ramp",
    )
    rec = {
        "png": str(png.relative_to(ROOT)),
        "recipe": "icm_identity + date_cut_ramp",
        "colour": False,
        "ramp_px": 128,
        "neighbour": stats_d,
        "n_leftover": plan.get("n_leftover"),
        "n_joined_primary": plan.get("n_joined_primary"),
        "boundaries_independent": plan.get("n_boundaries_independent"),
        "boundaries_icm": plan.get("n_boundaries_icm"),
    }
    (OUT / "delivery_icm_ramp.json").write_text(json.dumps(rec, indent=2) + "\n")
    print("delivery", rec["png"], flush=True)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--identity-plan", type=Path, default=OUT / "identity_plan.json")
    ap.add_argument(
        "--stage",
        choices=["manifest", "preview0", "preview", "delivery", "all"],
        default="all",
    )
    args = ap.parse_args()
    ipath = args.identity_plan if args.identity_plan.is_absolute() else ROOT / args.identity_plan
    plan = _load_plan(ipath)
    if args.stage in ("manifest", "all"):
        write_block_manifest(plan)
    if args.stage == "preview0":
        run_previews(plan, have_icm=False)
    if args.stage in ("preview", "all"):
        have = any(
            _sr_geotiff_path(PARENT, c["tile_id"], "prod_k4_icm").is_file()
            for c in plan["cells"]
            if c.get("icm_date") != c.get("independent_date")
        )
        run_previews(plan, have_icm=have)
    if args.stage == "delivery":
        g0 = _grid(plan, "prod_k4_base2")
        g1 = _grid(plan, "prod_k4_icm", fallback_unchanged=True)
        dates0 = {(c["iy"], c["ix"]): c["independent_date"] for c in plan["cells"]}
        dates1 = {(c["iy"], c["ix"]): c["icm_date"] for c in plan["cells"]}
        prev0 = stitch_preview(g0)
        stack = prev0.reshape(-1, 3)
        ok = stack.sum(1) > 1e-5
        lo, hi = np.percentile(stack[ok], [2, 98], axis=0)
        zooms = [
            ("y04_x17|x18 date island", 4, 17),
            ("y09_x12|x13 Jul18 edge", 9, 12),
            ("y02_x14 same-date interior", 2, 14),
        ]
        write_delivery_preview(plan, g0, g1, dates0, dates1, lo, hi, zooms)


if __name__ == "__main__":
    main()
