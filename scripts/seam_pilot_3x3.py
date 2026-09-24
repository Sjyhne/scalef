#!/usr/bin/env python3
"""32VNM 3×3 seam pilot: diagnose, overlap-train, colour-balance, blend.

Block is original LR512 cells y03–y05 × x16–x18 (includes y04_x17|x18).
Overlap trains keep the independent 2% identity rule. Common-base is a
separate reconstruction run. Colour gains are fit on LR (s2_bilinear),
never on independent full-tile SR histograms. Orthophotos are unused.

OTB Mosaic is not required; feather + Laplacian blend are implemented here.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.s2_cloud_mask import score_scl_cell  # noqa: E402
from scripts.make_granule_tiles import _write_tile  # noqa: E402
from scripts.mosaic_granule_sr import _edge_distance_weights  # noqa: E402
from scripts.national_cell_queue import filter_frames_for_cell, frame_day  # noqa: E402
from scripts.run_production import _sr_geotiff_path  # noqa: E402

Y0, X0 = 3, 16
N = 3
SIDE = 512
DF = 4
CENTER = date(2025, 7, 15)
PARENT = "32VNM"
S2_PARENT = ROOT / "data/s2_revisits/national_2025_v2/32VNM"
PLAN_ONDISK = ROOT / "production/national_2025/plans/32VNM/plan_ondisk.json"
GRANULE_MANIFEST = (
    ROOT / "data/s2_revisits/national_2025_v2/32VNM/granule_tiles_lr512_manifest.json"
)
OUT = ROOT / "production/seams/32VNM_3x3"
def _orig_id(iy: int, ix: int) -> str:
    return f"{PARENT}_t512_y{iy:02d}_x{ix:02d}"


def _qgis(parent: str, tile_id: str, prefix: str, name: str) -> Path:
    return _sr_geotiff_path(parent, tile_id, prefix).with_name(name)


def _read_rgb(path: Path) -> tuple[np.ndarray, object, object, float]:
    import rasterio

    with rasterio.open(path) as src:
        rgb = np.transpose(src.read(indexes=(1, 2, 3)), (1, 2, 0)).astype(np.float32)
        gsd = float(abs(src.transform.a))
        return rgb, src.transform, src.crs, gsd


def _fixed_stretch(rgb: np.ndarray, lo, hi, gamma: float = 1.0) -> np.ndarray:
    x = np.clip((rgb - lo) / np.maximum(hi - lo, 1e-6), 0.0, 1.0)
    if abs(gamma - 1.0) > 1e-6:
        x = np.power(np.clip(x, 0.0, 1.0), float(gamma))
    return x


def _phase_shift(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    la = a.mean(axis=2)
    lb = b.mean(axis=2)
    h = min(la.shape[0], lb.shape[0])
    w = min(la.shape[1], lb.shape[1])
    la, lb = la[:h, :w], lb[:h, :w]
    fa = np.fft.fft2(la - la.mean())
    fb = np.fft.fft2(lb - lb.mean())
    r = fa * np.conj(fb)
    r /= np.maximum(np.abs(r), 1e-9)
    peak = np.fft.ifft2(r).real
    iy, ix = np.unravel_index(int(np.argmax(peak)), peak.shape)
    if iy > h // 2:
        iy -= h
    if ix > w // 2:
        ix -= w
    return float(iy), float(ix)


def _hp_energy(x: np.ndarray) -> float:
    return float(np.abs(x - x.mean(axis=(0, 1), keepdims=True)).mean())


def _box_down(x: np.ndarray, k: int) -> np.ndarray:
    import cv2

    k = int(max(1, k))
    if k % 2 == 0:
        k += 1
    return cv2.blur(x, (k, k))


def _cell_scl_cloud(scl_path: Path, row0: int, col0: int, side: int) -> float | None:
    import rasterio
    from rasterio.windows import Window

    if not scl_path.is_file():
        return None
    with rasterio.open(scl_path) as src:
        scl = src.read(1, window=Window(col0, row0, side, side))
    _, cloud, _snow, _valid = score_scl_cell(
        scl, max_cloud_frac=1.0, min_valid_frac=0.0
    )
    return None if cloud != cloud else float(cloud)


def _consensus_date(cell_dates: list[dict], *, max_cloud: float) -> dict:
    """Date covering the most cells at or under ``max_cloud``, closest to CENTER."""
    counts: dict[str, list[str]] = {}
    for rec in cell_dates:
        for d, cloud in rec["date_cloud"].items():
            if cloud is not None and cloud <= max_cloud:
                counts.setdefault(d, []).append(rec["tile_id"])
    if not counts:
        return {"date": None, "n": 0, "tiles": [], "max_cloud": max_cloud}
    best = max(
        counts,
        key=lambda d: (
            len(counts[d]),
            -abs((date.fromisoformat(d) - CENTER).days),
            d,
        ),
    )
    return {
        "date": best,
        "n": len(counts[best]),
        "tiles": sorted(counts[best]),
        "max_cloud": max_cloud,
        "all_nine": len(counts[best]) == N * N,
    }


def write_subset_manifest() -> Path:
    man = json.loads(GRANULE_MANIFEST.read_text())
    keep = []
    for t in man["tiles"]:
        if Y0 <= int(t["iy"]) < Y0 + N and X0 <= int(t["ix"]) < X0 + N:
            keep.append(t)
    keep.sort(key=lambda t: (t["iy"], t["ix"]))
    out = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "parent": PARENT,
        "side": SIDE,
        "stride": SIDE,
        "overlap_frac": 0.0,
        "overlap_px": 0,
        "pilot": "3x3_base2_subset",
        "src": man.get("src"),
        "tiles": keep,
        "merge_method_hint": "first",
        "coverage_lr_px": N * SIDE,
        "coverage_km": N * SIDE * 0.01,
    }
    path = OUT / "manifest_3x3_base2.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")
    return path


def diagnose(prefix: str = "prod_k4_base2", stem: str = "diagnose") -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rasterio

    OUT.mkdir(parents=True, exist_ok=True)
    bases = {}
    log_path = ROOT / "production/national_2025/logs/train_32VNM_base2.log"
    if prefix == "prod_k4_base2" and log_path.is_file():
        date_re = re.compile(
            r"32VNM_t512_y(\d+)_x(\d+): .*?base frame 0 \(([^,]+), "
            r"(\d{4}-\d{2}-\d{2}), cloud ([0-9.]+)%"
        )
        for m in date_re.finditer(log_path.read_text(errors="replace")):
            bases[(int(m.group(1)), int(m.group(2)))] = {
                "file": m.group(3),
                "date": m.group(4),
                "cloud_pct": float(m.group(5)),
            }

    parent_meta = json.loads((S2_PARENT / "meta.json").read_text())
    plan = json.loads(PLAN_ONDISK.read_text())
    plan_by_xy = {(c["iy"], c["ix"]): c for c in plan["cells"]}
    frames_by_day = {frame_day(fr): fr for fr in parent_meta.get("frames") or []}

    tiles = []
    rgbs = []
    cell_dates = []
    for j in range(N):
        for i in range(N):
            iy, ix = Y0 + j, X0 + i
            tid = _orig_id(iy, ix)
            path = _sr_geotiff_path(PARENT, tid, prefix)
            rgb, transform, crs, gsd = _read_rgb(path)
            mpath = (
                ROOT
                / "single_samples"
                / PARENT
                / "sample"
                / f"{prefix}_{tid}"
                / "metrics.json"
            )
            if mpath.is_file():
                bf = json.loads(mpath.read_text()).get("base_frame") or {}
                if bf.get("date"):
                    bases[(iy, ix)] = {
                        "file": None,
                        "date": bf.get("date"),
                        "cloud_pct": None
                        if bf.get("cloud_frac") is None
                        else round(100.0 * float(bf["cloud_frac"]), 2),
                        "force_date": bf.get("force_date"),
                    }
            rec = {
                "tile_id": tid,
                "iy": iy,
                "ix": ix,
                "path": str(path.relative_to(ROOT)),
                "crs": str(crs),
                "gsd_m": gsd,
                "width": int(rgb.shape[1]),
                "height": int(rgb.shape[0]),
                "origin_xy": [float(transform.c), float(transform.f)],
                "pixel_size": [float(transform.a), float(transform.e)],
                "mean_rgb": [float(x) for x in rgb.mean(axis=(0, 1))],
                "base": bases.get((iy, ix)),
                "export": {
                    "per_tile_percentile_stretch": False,
                    "per_tile_gamma": False,
                    "unstandardize": "lr_mean/std of frozen base frame 0",
                    "clip": "[0,1] reflectance",
                },
            }
            tiles.append(rec)
            rgbs.append(rgb)

            prow = plan_by_xy.get((iy, ix), {})
            date_cloud = {}
            for d in prow.get("dates") or []:
                fr = frames_by_day.get(d)
                if fr is None or not fr.get("scl_path"):
                    date_cloud[d] = None
                    continue
                date_cloud[d] = _cell_scl_cloud(
                    S2_PARENT / fr["scl_path"],
                    int(prow["row_off"]),
                    int(prow["col_off"]),
                    SIDE,
                )
            cell_dates.append(
                {
                    "tile_id": tid,
                    "iy": iy,
                    "ix": ix,
                    "stack_dates": list(prow.get("dates") or []),
                    "date_cloud": date_cloud,
                    "july12_cloud": date_cloud.get("2025-07-12"),
                    "july12_ok_2pct": (
                        date_cloud.get("2025-07-12") is not None
                        and date_cloud["2025-07-12"] <= 0.02
                    ),
                    "july12_ok_5pct": (
                        date_cloud.get("2025-07-12") is not None
                        and date_cloud["2025-07-12"] <= 0.05
                    ),
                }
            )

    crs_ok = len({t["crs"] for t in tiles}) == 1
    gsd_ok = max(abs(t["gsd_m"] - 2.5) for t in tiles) < 1e-6
    gaps = []
    for rec in tiles:
        j, i = rec["iy"] - Y0, rec["ix"] - X0
        if i + 1 < N:
            east = tiles[j * N + (i + 1)]
            expected_x = rec["origin_xy"][0] + rec["width"] * rec["pixel_size"][0]
            dx = east["origin_xy"][0] - expected_x
            gaps.append(
                {
                    "seam": f"{rec['tile_id']}|east",
                    "dx_m": dx,
                    "aligned": abs(dx) < 0.01,
                }
            )
        if j + 1 < N:
            south = tiles[(j + 1) * N + i]
            expected_y = rec["origin_xy"][1] + rec["height"] * rec["pixel_size"][1]
            dy = south["origin_xy"][1] - expected_y
            gaps.append(
                {
                    "seam": f"{rec['tile_id']}|south",
                    "dy_m": dy,
                    "aligned": abs(dy) < 0.01,
                }
            )

    cuts = []
    pad = 64
    for j in range(N):
        for i in range(N - 1):
            left = rgbs[j * N + i]
            right = rgbs[j * N + i + 1]
            l = left[:, -pad:]
            r = right[:, :pad]
            mae = float(np.abs(left[:, -1, :] - right[:, 0, :]).mean())
            dmu = float(left[:, -1, :].mean() - right[:, 0, :].mean())
            dy, dx = _phase_shift(l, r)
            # 0-overlap 64 px strips are adjacent land, not corresponding pixels.
            if abs(dmu) > 0.008:
                kind = "colour_jump"
            elif mae > 0.02:
                kind = "high_edge_mae"
            else:
                kind = "weak_or_aligned"
            cuts.append(
                {
                    "pair": f"y{Y0+j:02d}_x{X0+i:02d}|x{X0+i+1:02d}",
                    "edge_mae": mae,
                    "delta_mean": dmu,
                    "phase_dy_px": dy,
                    "phase_dx_px": dx,
                    "kind": kind,
                    "left_base": tiles[j * N + i]["base"],
                    "right_base": tiles[j * N + i + 1]["base"],
                    "interior_hp": _hp_energy(left[:, pad:-pad])
                    if left.shape[1] > 2 * pad
                    else None,
                    "boundary_hp": _hp_energy(l),
                    "right_boundary_hp": _hp_energy(r),
                }
            )
    for j in range(N - 1):
        for i in range(N):
            top = rgbs[j * N + i]
            bot = rgbs[(j + 1) * N + i]
            mae = float(np.abs(top[-1, :, :] - bot[0, :, :]).mean())
            dmu = float(top[-1, :, :].mean() - bot[0, :, :].mean())
            dy, dx = _phase_shift(top[-pad:], bot[:pad])
            if abs(dmu) > 0.008:
                kind = "colour_jump"
            elif mae > 0.02:
                kind = "high_edge_mae"
            else:
                kind = "weak_or_aligned"
            cuts.append(
                {
                    "pair": f"y{Y0+j:02d}|y{Y0+j+1:02d}_x{X0+i:02d}",
                    "edge_mae": mae,
                    "delta_mean": dmu,
                    "phase_dy_px": dy,
                    "phase_dx_px": dx,
                    "kind": kind,
                    "left_base": tiles[j * N + i]["base"],
                    "right_base": tiles[(j + 1) * N + i]["base"],
                }
            )

    stack = np.concatenate([r.reshape(-1, 3) for r in rgbs], axis=0)
    lo, hi = np.percentile(stack, [2, 98], axis=0)
    fig, axes = plt.subplots(N, N, figsize=(10.5, 10.5))
    for rec, rgb, ax in zip(tiles, rgbs, axes.ravel()):
        ax.imshow(_fixed_stretch(rgb, lo, hi, gamma=1.0), interpolation="nearest")
        b = rec["base"] or {}
        ax.set_title(
            f"{rec['tile_id'][-8:]}\n{b.get('date', '?')}  {b.get('cloud_pct', '?')}%",
            fontsize=7,
        )
        ax.set_axis_off()
    fig.suptitle(
        "32VNM 3×3  ·  one shared 2–98% stretch, gamma=1  ·  GeoTIFF is not per-tile display-normed",
        fontsize=11,
    )
    fig.tight_layout()
    png = OUT / f"{stem}_fixed_stretch.png"
    fig.savefig(png, dpi=120)
    plt.close(fig)

    consensus_2 = _consensus_date(cell_dates, max_cloud=0.02)
    consensus_5 = _consensus_date(cell_dates, max_cloud=0.05)
    subset = write_subset_manifest()
    otb = shutil.which("otbcli_Mosaic")

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "block": {"y0": Y0, "x0": X0, "n": N},
        "export_normalization": {
            "geotiff_percentile_stretch": False,
            "geotiff_gamma": False,
            "unstandardize": "per-tile frozen base-frame mean/std (optimize.py)",
            "clip_01": True,
            "conclusion": (
                "GeoTIFFs are destandardized reflectance, clipped to [0,1]. "
                "There is no per-tile percentile or gamma in export. Tile DC "
                "follows the identity LR frame of that cell."
            ),
        },
        "geo": {
            "crs_identical": crs_ok,
            "gsd_2p5m": gsd_ok,
            "seams": gaps,
            "all_aligned": all(g.get("aligned") for g in gaps),
        },
        "base_dates": dict(Counter((t["base"] or {}).get("date") for t in tiles)),
        "cell_dates": cell_dates,
        "consensus": {"cloud_2pct": consensus_2, "cloud_5pct": consensus_5},
        "cuts": cuts,
        "tiles": tiles,
        "png": str(png.relative_to(ROOT)),
        "stretch_lo": lo.tolist(),
        "stretch_hi": hi.tolist(),
        "subset_manifest": str(subset.relative_to(ROOT)),
        "otbcli_Mosaic": otb,
        "otb_note": (
            "Orfeo ToolBox Mosaic not installed; feather/Laplacian implemented here."
            if not otb
            else f"OTB Mosaic at {otb} (not used for this pilot)."
        ),
        "orthophotos": "unused (reserved for evaluation; this 3x3 is --allow_no_hr)",
    }
    (OUT / f"{stem}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in ("geo", "base_dates", "export_normalization", "consensus")
            },
            indent=2,
        )
    )
    print("cuts:")
    for c in cuts:
        lb = (c.get("left_base") or {}).get("date")
        rb = (c.get("right_base") or {}).get("date")
        print(
            f"  {c['pair']:24} {c['kind']:16} MAE={c['edge_mae']:.4f} "
            f"Δμ={c['delta_mean']:+.4f} shift=({c['phase_dy_px']:+.1f},"
            f"{c['phase_dx_px']:+.1f})  {lb} | {rb}"
        )
    print("wrote", png)
    return report


def _union_dates_for_window(plan_by_xy: dict, row0: int, col0: int, side: int) -> list[str]:
    dates: set[str] = set()
    r1, c1 = row0 + side - 1, col0 + side - 1
    for rec in plan_by_xy.values():
        rr, cc = int(rec["row_off"]), int(rec["col_off"])
        if rr + SIDE - 1 < row0 or rr > r1 or cc + SIDE - 1 < col0 or cc > c1:
            continue
        dates.update(rec.get("dates") or [])
    ranked = sorted(
        dates, key=lambda d: (abs((date.fromisoformat(d) - CENTER).days), d)
    )
    return ranked[:16]


def tile_overlap(
    stride: int,
    *,
    force: bool = False,
    y0: int | None = None,
    x0: int | None = None,
    out_dir: Path | None = None,
    tag: str | None = None,
) -> Path:
    overlap = SIDE - int(stride)
    ovl_tag = f"ovl{overlap}"
    origin_y = Y0 if y0 is None else int(y0)
    origin_x = X0 if x0 is None else int(x0)
    destination_root = OUT if out_dir is None else out_dir
    name_suffix = "" if tag is None else f"_{tag}"
    parent_meta = json.loads((S2_PARENT / "meta.json").read_text())
    plan = json.loads(PLAN_ONDISK.read_text())
    plan_by_xy = {(c["iy"], c["ix"]): c for c in plan["cells"]}
    out_root = ROOT / "data/s2_revisits/national_2025_v2"
    tiles = []
    row_base, col_base = origin_y * SIDE, origin_x * SIDE
    t0 = time.time()
    unique_area = (SIDE + (N - 1) * stride) ** 2
    tile_area = N * N * SIDE * SIDE
    for j in range(N):
        for i in range(N):
            row0 = row_base + j * stride
            col0 = col_base + i * stride
            dest = (
                out_root
                / f"32VNM_t512_{ovl_tag}{name_suffix}_y{j:02d}_x{i:02d}"
            )
            dates = _union_dates_for_window(plan_by_xy, row0, col0, SIDE)
            frames = filter_frames_for_cell(parent_meta.get("frames") or [], dates)
            rec = {"iy": j, "ix": i, "dates": dates, "n_frames": len(frames)}
            row = _write_tile(
                S2_PARENT,
                parent_meta,
                dest,
                row0=row0,
                col0=col0,
                side=SIDE,
                iy=j,
                ix=i,
                force=force,
                stride=stride,
                overlap_frac=overlap / SIDE,
                frames=frames,
                cell_rec=rec,
            )
            tiles.append(row)
            print(
                f"{dest.name} row={row0} col={col0} n_frames={len(frames)}",
                flush=True,
            )
    man = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "parent": PARENT,
        "side": SIDE,
        "stride": stride,
        "overlap_px": overlap,
        "overlap_frac": overlap / SIDE,
        "pilot": "3x3",
        "origin_lr": {
            "row_off": row_base,
            "col_off": col_base,
            "y0": origin_y,
            "x0": origin_x,
        },
        "tiles": tiles,
        "tile_seconds": time.time() - t0,
        "merge_method_hint": "feather",
        "coverage_lr_px": SIDE + (N - 1) * stride,
        "coverage_km": (SIDE + (N - 1) * stride) * 0.01,
        "pixel_overhead": {
            "unique_lr_px2": unique_area,
            "reconstructed_lr_px2": tile_area,
            "extra_frac": tile_area / unique_area - 1.0,
        },
    }
    man_path = destination_root / f"manifest_{ovl_tag}{name_suffix}.json"
    man_path.parent.mkdir(parents=True, exist_ok=True)
    man_path.write_text(json.dumps(man, indent=2) + "\n")
    print(
        f"Wrote {man_path}  coverage {man['coverage_lr_px']} LR px "
        f"({man['coverage_km']:.2f} km)  extra {man['pixel_overhead']['extra_frac']*100:.1f}%"
    )
    return man_path


def _overlap_interior(h: int, w: int, margin: int) -> np.ndarray:
    m = np.ones((h, w), dtype=bool)
    m[:margin, :] = False
    m[-margin:, :] = False
    m[:, :margin] = False
    m[:, -margin:] = False
    return m


def _intersect_arrays(
    path_a: Path, path_b: Path
) -> tuple[np.ndarray, np.ndarray] | None:
    import rasterio
    from rasterio.windows import from_bounds as win_from_bounds

    with rasterio.open(path_a) as sa, rasterio.open(path_b) as sb:
        ia, ja = sa.bounds, sb.bounds
        left, right = max(ia.left, ja.left), min(ia.right, ja.right)
        bottom, top = max(ia.bottom, ja.bottom), min(ia.top, ja.top)
        if right - left < 8 * abs(sa.transform.a) or top - bottom < 8 * abs(sa.transform.e):
            return None
        wa = sa.read(
            indexes=(1, 2, 3),
            window=win_from_bounds(left, bottom, right, top, sa.transform),
            boundless=True,
            fill_value=0.0,
        )
        wb = sb.read(
            indexes=(1, 2, 3),
            window=win_from_bounds(left, bottom, right, top, sb.transform),
            boundless=True,
            fill_value=0.0,
        )
    a = np.transpose(wa, (1, 2, 0)).astype(np.float32)
    b = np.transpose(wb, (1, 2, 0)).astype(np.float32)
    h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
    return a[:h, :w], b[:h, :w]


def solve_gains_from_lr(
    lr_paths: list[Path],
    *,
    lf_k: int = 16,
    margin: int = 8,
    change_thr: float = 0.25,
    lam_a: float = 8.0,
    lam_b: float = 30.0,
    max_samples: int = 800,
) -> dict:
    """Joint per-tile per-band gain/offset from LF LR overlap. Tile 0 is the anchor."""
    n = len(lr_paths)
    rows: list[np.ndarray] = []
    rhs: list[float] = []
    pair_stats = []
    n_params = 6 * max(n - 1, 0)

    def col(t: int, kind: int) -> int:
        return 6 * (t - 1) + kind

    for i in range(n):
        for j in range(i + 1, n):
            got = _intersect_arrays(lr_paths[i], lr_paths[j])
            if got is None:
                pair_stats.append({"i": i, "j": j, "n": 0, "skipped": True, "reason": "no_overlap"})
                continue
            ai, aj = got
            h, w = ai.shape[:2]
            valid = (ai.sum(2) > 1e-5) & (aj.sum(2) > 1e-5) & _overlap_interior(h, w, margin)
            delta = np.abs(ai - aj).mean(axis=2)
            valid &= delta < change_thr
            n_valid = int(valid.sum())
            if n_valid < 64:
                pair_stats.append(
                    {
                        "i": i,
                        "j": j,
                        "n": n_valid,
                        "skipped": True,
                        "reason": "too_few_pixels",
                    }
                )
                continue
            fi = _box_down(ai, lf_k)
            fj = _box_down(aj, lf_k)
            ys, xs = np.where(valid)
            pick = np.linspace(0, len(ys) - 1, num=min(max_samples, len(ys)), dtype=int)
            ys, xs = ys[pick], xs[pick]
            mae = float(delta[valid].mean())
            # Pair-mean LF observation (the date-island colour step). Weighted
            # so a systematic ~0.02 DC is not washed out by per-pixel noise
            # plus identity regularization.
            mean_i = fi[valid].mean(axis=0)
            mean_j = fj[valid].mean(axis=0)
            w_mean = 400.0
            for c in range(3):
                row = np.zeros(n_params, dtype=np.float64)
                bval = 0.0
                xi_v, xj_v = float(mean_i[c]) * w_mean, float(mean_j[c]) * w_mean
                if i == 0:
                    bval -= xi_v
                else:
                    row[col(i, c)] += xi_v
                    row[col(i, 3 + c)] += w_mean
                if j == 0:
                    bval += xj_v
                else:
                    row[col(j, c)] -= xj_v
                    row[col(j, 3 + c)] -= w_mean
                rows.append(row)
                rhs.append(bval)
            for c in range(3):
                for xi_v, xj_v in zip(fi[ys, xs, c].tolist(), fj[ys, xs, c].tolist()):
                    row = np.zeros(n_params, dtype=np.float64)
                    bval = 0.0
                    if i == 0:
                        bval -= xi_v
                    else:
                        row[col(i, c)] += xi_v
                        row[col(i, 3 + c)] += 1.0
                    if j == 0:
                        bval += xj_v
                    else:
                        row[col(j, c)] -= xj_v
                        row[col(j, 3 + c)] -= 1.0
                    rows.append(row)
                    rhs.append(bval)
            pair_stats.append(
                {"i": i, "j": j, "n": n_valid, "skipped": False, "mae_lr": mae}
            )

    if n_params == 0:
        return {
            "gains": [{"a": [1.0, 1.0, 1.0], "b": [0.0, 0.0, 0.0]}],
            "pairs": pair_stats,
            "n_eq": 0,
            "anchor": 0,
            "residual_rmse": 0.0,
        }

    extra, extra_b = [], []
    for t in range(1, n):
        for c in range(3):
            r = np.zeros(n_params)
            r[col(t, c)] = lam_a
            extra.append(r)
            extra_b.append(lam_a * 1.0)
            r2 = np.zeros(n_params)
            r2[col(t, 3 + c)] = lam_b
            extra.append(r2)
            extra_b.append(0.0)

    if not rows:
        x = np.zeros(n_params)
        residual_rmse = None
        n_eq = 0
    else:
        A = np.vstack([np.stack(rows), np.stack(extra)])
        b = np.concatenate([np.asarray(rhs, dtype=np.float64), extra_b])
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
        pred = np.stack(rows) @ x
        residual_rmse = float(np.sqrt(np.mean((pred - np.asarray(rhs)) ** 2)))
        n_eq = len(rhs)

    gains = [{"a": [1.0, 1.0, 1.0], "b": [0.0, 0.0, 0.0]}]
    for t in range(1, n):
        sl = x[6 * (t - 1) : 6 * t]
        gains.append({"a": sl[:3].tolist(), "b": sl[3:].tolist()})
    failures = [p for p in pair_stats if p.get("skipped")]
    return {
        "gains": gains,
        "pairs": pair_stats,
        "n_eq": n_eq,
        "anchor": 0,
        "residual_rmse": residual_rmse,
        "failures": failures,
        "lam_a": lam_a,
        "lam_b": lam_b,
        "lf_k": lf_k,
        "change_thr": change_thr,
        "fit_on": "s2_bilinear low-frequency overlap (not SR histograms)",
    }


def apply_gain(rgb: np.ndarray, g: dict) -> np.ndarray:
    a = np.asarray(g["a"], dtype=np.float32).reshape(1, 1, 3)
    b = np.asarray(g["b"], dtype=np.float32).reshape(1, 1, 3)
    return np.clip(rgb * a + b, 0.0, 1.0)


def _canvas_meta(paths: list[Path]):
    import rasterio
    from rasterio.transform import from_bounds

    bounds = None
    res = None
    crs = None
    count = 3
    for p in paths:
        with rasterio.open(p) as src:
            b = src.bounds
            res = src.res
            crs = src.crs
            count = src.count
            if bounds is None:
                bounds = [b.left, b.bottom, b.right, b.top]
            else:
                bounds = [
                    min(bounds[0], b.left),
                    min(bounds[1], b.bottom),
                    max(bounds[2], b.right),
                    max(bounds[3], b.top),
                ]
    width = int(round((bounds[2] - bounds[0]) / res[0]))
    height = int(round((bounds[3] - bounds[1]) / abs(res[1])))
    transform = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], width, height)
    return height, width, transform, crs, count


def _place_tile(path: Path, transform, height: int, width: int, gain=None, *, feather_px: float = 128.0):
    import rasterio
    from rasterio.transform import rowcol

    with rasterio.open(path) as src:
        data = np.transpose(src.read(indexes=(1, 2, 3)), (1, 2, 0)).astype(np.float32)
        h, w = data.shape[:2]
        r0, c0 = rowcol(transform, src.bounds.left, src.bounds.top)
        r0, c0 = int(r0), int(c0)
    if gain is not None:
        data = apply_gain(data, gain)
    valid = data.sum(2) > 1e-6
    weight = _edge_distance_weights(h, w, feather_px=float(feather_px))
    weight = np.where(valid, weight, 0.0).astype(np.float32)
    return data, weight, r0, c0


def _paste(canvas, weight_acc, data, weight, r0, c0, *, method: str, seen=None):
    h, w = data.shape[:2]
    sl = canvas[r0 : r0 + h, c0 : c0 + w]
    ww = weight_acc[r0 : r0 + h, c0 : c0 + w]
    if method == "first":
        write = (weight > 0) & (~seen[r0 : r0 + h, c0 : c0 + w])
        sl[write] = data[write]
        seen[r0 : r0 + h, c0 : c0 + w] |= weight > 0
        return
    sl += data * weight[..., None]
    ww += weight


def _weighted_pyr_down(img: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    num = cv2.pyrDown(img * w[..., None])
    den = cv2.pyrDown(w)
    out = np.zeros_like(num)
    m = den > 1e-8
    out[m] = num[m] / den[m, None]
    return out, den


def _laplacian_reconstruct(placed, weights, *, levels: int = 4) -> np.ndarray:
    """Multiband blend: weighted Gaussian pyramid, Laplacian high-pass, reconstruct.

    Weighted downsample avoids darkening at tile footprints where the canvas is 0.
    """
    import cv2

    gaussians = []  # list of (imgs, weights) per level, coarse last
    cur_imgs = [img.astype(np.float32) for img in placed]
    cur_w = [w.astype(np.float32) for w in weights]
    gaussians.append((cur_imgs, cur_w))
    for _ in range(levels):
        nxt_i, nxt_w = [], []
        for img, w in zip(cur_imgs, cur_w):
            di, dw = _weighted_pyr_down(img, w)
            nxt_i.append(di)
            nxt_w.append(dw)
        gaussians.append((nxt_i, nxt_w))
        cur_imgs, cur_w = nxt_i, nxt_w

    def _blend(imgs, ws):
        acc = np.zeros_like(imgs[0])
        wsum = np.zeros(imgs[0].shape[:2], np.float32)
        for img, w in zip(imgs, ws):
            acc += img * w[..., None]
            wsum += w
        out = np.zeros_like(acc)
        m = wsum > 1e-8
        out[m] = acc[m] / wsum[m, None]
        return out

    out = _blend(*gaussians[-1])
    for lvl in range(len(gaussians) - 2, -1, -1):
        imgs, ws = gaussians[lvl]
        parent_i, parent_w = gaussians[lvl + 1]
        target = (imgs[0].shape[1], imgs[0].shape[0])
        up = cv2.pyrUp(out, dstsize=target)
        # Laplacian of each tile: G_l - upsample(G_{l+1})
        hps = []
        for img, pimg in zip(imgs, parent_i):
            up_i = cv2.pyrUp(pimg, dstsize=(img.shape[1], img.shape[0]))
            hps.append(img - up_i)
        # Finest Laplacian: winner-take-all from the highest-weight tile so
        # fine detail is not averaged into double edges / blur.
        if lvl == 0:
            stacked_w = np.stack(ws, axis=0)
            winner = np.argmax(stacked_w, axis=0)
            hp = np.zeros_like(hps[0])
            for t, tile_hp in enumerate(hps):
                sel = winner == t
                hp[sel] = tile_hp[sel]
        else:
            hp = _blend(hps, ws)
        out = np.clip(up + hp, 0.0, 1.0)
    return out


def mosaic_from_paths(
    paths: list[Path],
    *,
    method: str,
    gains=None,
    feather_px: float = 128.0,
) -> tuple[np.ndarray, object, object]:
    height, width, transform, crs, _count = _canvas_meta(paths)
    if method == "multiband":
        placed, weights = [], []
        for ti, p in enumerate(paths):
            canvas = np.zeros((height, width, 3), np.float32)
            wcanvas = np.zeros((height, width), np.float32)
            data, weight, r0, c0 = _place_tile(
                p, transform, height, width, None if gains is None else gains[ti],
                feather_px=feather_px,
            )
            h, w = data.shape[:2]
            canvas[r0 : r0 + h, c0 : c0 + w] = data
            wcanvas[r0 : r0 + h, c0 : c0 + w] = weight
            placed.append(canvas)
            weights.append(wcanvas)
        out = _laplacian_reconstruct(placed, weights, levels=4)
        return out, transform, crs

    acc = np.zeros((height, width, 3), np.float32)
    wacc = np.zeros((height, width), np.float32)
    seen = np.zeros((height, width), dtype=bool)
    for ti, p in enumerate(paths):
        data, weight, r0, c0 = _place_tile(
            p, transform, height, width, None if gains is None else gains[ti],
            feather_px=feather_px,
        )
        if method == "first":
            _paste(acc, wacc, data, weight, r0, c0, method="first", seen=seen)
        else:
            _paste(acc, wacc, data, weight, r0, c0, method="feather")
    if method == "first":
        return acc, transform, crs
    out = np.zeros_like(acc)
    m = wacc > 0
    out[m] = acc[m] / wacc[m, None]
    return out, transform, crs


def write_png(rgb, lo, hi, path: Path, title: str, *, crops=None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = 1 + (0 if not crops else len(crops))
    fig, axes = plt.subplots(1, n, figsize=(4.8 * n, 4.8))
    if n == 1:
        axes = [axes]
    axes[0].imshow(_fixed_stretch(rgb, lo, hi), interpolation="bilinear")
    axes[0].set_title(title, fontsize=9)
    axes[0].set_axis_off()
    if crops:
        for ax, (name, sl) in zip(axes[1:], crops):
            ax.imshow(_fixed_stretch(rgb[sl], lo, hi), interpolation="nearest")
            ax.set_title(name, fontsize=8)
            ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _sharpness(rgb: np.ndarray) -> float:
    import cv2

    g = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(
        np.float32
    )
    return float(cv2.Laplacian(g, cv2.CV_32F).var())


def _lr_consistency(sr: np.ndarray, bilinear: np.ndarray) -> dict:
    """SR vs identity-LR bilinear on the same canvas (not NIB)."""
    valid = (sr.sum(2) > 1e-5) & (bilinear.sum(2) > 1e-5)
    if not valid.any():
        return {"mae": None, "n": 0}
    return {
        "mae": float(np.abs(sr[valid] - bilinear[valid]).mean()),
        "n": int(valid.sum()),
    }


def blend_stage(prefix: str, man_path: Path) -> dict:
    import rasterio

    man = json.loads(man_path.read_text())
    sr_paths, lr_paths = [], []
    for t in man["tiles"]:
        sr = _sr_geotiff_path(man["parent"], t["tile_id"], prefix)
        lr = _qgis(man["parent"], t["tile_id"], prefix, "s2_bilinear.tif")
        if not sr.is_file():
            raise SystemExit(f"missing {sr}")
        sr_paths.append(sr)
        lr_paths.append(lr)

    overlap_hr = int(man.get("overlap_px") or 32) * DF
    t0 = time.time()
    hard, transform, crs = mosaic_from_paths(sr_paths, method="first", feather_px=overlap_hr)
    t_hard = time.time() - t0
    t0 = time.time()
    feather, _, _ = mosaic_from_paths(sr_paths, method="feather", feather_px=overlap_hr)
    t_feather = time.time() - t0
    t0 = time.time()
    solved = solve_gains_from_lr(lr_paths)
    t_solve = time.time() - t0
    t0 = time.time()
    col_feather, _, _ = mosaic_from_paths(
        sr_paths, method="feather", gains=solved["gains"], feather_px=overlap_hr
    )
    t_cf = time.time() - t0
    t0 = time.time()
    col_mb, _, _ = mosaic_from_paths(
        sr_paths, method="multiband", gains=solved["gains"], feather_px=overlap_hr
    )
    t_mb = time.time() - t0
    t0 = time.time()
    lr_mosaic, _, _ = mosaic_from_paths(lr_paths, method="feather", feather_px=overlap_hr)
    t_lr = time.time() - t0

    overlap_mae = []
    for i in range(len(sr_paths)):
        for j in range(i + 1, len(sr_paths)):
            pair = _intersect_arrays(sr_paths[i], sr_paths[j])
            if pair is None:
                continue
            a, b = pair
            h, w = a.shape[:2]
            m = _overlap_interior(h, w, 8) & (a.sum(2) > 1e-5) & (b.sum(2) > 1e-5)
            if int(m.sum()) < 64:
                continue
            mae0 = float(np.abs(a[m] - b[m]).mean())
            a2 = apply_gain(a, solved["gains"][i])
            b2 = apply_gain(b, solved["gains"][j])
            mae1 = float(np.abs(a2[m] - b2[m]).mean())
            dy, dx = _phase_shift(a, b)
            overlap_mae.append(
                {
                    "pair": f"{i}|{j}",
                    "mae_sr_before": mae0,
                    "mae_sr_after": mae1,
                    "phase_dy_px": dy,
                    "phase_dx_px": dx,
                }
            )

    stack = np.concatenate(
        [x.reshape(-1, 3) for x in (hard, feather, col_feather, col_mb)], axis=0
    )
    ok = stack.sum(1) > 1e-5
    lo, hi = np.percentile(stack[ok], [2, 98], axis=0)

    h, w = hard.shape[:2]
    # Difficult: east-central seam (original y04_x17|x18 neighborhood).
    # Successful: west-central, usually same July 12 blob.
    crops = [
        ("east seam (hard date cut)", (slice(h // 3, 2 * h // 3), slice(2 * w // 3 - 80, 2 * w // 3 + 80))),
        ("west interior (same-date)", (slice(h // 3, 2 * h // 3), slice(w // 3 - 80, w // 3 + 80))),
    ]
    names = [
        ("01_hard.png", hard, f"hard cuts (first)  {t_hard:.1f}s"),
        ("02_feather.png", feather, f"feather only  {t_feather:.1f}s"),
        ("03_colour_feather.png", col_feather, f"LR colour + feather  solve {t_solve:.1f}s"),
        ("04_colour_multiband.png", col_mb, "LR colour + Laplacian blend"),
        ("00_lr_bilinear.png", lr_mosaic, "identity-LR bilinear mosaic (reference, not NIB)"),
    ]
    for fn, arr, title in names:
        write_png(arr, lo, hi, OUT / f"{prefix}_{fn}", title, crops=crops)

    def _write_tif(arr, path: Path):
        profile = {
            "driver": "GTiff",
            "height": arr.shape[0],
            "width": arr.shape[1],
            "count": 3,
            "dtype": "float32",
            "crs": crs,
            "transform": transform,
            "nodata": 0.0,
            "compress": "deflate",
            "tiled": True,
        }
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(np.transpose(arr, (2, 0, 1)))

    _write_tif(hard, OUT / f"{prefix}_hard.tif")
    _write_tif(feather, OUT / f"{prefix}_feather.tif")
    _write_tif(col_feather, OUT / f"{prefix}_colour_feather.tif")
    _write_tif(col_mb, OUT / f"{prefix}_colour_multiband.tif")

    report = {
        "prefix": prefix,
        "manifest": str(man_path.relative_to(ROOT)),
        "times_s": {
            "hard": t_hard,
            "feather": t_feather,
            "solve": t_solve,
            "colour_feather": t_cf,
            "colour_multiband": t_mb,
            "lr_mosaic": t_lr,
        },
        "gains": solved,
        "overlap_mae": overlap_mae,
        "sharpness": {
            "hard": _sharpness(hard),
            "feather": _sharpness(feather),
            "colour_feather": _sharpness(col_feather),
            "colour_multiband": _sharpness(col_mb),
        },
        "lr_consistency_mae": {
            "hard": _lr_consistency(hard, lr_mosaic),
            "feather": _lr_consistency(feather, lr_mosaic),
            "colour_feather": _lr_consistency(col_feather, lr_mosaic),
            "colour_multiband": _lr_consistency(col_mb, lr_mosaic),
        },
        "coverage_km": man.get("coverage_km"),
        "overlap_px": man.get("overlap_px"),
        "pixel_overhead": man.get("pixel_overhead"),
        "n_tiles": len(sr_paths),
        "hr_reference": None,
        "hr_note": "These national cells were trained with --allow_no_hr; NIB is not used.",
        "otb": shutil.which("otbcli_Mosaic"),
    }
    (OUT / f"blend_{prefix}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("times_s", "overlap_mae", "sharpness")}, indent=2))
    return report


def write_report() -> Path:
    diag_p = OUT / "diagnose.json"
    diag = json.loads(diag_p.read_text()) if diag_p.is_file() else {}
    blends = []
    for p in sorted(OUT.glob("blend_*.json")):
        blends.append(json.loads(p.read_text()))
    trains = []
    for p in sorted((ROOT / "single_samples/sweep_results").glob("seam_pilot_*.json")):
        trains.append(json.loads(p.read_text()))

    lines = [
        "# 32VNM 3×3 seam pilot",
        "",
        f"Block: original LR512 cells y{Y0:02d}–y{Y0+N-1:02d} × x{X0:02d}–x{X0+N-1:02d}.",
        "Display stretch is shared (2–98%, gamma=1) across all tiles; GeoTIFF export is destandardize+clip, not per-tile histogram matching.",
        "",
        "## Export, CRS, grid",
        "",
        json.dumps(diag.get("export_normalization"), indent=2),
        "",
        json.dumps(diag.get("geo"), indent=2),
        "",
        "## Base acquisitions",
        "",
        json.dumps(diag.get("base_dates"), indent=2),
        "",
        json.dumps(diag.get("consensus"), indent=2),
        "",
        "## Colour vs geometry (0-overlap base2)",
        "",
    ]
    for c in diag.get("cuts") or []:
        lines.append(
            f"- `{c['pair']}` **{c['kind']}** MAE={c['edge_mae']:.4f} Δμ={c['delta_mean']:+.4f} "
            f"shift=({c.get('phase_dy_px', 0):+.1f},{c.get('phase_dx_px', 0):+.1f}) px "
            f"{(c.get('left_base') or {}).get('date')} | {(c.get('right_base') or {}).get('date')}"
        )
    lines += ["", "## Overlap reconstructions + blend", ""]
    for b in blends:
        lines.append(f"### {b.get('prefix')}")
        lines.append(f"- overlap {b.get('overlap_px')} LR px, coverage {b.get('coverage_km')} km")
        lines.append(f"- pixel overhead {b.get('pixel_overhead')}")
        lines.append(f"- times (s) {b.get('times_s')}")
        lines.append(f"- sharpness {b.get('sharpness')}")
        lines.append(f"- LR consistency {b.get('lr_consistency_mae')}")
        for row in b.get("overlap_mae") or []:
            lines.append(
                f"- overlap {row['pair']}: SR MAE {row['mae_sr_before']:.4f} → {row['mae_sr_after']:.4f} "
                f"shift ({row['phase_dy_px']:+.1f},{row['phase_dx_px']:+.1f})"
            )
        g = (b.get("gains") or {}).get("gains") or []
        for i, gi in enumerate(g):
            lines.append(f"- tile {i} gain {gi}")
        fails = (b.get("gains") or {}).get("failures") or []
        if fails:
            lines.append(f"- fit failures: {fails}")
        lines.append("")
    lines += ["## Training cost", ""]
    for t in trains:
        agg = t.get("aggregate") or {}
        lines.append(
            f"- `{t.get('run_prefix')}` force_base={t.get('force_base_date')} "
            f"ok={agg.get('n_ok')} fail={agg.get('n_fail')} "
            f"mean_train_s={agg.get('mean_train_s')} peak listed per tile in JSON"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- Common-base identity is a **separate** reconstruction run from overlap blending.",
        "- Colour gains are estimated from **LR bilinear** overlap (low-pass), then applied to SR.",
        "- Independent full-tile histogram matching was not used.",
        "- NIB orthophotos were not used ( `--allow_no_hr` ).",
        f"- {diag.get('otb_note')}",
        "",
    ]
    path = OUT / "REPORT.md"
    path.write_text("\n".join(lines) + "\n")
    print("wrote", path)
    return path


def main() -> None:
    global OUT

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--stage",
        choices=["diagnose", "tile", "blend", "report", "all"],
        default="diagnose",
    )
    ap.add_argument("--stride", type=int, default=480, help="LR stride (480=ovl32, 448=ovl64).")
    ap.add_argument("--prefix", default="prod_k4_ovl32")
    ap.add_argument("--diagnose-prefix", default="prod_k4_base2")
    ap.add_argument("--diagnose-stem", default="diagnose")
    ap.add_argument("--y0", type=int, default=Y0)
    ap.add_argument("--x0", type=int, default=X0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    OUT = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    if args.stage in ("diagnose", "all"):
        diagnose(prefix=args.diagnose_prefix, stem=args.diagnose_stem)
    if args.stage in ("tile", "all"):
        tile_overlap(
            args.stride,
            force=args.force,
            y0=args.y0,
            x0=args.x0,
            out_dir=OUT,
            tag=args.tag,
        )
    if args.stage == "blend":
        ovl = SIDE - args.stride
        suffix = "" if args.tag is None else f"_{args.tag}"
        blend_stage(args.prefix, OUT / f"manifest_ovl{ovl}{suffix}.json")
    if args.stage in ("report", "all"):
        write_report()


if __name__ == "__main__":
    main()
