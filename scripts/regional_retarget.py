#!/usr/bin/env python3
"""Affine output retargeting of independently fitted LR512 cells to a regional reference.

ScaleF predicts frame t of cell i as D[a_it * F_i + b_it] per band (frame 0 is the identity) and
exports F_i with the base frame's statistics. Replacing F_i by g_i F_i + h_i while setting
a_it' = a_it / g_i and b_it' = b_it - a_it h_i / g_i leaves every predicted LR observation
unchanged, so a per-cell, per-band affine can move each cell into a shared radiometric convention
without retraining. This script measures how much of the between-cell seam that removes.

Stages (all on one MGRS granule, outputs under ``--out``):
  reference  Regional reference C and confidence q on a granule-aligned block grid (``--block``
             LR px). Each frame is block-averaged over SCL-clear pixels; each date is calibrated
             per band to the across-date median with a trimmed robust affine fit; C is the median
             of the calibrated dates, q falls with few observations and with their spread.
  summarize  Per cell and output prefix: block means of the rendered field and thin strips at the
             four edges and at interior control lines, enough to score seams exactly under any
             per-band affine.
  fit        Per cell and band, the q-weighted least-squares gain and offset mapping the cell's
             block means to C (and an offset-only variant).
  evaluate   Seam discontinuities between abutting cells (coarse strip-mean offsets and 1-px
             steps) relative to the same statistics on interior control lines, split into
             same-base-date and different-base-date boundaries; interior change; and base-frame
             sensitivity on cells fitted with two base dates.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
CLEAR_SCL = (4, 5, 6)
HR_PER_LR = 4
CELL_LR = 512
CELL_HR = CELL_LR * HR_PER_LR
STRIP = 16          # HR px averaged next to a boundary (40 m)
REF_DEPTH = 256     # HR px from the edge of the interior control line (640 m)
SEG = 32            # HR px per boundary segment (80 m)
SIDES = ("N", "S", "W", "E")


def granule_frames(s2_dir: Path) -> list[dict]:
    return json.loads((s2_dir / "meta.json").read_text())["frames"]


def block_reduce_frame(args) -> tuple[str, np.ndarray, np.ndarray]:
    s2_dir, fr, block, nb = args
    with rasterio.open(s2_dir / fr["path"]) as src:
        rgb = src.read([1, 2, 3]).astype(np.float32)
    with rasterio.open(s2_dir / fr["scl_path"]) as src:
        scl = src.read(1)
    n = nb * block
    rgb, scl = rgb[:, :n, :n], scl[:n, :n]
    clear = np.isin(scl, CLEAR_SCL) & np.all(rgb > 0, axis=0)
    refl = np.clip(rgb / 10000.0, 0.0, 1.0)
    c = clear.reshape(nb, block, nb, block).sum(axis=(1, 3)).astype(np.float32)
    s = (refl * clear).reshape(3, nb, block, nb, block).sum(axis=(2, 4))
    frac = c / float(block * block)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s / c
    return fr["path"], mean.astype(np.float32), frac


def robust_affine(y: np.ndarray, x: np.ndarray, iters: int = 4, k: float = 2.5) -> tuple[float, float, float]:
    """Fit y ~ a x + b, trimming residuals beyond k robust SDs; returns (a, b, kept fraction)."""
    keep = np.isfinite(x) & np.isfinite(y)
    idx = np.flatnonzero(keep)
    a, b = 1.0, 0.0
    sel = idx
    for _ in range(iters):
        if sel.size < 50:
            break
        A = np.stack([x[sel], np.ones(sel.size)], axis=1)
        a, b = np.linalg.lstsq(A, y[sel], rcond=None)[0]
        r = y[idx] - (a * x[idx] + b)
        s = 1.4826 * np.median(np.abs(r - np.median(r)))
        sel = idx[np.abs(r) <= k * max(s, 1e-4)]
    return float(a), float(b), float(sel.size / max(idx.size, 1))


def stage_reference(args) -> None:
    s2_dir = ROOT / args.s2_dir
    frames = granule_frames(s2_dir)
    nb = 10980 // args.block
    jobs = [(s2_dir, fr, args.block, nb) for fr in frames]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        res = list(ex.map(block_reduce_frame, jobs))
    paths = [r[0] for r in res]
    means = np.stack([r[1] for r in res])            # T,3,nb,nb
    fracs = np.stack([r[2] for r in res])            # T,nb,nb
    means[np.broadcast_to((fracs < args.min_clear)[:, None], means.shape)] = np.nan

    med0 = np.nanmedian(means, axis=0)
    calib = []
    norm = np.empty_like(means)
    for t in range(means.shape[0]):
        row = {"frame": paths[t], "clear_blocks": float(np.isfinite(means[t, 0]).mean())}
        for b in range(3):
            y, x = means[t, b].ravel(), med0[b].ravel()
            if np.isfinite(y).sum() < args.min_blocks:
                a, c, kept = np.nan, np.nan, 0.0
                norm[t, b] = np.nan
            else:
                a, c, kept = robust_affine(y, x)
                norm[t, b] = (means[t, b] - c) / a
            row[f"band{b}"] = {"gain": a, "offset": c, "kept": kept}
        calib.append(row)
    C = np.nanmedian(norm, axis=0)
    n_obs = np.isfinite(norm[:, 0]).sum(axis=0)
    spread = np.nanmedian(np.abs(norm - C[None]), axis=0).mean(axis=0) * 1.4826
    q = np.where(n_obs >= args.min_obs, np.minimum(1.0, n_obs / 6.0) / (1.0 + (spread / args.spread_scale) ** 2), 0.0)
    q = np.nan_to_num(q).astype(np.float32)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "reference.npz", C=C.astype(np.float32), q=q, n_obs=n_obs.astype(np.int16),
                        spread=spread.astype(np.float32), med0=med0.astype(np.float32))
    (out / "reference.json").write_text(json.dumps({
        "s2_dir": args.s2_dir, "block_lr_px": args.block, "block_m": args.block * 10, "n_blocks": nb,
        "clear_scl": CLEAR_SCL, "min_clear": args.min_clear, "min_obs": args.min_obs,
        "spread_scale": args.spread_scale, "n_frames": len(paths), "calibration": calib,
        "q_quantiles": np.quantile(q, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist(),
        "n_obs_quantiles": np.quantile(n_obs, [0.1, 0.5, 0.9]).tolist(),
    }, indent=1))
    print(f"reference: {len(paths)} frames, blocks {nb}^2, median n_obs {np.median(n_obs):.0f}, "
          f"median q {np.median(q):.2f}")


def cell_summary(job) -> tuple[str, str, dict]:
    cell, prefix, path, block_hr = job
    with rasterio.open(path) as src:
        x = src.read().astype(np.float32)
    nbc = CELL_HR // block_hr
    blocks = x.reshape(3, nbc, block_hr, nbc, block_hr).mean(axis=(2, 4))
    views = {"N": x, "S": x[:, ::-1, :], "W": np.swapaxes(x, 1, 2), "E": np.swapaxes(x, 1, 2)[:, ::-1, :]}
    out = {"blocks": blocks, "mean": x.mean(axis=(1, 2)), "std": x.std(axis=(1, 2))}
    for s, v in views.items():
        # rows of v run away from the edge; columns run along it
        out[f"{s}_strip"] = v[:, :STRIP].mean(axis=1)
        out[f"{s}_line"] = v[:, 0]
        out[f"{s}_refa_strip"] = v[:, REF_DEPTH - STRIP:REF_DEPTH].mean(axis=1)
        out[f"{s}_refb_strip"] = v[:, REF_DEPTH:REF_DEPTH + STRIP].mean(axis=1)
        out[f"{s}_refa_line"] = v[:, REF_DEPTH - 1]
        out[f"{s}_refb_line"] = v[:, REF_DEPTH]
    return cell, prefix, out


def load_plan(args) -> dict:
    return json.loads((ROOT / args.plan).read_text())


def cell_index(cell: str) -> tuple[int, int]:
    parts = cell.split("_")
    return int(parts[-2][1:]), int(parts[-1][1:])


def sr_path(args, prefix: str, cell: str) -> Path:
    return ROOT / args.samples / f"{prefix}_{cell}" / "qgis" / "sr_pred.tif"


def stage_summarize(args) -> None:
    plan = load_plan(args)
    cells = sorted(plan["independent"])
    changed = [c for c in cells if plan["assignment"][c] != plan["independent"][c]]
    block_hr = args.block * HR_PER_LR
    jobs = [(c, args.indep_prefix, sr_path(args, args.indep_prefix, c), block_hr) for c in cells]
    jobs += [(c, args.coord_prefix, sr_path(args, args.coord_prefix, c), block_hr) for c in changed]
    missing = [str(j[2]) for j in jobs if not j[2].is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} outputs missing, e.g. {missing[:3]}")
    store: dict[str, np.ndarray] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for cell, prefix, out in ex.map(cell_summary, jobs, chunksize=4):
            for k, v in out.items():
                store[f"{prefix}|{cell}|{k}"] = v
    np.savez_compressed(Path(args.out) / "cells.npz", **store)
    print(f"summarized {len(jobs)} outputs ({len(cells)} cells, {len(changed)} with a second base date)")


def affine_fit(F: np.ndarray, C: np.ndarray, w: np.ndarray, offset_only: bool) -> tuple[float, float]:
    ok = np.isfinite(F) & np.isfinite(C) & (w > 0)
    if ok.sum() < 16:
        return 1.0, 0.0
    f, c, ww = F[ok], C[ok], w[ok]
    if offset_only:
        return 1.0, float(np.sum(ww * (c - f)) / np.sum(ww))
    A = np.stack([f, np.ones_like(f)], axis=1) * np.sqrt(ww)[:, None]
    g, h = np.linalg.lstsq(A, c * np.sqrt(ww), rcond=None)[0]
    return float(g), float(h)


def stage_fit(args) -> None:
    plan = load_plan(args)
    ref = np.load(Path(args.out) / "reference.npz")
    C, q = ref["C"], ref["q"]
    cells = np.load(Path(args.out) / "cells.npz")
    nbc = CELL_LR // args.block
    fits: dict = {}
    for key in cells.files:
        prefix, cell, what = key.split("|")
        if what != "blocks":
            continue
        iy, ix = cell_index(cell)
        Cc = C[:, iy * nbc:(iy + 1) * nbc, ix * nbc:(ix + 1) * nbc]
        qc = q[iy * nbc:(iy + 1) * nbc, ix * nbc:(ix + 1) * nbc]
        F = cells[key]
        rec = {"q_mean": float(qc.mean())}
        for mode in ("affine", "offset"):
            gh = [affine_fit(F[b], Cc[b], qc, mode == "offset") for b in range(3)]
            rec[mode] = {"g": [p[0] for p in gh], "h": [p[1] for p in gh]}
        fits[f"{prefix}|{cell}"] = rec
    (Path(args.out) / "fits.json").write_text(json.dumps(fits, indent=1))
    g = np.array([v["affine"]["g"] for v in fits.values()])
    h = np.array([v["affine"]["h"] for v in fits.values()])
    print(f"fits: {len(fits)}; gain quantiles {np.quantile(g, [0.05, 0.5, 0.95]).round(3).tolist()}, "
          f"offset quantiles {np.quantile(h, [0.05, 0.5, 0.95]).round(4).tolist()}")


def seg_mean(v: np.ndarray) -> np.ndarray:
    return v.reshape(3, -1, SEG).mean(axis=2)


def apply(v: np.ndarray, gh: tuple[list, list] | None) -> np.ndarray:
    if gh is None:
        return v
    g, h = np.asarray(gh[0])[:, None], np.asarray(gh[1])[:, None]
    return v * g + h


def stage_evaluate(args) -> None:
    plan = load_plan(args)
    out = Path(args.out)
    cells = np.load(out / "cells.npz")
    fits = json.loads((out / "fits.json").read_text())
    ref = np.load(out / "reference.npz")
    C, q = ref["C"], ref["q"]
    nbc = CELL_LR // args.block
    have = set(plan["independent"])

    grids = {
        "independent": ({c: args.indep_prefix for c in have}, plan["independent"]),
        "coordinated": ({c: (args.coord_prefix if plan["assignment"][c] != plan["independent"][c] else args.indep_prefix)
                         for c in have}, plan["assignment"]),
    }
    by_pos = {cell_index(c): c for c in have}

    def get(prefix, cell, what):
        return cells[f"{prefix}|{cell}|{what}"]

    report: dict = {"reference": json.loads((out / "reference.json").read_text())["q_quantiles"]}
    for gname, (choice, dates) in grids.items():
        for mode in ("raw", "offset", "affine"):
            rows = []
            for (iy, ix), a in by_pos.items():
                for (dy, dx, sa, sb) in ((0, 1, "E", "W"), (1, 0, "S", "N")):
                    b = by_pos.get((iy + dy, ix + dx))
                    if b is None:
                        continue
                    pa, pb = choice[a], choice[b]
                    ga = None if mode == "raw" else (fits[f"{pa}|{a}"][mode]["g"], fits[f"{pa}|{a}"][mode]["h"])
                    gb = None if mode == "raw" else (fits[f"{pb}|{b}"][mode]["g"], fits[f"{pb}|{b}"][mode]["h"])
                    sA, sB = apply(get(pa, a, f"{sa}_strip"), ga), apply(get(pb, b, f"{sb}_strip"), gb)
                    lA, lB = apply(get(pa, a, f"{sa}_line"), ga), apply(get(pb, b, f"{sb}_line"), gb)
                    coarse = float(np.abs(seg_mean(sA) - seg_mean(sB)).mean())
                    signed = (seg_mean(sA) - seg_mean(sB)).mean(axis=1)
                    fine = float(np.abs(lA - lB).mean())
                    ref_c, ref_f = [], []
                    for p, cell, s, gg in ((pa, a, sa, ga), (pb, b, sb, gb)):
                        ra, rb = apply(get(p, cell, f"{s}_refa_strip"), gg), apply(get(p, cell, f"{s}_refb_strip"), gg)
                        ref_c.append(np.abs(seg_mean(ra) - seg_mean(rb)).mean())
                        la, lb = apply(get(p, cell, f"{s}_refa_line"), gg), apply(get(p, cell, f"{s}_refb_line"), gg)
                        ref_f.append(np.abs(la - lb).mean())
                    rows.append({"a": a, "b": b, "same_date": dates[a] == dates[b], "coarse": coarse,
                                 "coarse_ref": float(np.mean(ref_c)), "fine": fine, "fine_ref": float(np.mean(ref_f)),
                                 "signed": signed.tolist()})
            summary = {}
            for label, sel in (("all", lambda r: True), ("same_date", lambda r: r["same_date"]),
                               ("different_date", lambda r: not r["same_date"])):
                rs = [r for r in rows if sel(r)]
                if not rs:
                    continue
                co = np.array([r["coarse"] for r in rs]); cr = np.array([r["coarse_ref"] for r in rs])
                fi = np.array([r["fine"] for r in rs]); fr = np.array([r["fine_ref"] for r in rs])
                summary[label] = {
                    "n": len(rs),
                    "coarse_mean": float(co.mean()), "coarse_ref_mean": float(cr.mean()),
                    "coarse_ratio_median": float(np.median(co / cr)), "coarse_excess_mean": float((co - cr).mean()),
                    "fine_mean": float(fi.mean()), "fine_ref_mean": float(fr.mean()),
                    "fine_ratio_median": float(np.median(fi / fr)),
                    "frac_coarse_gt_2x_ref": float(np.mean(co > 2 * cr)),
                }
            report[f"{gname}|{mode}"] = summary
            (out / f"boundaries_{gname}_{mode}.json").write_text(json.dumps(rows))

    # Interior change and agreement with C.
    inter = {}
    for key, rec in fits.items():
        prefix, cell = key.split("|")
        iy, ix = cell_index(cell)
        Cc = C[:, iy * nbc:(iy + 1) * nbc, ix * nbc:(ix + 1) * nbc]
        qc = q[iy * nbc:(iy + 1) * nbc, ix * nbc:(ix + 1) * nbc]
        F = get(prefix, cell, "blocks")
        std = get(prefix, cell, "std")
        row = {}
        for mode in ("raw", "offset", "affine"):
            Fm = F if mode == "raw" else apply(F.reshape(3, -1), (rec[mode]["g"], rec[mode]["h"])).reshape(F.shape)
            w = qc[None] * np.isfinite(Cc)
            row[f"rms_to_C_{mode}"] = float(np.sqrt(np.nansum(w * (Fm - Cc) ** 2) / np.sum(w)))
            if mode != "raw":
                row[f"mean_abs_change_{mode}"] = float(np.abs(Fm - F).mean())
                row[f"contrast_ratio_{mode}"] = rec[mode]["g"]
        row["std"] = std.tolist()
        inter[key] = row
    report["interior"] = {
        k: float(np.median([v[k] for v in inter.values()]))
        for k in ("rms_to_C_raw", "rms_to_C_offset", "rms_to_C_affine", "mean_abs_change_offset", "mean_abs_change_affine")
    }
    g = np.array([v["contrast_ratio_affine"] for v in inter.values()])
    report["interior"]["gain_quantiles_5_50_95"] = np.quantile(g, [0.05, 0.5, 0.95]).tolist()

    # Base-frame sensitivity: cells fitted with two base dates.
    changed = sorted(c for c in have if plan["assignment"][c] != plan["independent"][c])
    sens = []
    for c in changed:
        Fa, Fb = get(args.indep_prefix, c, "blocks"), get(args.coord_prefix, c, "blocks")
        row = {"cell": c, "date_indep": plan["independent"][c], "date_coord": plan["assignment"][c]}
        for mode in ("raw", "offset", "affine"):
            if mode == "raw":
                A, B = Fa, Fb
            else:
                fa, fb = fits[f"{args.indep_prefix}|{c}"][mode], fits[f"{args.coord_prefix}|{c}"][mode]
                A = apply(Fa.reshape(3, -1), (fa["g"], fa["h"])).reshape(Fa.shape)
                B = apply(Fb.reshape(3, -1), (fb["g"], fb["h"])).reshape(Fb.shape)
            row[f"coarse_rms_{mode}"] = float(np.sqrt(np.mean((A - B) ** 2)))
            row[f"coarse_mean_diff_{mode}"] = (A - B).mean(axis=(1, 2)).tolist()
        sens.append(row)
    if args.full_res_sensitivity:
        for row in sens:
            c = row["cell"]
            with rasterio.open(sr_path(args, args.indep_prefix, c)) as s:
                A = s.read().astype(np.float32)
            with rasterio.open(sr_path(args, args.coord_prefix, c)) as s:
                B = s.read().astype(np.float32)
            for mode in ("raw", "affine"):
                if mode == "raw":
                    a_, b_ = A, B
                else:
                    fa, fb = fits[f"{args.indep_prefix}|{c}"][mode], fits[f"{args.coord_prefix}|{c}"][mode]
                    a_ = A * np.asarray(fa["g"])[:, None, None] + np.asarray(fa["h"])[:, None, None]
                    b_ = B * np.asarray(fb["g"])[:, None, None] + np.asarray(fb["h"])[:, None, None]
                d = a_ - b_
                k = args.block * HR_PER_LR
                low = d.reshape(3, CELL_HR // k, k, CELL_HR // k, k).mean(axis=(2, 4))
                high = d - np.repeat(np.repeat(low, k, axis=1), k, axis=2)
                row[f"full_rms_{mode}"] = float(np.sqrt(np.mean(d ** 2)))
                row[f"highpass_rms_{mode}"] = float(np.sqrt(np.mean(high ** 2)))
    (out / "sensitivity.json").write_text(json.dumps(sens, indent=1))
    keys = [k for k in sens[0] if k.startswith(("coarse_rms", "full_rms", "highpass_rms"))]
    report["sensitivity"] = {"n": len(sens), **{k: float(np.median([r[k] for r in sens])) for k in keys},
                             "n_coarse_improved_affine": int(sum(r["coarse_rms_affine"] < r["coarse_rms_raw"] for r in sens))}
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "interior"}, indent=1))
    print("interior", json.dumps(report["interior"], indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["reference", "summarize", "fit", "evaluate", "all"])
    ap.add_argument("--s2_dir", default="data/s2_revisits/national_2025_v2/32VNM")
    ap.add_argument("--plan", default="production/seams/32VNM_full/identity_plan.json")
    ap.add_argument("--samples", default="single_samples/32VNM/sample")
    ap.add_argument("--indep_prefix", default="prod_k4_base2")
    ap.add_argument("--coord_prefix", default="prod_k4_icm")
    ap.add_argument("--out", default=str(ROOT / "paper/results/regional_retarget_v1/32VNM"))
    ap.add_argument("--block", type=int, default=8, help="reference block size in LR px (8 = 80 m)")
    ap.add_argument("--min_clear", type=float, default=0.75)
    ap.add_argument("--min_blocks", type=int, default=2000)
    ap.add_argument("--min_obs", type=int, default=3)
    ap.add_argument("--spread_scale", type=float, default=0.01)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--full_res_sensitivity", action="store_true")
    args = ap.parse_args()
    stages = ["reference", "summarize", "fit", "evaluate"] if args.stage == "all" else [args.stage]
    for s in stages:
        globals()[f"stage_{s}"](args)


if __name__ == "__main__":
    main()
