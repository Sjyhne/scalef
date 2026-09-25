#!/usr/bin/env python3
"""Recompute OmniCloudMask class maps over the full LR512 window and measure what the old masks miss.

The fetch step ran OmniCloudMask on the 2.5 km screening window only and saved its class map
(0 clear, 1 thick, 2 thin, 3 shadow) as ``*_aoi_cloud.tif``; the loader reprojects that map onto
the fitted LR window and treats everything outside it as clear. This script reruns
OmniCloudMask, with the fetch step's inputs (red, green, NIR as DN/10000), on each frame's
full ``aoi_window`` and writes ``<stem>_lr512_ocm.tif`` next to the frame. It does not change
``meta.json``, so existing fits and loaders are unaffected.

Per frame it records the class fractions over the full window and over the ring outside the old
screening window, the fraction the current masks flag, agreement with the old map inside the
screening window, and whether the frame would pass the 15% admission limit on the full window
(thick+thin, and with shadow).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds, transform as window_transform

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

V5_MANIFEST = ROOT / "paper/results/run_manifests/confirmatory_v5_boa__run_manifest.json"
NESTED_FULL = ROOT / "single_samples/sweep_results/nested_floatval_v3_full_manifest.json"
OUT_JSON = ROOT / "paper/results/ocm_full_window.json"
CLEAR, THICK, THIN, SHADOW = 0, 1, 2, 3
MAX_CLOUD_FRAC = 0.15


def stack_dirs(which: str) -> list[Path]:
    dirs: list[Path] = []
    if which in ("named", "all"):
        jobs = json.loads(V5_MANIFEST.read_text())["jobs"]
        dirs += sorted({Path(j["command"][j["command"].index("--s2-dir") + 1]) for j in jobs})
    if which in ("complete", "all"):
        jobs = [j for j in json.loads(NESTED_FULL.read_text())["jobs"] if j["side"] == 512]
        dirs += sorted({Path(j["command"][j["command"].index("--s2-dir") + 1]) for j in jobs})
    return dirs


def write_classes(path: Path, classes: np.ndarray, transform, crs) -> None:
    profile = dict(driver="GTiff", height=classes.shape[0], width=classes.shape[1], count=1, dtype="uint8",
                   crs=crs, transform=transform, compress="deflate")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(classes.astype(np.uint8), 1)
        dst.set_band_description(1, "ocm: 0 clear, 1 thick, 2 thin, 3 shadow")


def frac(classes: np.ndarray, valid: np.ndarray, ids) -> float:
    return float(np.isin(classes[valid], list(ids)).mean()) if valid.any() else float("nan")


def process_dir(d: Path, device: str, overwrite: bool) -> list[dict]:
    from omnicloudmask import predict_from_array

    meta = json.loads((d / "meta.json").read_text())
    aoi = meta["aoi_window"]
    win = Window(aoi["col_off"], aoi["row_off"], aoi["width"], aoi["height"])
    rows = []
    for fr in meta["frames"]:
        stem = Path(fr["path"]).stem
        out = d / f"{stem}_lr512_ocm.tif"
        with rasterio.open(d / fr["path"]) as src:
            stack = src.read(window=win)
            transform = window_transform(win, src.transform)
            crs = src.crs
        valid = np.any(stack > 0, axis=0)
        if out.is_file() and not overwrite:
            with rasterio.open(out) as s:
                classes = s.read(1)
        else:
            refl = np.clip(stack.astype(np.float32) / 10000.0, 0.0, 1.5)
            pred = predict_from_array(refl[[0, 1, 3]], inference_device=device)
            classes = np.asarray(pred).squeeze().astype(np.uint8)
            write_classes(out, classes, transform, crs)

        inner = np.zeros(classes.shape, bool)
        old = np.zeros(classes.shape, np.uint8)
        if fr.get("cloud_mask") and (d / fr["cloud_mask"]).is_file():
            with rasterio.open(d / fr["cloud_mask"]) as s:
                ow = from_bounds(*s.bounds, transform=transform).round_offsets().round_lengths()
                r0, c0 = int(ow.row_off), int(ow.col_off)
                h, w = s.height, s.width
                old[r0:r0 + h, c0:c0 + w] = s.read(1)
                inner[r0:r0 + h, c0:c0 + w] = True
        ring = valid & ~inner
        v_in = valid & inner
        rows.append({
            "stack": str(d.relative_to(ROOT)), "frame": fr["path"], "date": str(fr.get("datetime", ""))[:10],
            "base": fr is meta["frames"][0],
            "valid_frac": float(valid.mean()), "screen_window_frac": float(inner.mean()),
            "full": {k: frac(classes, valid, ids) for k, ids in (("cloud", (THICK, THIN)), ("shadow", (SHADOW,)))},
            "ring": {k: frac(classes, ring, ids) for k, ids in (("cloud", (THICK, THIN)), ("shadow", (SHADOW,)))},
            "flagged_now": frac(old, valid, (THICK, THIN, SHADOW)),
            "flagged_new": frac(classes, valid, (THICK, THIN, SHADOW)),
            "agreement_in_screen_window": float(((old > 0) == (classes > 0))[v_in].mean()) if v_in.any() else None,
            "old_admission_cloud_frac": fr.get("cloud_frac"),
        })
        r = rows[-1]
        r["passes_full_window"] = bool(r["full"]["cloud"] <= MAX_CLOUD_FRAC)
        r["passes_full_window_with_shadow"] = bool(r["full"]["cloud"] + r["full"]["shadow"] <= MAX_CLOUD_FRAC)
    return rows


def summarize(rows: list[dict]) -> dict:
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["stack"], []).append(r)
    per_stack = {}
    for s, rs in by.items():
        per_stack[s] = {
            "n_frames": len(rs),
            "flagged_now_mean": float(np.mean([r["flagged_now"] for r in rs])),
            "flagged_new_mean": float(np.mean([r["flagged_new"] for r in rs])),
            "flagged_new_max": float(np.max([r["flagged_new"] for r in rs])),
            "ring_cloud_max": float(np.nanmax([r["ring"]["cloud"] for r in rs])),
            "n_fail_full_window": int(sum(not r["passes_full_window"] for r in rs)),
            "n_fail_full_window_with_shadow": int(sum(not r["passes_full_window_with_shadow"] for r in rs)),
            "base_flagged_new": next(r["flagged_new"] for r in rs if r["base"]),
            "agreement_in_screen_window_min": float(np.nanmin([r["agreement_in_screen_window"] or np.nan for r in rs])),
        }
    agree = [r["agreement_in_screen_window"] for r in rows if r["agreement_in_screen_window"] is not None]
    return {
        "n_stacks": len(by), "n_frames": len(rows),
        "agreement_in_screen_window_median": float(np.median(agree)) if agree else None,
        "flagged_now_mean": float(np.mean([r["flagged_now"] for r in rows])),
        "flagged_new_mean": float(np.mean([r["flagged_new"] for r in rows])),
        "n_frames_fail_full_window": int(sum(not r["passes_full_window"] for r in rows)),
        "n_frames_fail_full_window_with_shadow": int(sum(not r["passes_full_window_with_shadow"] for r in rows)),
        "per_stack": per_stack,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", choices=("named", "complete", "all"), default="named")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    rows = []
    dirs = stack_dirs(args.set)
    for i, d in enumerate(dirs, 1):
        rows += process_dir(d, args.device, args.overwrite)
        print(f"[{i}/{len(dirs)}] {d.name}", flush=True)
    summary = summarize(rows)
    out = args.out or OUT_JSON.with_name(f"ocm_full_window_{args.set}.json")
    out.write_text(json.dumps({"summary": summary, "frames": rows}, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_stack"}, indent=1))


if __name__ == "__main__":
    main()
