#!/usr/bin/env python3
"""Build screened stacks that replace excluded frames with the next admissible revisits.

For each stack in a frame screen (``scripts/build_frame_screen.py``) this keeps the frames the
screen kept and continues the original admission walk: L2A scenes on the stack's MGRS tile within
its original ``date_range``, ranked by distance from the centre (NIB) date and then scene cloud
cover, first product per day, skipping days already in the v5 stack. A candidate is admitted when,
over the full LR512 window, valid pixels >= 85%, OmniCloudMask thick+thin+shadow <= the screen's
``max_masked`` and the white fraction <= ``max_white``. The walk stops at the v5 frame count or
when the date range is exhausted; shortfalls are reported, not filled from outside the range.

Only the LR512 window is read from the COGs. New frames are written on the full granule grid
(tiled, sparse outside the window) so the loader's ``aoi_window`` crop works unchanged.

Output: ``data/s2_revisits_screen_v1/{city}_lr512/`` with symlinks to the kept v5 frames and their
``_lr512_ocm`` maps, the new frames and maps, and a ``meta.json`` whose ``cloud_mask`` entries
point at the full-window maps. The v5 stacks are not modified.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import timezone
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window, transform as window_transform

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from fetch_s2_revisits import BANDS_L2A, L2A, _sign, query_items, rank_items  # noqa: E402

OUT_ROOT = ROOT / "data" / "s2_revisits_screen_v1"
BASELINES = ROOT / "data" / "s2_revisits" / "processing_baselines.json"
MIN_VALID = 0.85
MASK_SUFFIX = "_lr512_ocm"


def read_window(item, win: Window) -> tuple[np.ndarray, dict]:
    import planetary_computer as pc

    signed = _sign(item)

    def one(band: str):
        with rasterio.open(pc.sign(signed.assets[band].href)) as src:
            return src.read(1, window=win), src.profile, src.transform, src.crs

    with ThreadPoolExecutor(max_workers=len(BANDS_L2A)) as pool:
        out = list(pool.map(one, BANDS_L2A))
    shapes = {(o[1]["width"], o[1]["height"], tuple(o[2])[:6]) for o in out}
    if len(shapes) != 1:
        raise ValueError(f"{item.id}: 10 m bands are not on one grid")
    prof = out[0][1]
    return np.stack([o[0] for o in out]), {"width": prof["width"], "height": prof["height"],
                                            "transform": out[0][2], "crs": out[0][3]}


def offset_for(baseline: str | None) -> float:
    if baseline is None:
        raise KeyError("item has no s2:processing_baseline")
    major, minor = (int(p) for p in str(baseline).split("."))
    return 0.1 if (major, minor) >= (4, 0) else 0.0


def screen_arrays(stack: np.ndarray, classes: np.ndarray, offset: float, white_level: float) -> dict:
    valid = np.any(stack > 0, axis=0)
    n = max(int(valid.sum()), 1)
    masked = float(np.isin(classes[valid], [1, 2, 3]).sum() / n)
    refl = stack[:3].astype(np.float32) / 10000.0 - offset
    white = float((np.all(refl > white_level, axis=0) & np.all(stack[:3] > 0, axis=0)).sum() / n)
    return {"valid_frac": float(valid.mean()), "masked": masked, "white": white}


def write_sparse_frame(path: Path, stack: np.ndarray, grid: dict, win: Window) -> None:
    profile = dict(driver="GTiff", width=grid["width"], height=grid["height"], count=stack.shape[0],
                   dtype="uint16", crs=grid["crs"], transform=grid["transform"], nodata=0,
                   tiled=True, blockxsize=512, blockysize=512, compress="deflate", predictor=2,
                   SPARSE_OK=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(stack.astype(np.uint16), window=win)
        for i, b in enumerate(BANDS_L2A, start=1):
            dst.set_band_description(i, b)


def write_classes(path: Path, classes: np.ndarray, transform, crs) -> None:
    profile = dict(driver="GTiff", height=classes.shape[0], width=classes.shape[1], count=1, dtype="uint8",
                   crs=crs, transform=transform, compress="deflate")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(classes.astype(np.uint8), 1)
        dst.set_band_description(1, "ocm: 0 clear, 1 thick, 2 thin, 3 shadow")


def link(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src.resolve())


def build_stack(name: str, entry: dict, rule: dict, device: str, max_candidates: int, dry_run: bool) -> dict:
    from omnicloudmask import predict_from_array

    src_dir = ROOT / entry["stack"]
    meta = json.loads((src_dir / "meta.json").read_text())
    aw = meta["aoi_window"]
    win = Window(aw["col_off"], aw["row_off"], aw["width"], aw["height"])
    excluded = set(entry["exclude"])
    kept = [dict(fr) for fr in meta["frames"] if fr["path"] not in excluded]
    target = len(meta["frames"])
    used_days = {str(fr["datetime"])[:10] for fr in meta["frames"]}
    need = target - len(kept)

    out_dir = OUT_ROOT / name
    report = {"stack": name, "source": entry["stack"], "n_v5": target, "n_kept": len(kept), "need": need,
              "date_range": meta["date_range"], "center_date": meta["center_date"], "added": [],
              "rejected": [], "n_candidates_checked": 0}
    new_frames: list[dict] = []
    if need > 0:
        items = rank_items(query_items(L2A, meta["bbox_wgs84"], meta["date_range"], max_cloud=100.0,
                                       max_items=1000, mgrs_tile=meta["mgrs_tile"]), meta["center_date"])
        seen: set[str] = set()
        for it in items:
            if len(new_frames) >= need or report["n_candidates_checked"] >= max_candidates:
                break
            day = it.datetime.astimezone(timezone.utc).strftime("%Y-%m-%d")
            if day in used_days or day in seen:
                continue
            seen.add(day)
            report["n_candidates_checked"] += 1
            try:
                stack, grid = read_window(it, win)
                offset = offset_for(it.properties.get("s2:processing_baseline"))
            except Exception as exc:  # noqa: BLE001
                report["rejected"].append({"date": day, "stac_id": it.id, "why": [f"read: {exc}"]})
                continue
            if (grid["width"], grid["height"]) != (meta["width"], meta["height"]) or \
                    list(grid["transform"])[:6] != list(meta["transform"]):
                report["rejected"].append({"date": day, "stac_id": it.id, "why": ["grid mismatch"]})
                continue
            refl = np.clip(stack.astype(np.float32) / 10000.0, 0.0, 1.5)
            classes = np.asarray(predict_from_array(refl[[0, 1, 3]], inference_device=device)).squeeze().astype(np.uint8)
            s = screen_arrays(stack, classes, offset, rule["white_level"])
            why = []
            if s["valid_frac"] < MIN_VALID:
                why.append(f"valid={s['valid_frac']:.3f}")
            if s["masked"] > rule["max_masked"]:
                why.append(f"masked={s['masked']:.3f}")
            if s["white"] > rule["max_white"]:
                why.append(f"white={s['white']:.3f}")
            rec = {"date": day, "stac_id": it.id, **{k: round(v, 4) for k, v in s.items()}}
            if why:
                report["rejected"].append({**rec, "why": why})
                continue
            stem = f"r{len(new_frames) + 1:02d}_{day.replace('-', '')}"
            if not dry_run:
                out_dir.mkdir(parents=True, exist_ok=True)
                write_sparse_frame(out_dir / f"{stem}.tif", stack, grid, win)
                write_classes(out_dir / f"{stem}{MASK_SUFFIX}.tif", classes,
                              window_transform(win, grid["transform"]), grid["crs"])
            new_frames.append({
                "index": target + len(new_frames) + 1, "path": f"{stem}.tif", "cloud_mask": f"{stem}{MASK_SUFFIX}.tif",
                "stac_id": it.id, "mgrs_tile": meta["mgrs_tile"], "datetime": it.datetime.isoformat(),
                "eo:cloud_cover": it.properties.get("eo:cloud_cover"), "cloud_frac": s["masked"],
                "valid_frac": s["valid_frac"], "white_frac": s["white"], "replacement": True,
                "s2:processing_baseline": it.properties.get("s2:processing_baseline"),
            })
            report["added"].append(rec)
            print(f"  {name}: + {day} masked={s['masked']:.3f} white={s['white']:.3f}", flush=True)

    report["n_final"] = len(kept) + len(new_frames)
    report["shortfall"] = target - report["n_final"]
    if dry_run:
        return report

    out_dir.mkdir(parents=True, exist_ok=True)
    for fr in kept:
        stem = Path(fr["path"]).stem
        link(src_dir / fr["path"], out_dir / fr["path"])
        link(src_dir / f"{stem}{MASK_SUFFIX}.tif", out_dir / f"{stem}{MASK_SUFFIX}.tif")
        fr["cloud_mask"] = f"{stem}{MASK_SUFFIX}.tif"
    for extra in ("preview.png",):
        if (src_dir / extra).is_file():
            link(src_dir / extra, out_dir / extra)
    new_meta = {k: v for k, v in meta.items() if k != "frames"}
    new_meta.update(frames=kept + new_frames, screen_v1={"rule": rule, "source_stack": entry["stack"],
                                                         "excluded": sorted(excluded), "report": {
                                                             k: report[k] for k in ("n_v5", "n_kept", "n_final", "shortfall")}})
    (out_dir / "meta.json").write_text(json.dumps(new_meta, indent=2))

    if new_frames:
        cache = json.loads(BASELINES.read_text())
        for fr in new_frames:
            cache[fr["stac_id"]] = fr["s2:processing_baseline"]
        BASELINES.write_text(json.dumps(dict(sorted(cache.items())), indent=1) + "\n")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--screen", type=Path, default=ROOT / "paper/results/frame_screen_v1_named.json")
    ap.add_argument("--only", nargs="*", default=None, help="stack names, e.g. kautokeino_lr512")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_candidates", type=int, default=200)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    screen = json.loads(args.screen.read_text())
    reports = {}
    for name, entry in sorted(screen["stacks"].items()):
        if args.only and name not in args.only:
            continue
        r = build_stack(name, entry, screen["rule"], args.device, args.max_candidates, args.dry_run)
        reports[name] = r
        print(f"{name:20s} v5 {r['n_v5']:2d} kept {r['n_kept']:2d} added {len(r['added']):2d} "
              f"final {r['n_final']:2d} shortfall {r['shortfall']:2d} checked {r['n_candidates_checked']}", flush=True)
    out = ROOT / "paper/results/screened_refits/replacements_v1.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        prev = json.loads(out.read_text()) if out.is_file() else {}
        prev.update(reports)
        out.write_text(json.dumps(prev, indent=2))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
