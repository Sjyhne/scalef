#!/usr/bin/env python3
"""Run OmniCloudMask over full Sentinel-2 granules and write one class map per frame.

The fetch step ran OmniCloudMask on a 2.75 km screening window only, so every LR window cut from
a granule outside that window (nested patches, complete fields) had no cloud mask, and the
national 32VNM stacks fall back to SCL. This writes ``<stem>_ocm.tif`` (uint8: 0 clear, 1 thick,
2 thin, 3 shadow; full granule grid, tiled, deflate) next to each frame, using the fetch step's
inputs (red, green, NIR as DN/10000). The loader reads full-granule masks by window, so any stack
whose frames symlink to these granules can use them via a frame screen with
``cloud_mask_suffix: "_ocm"``. ``meta.json`` is not modified.

Dirs are split round-robin over ``--shard i/n`` so several GPUs can share the work.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
NESTED_FULL = ROOT / "single_samples/sweep_results/nested_floatval_v3_full_manifest.json"


def parent_dirs() -> list[Path]:
    """City granule dirs behind the nested/complete-field stacks (resolved through symlinks)."""
    jobs = json.loads(NESTED_FULL.read_text())["jobs"]
    out: set[Path] = set()
    for j in jobs:
        if j["side"] != 512:
            continue
        d = Path(j["command"][j["command"].index("--s2-dir") + 1])
        meta = json.loads((d / "meta.json").read_text())
        out.add(Path(meta["parent_s2_dir"]))
    return sorted(out)


def write_mask(path: Path, classes: np.ndarray, src_profile: dict) -> None:
    prof = dict(driver="GTiff", height=classes.shape[0], width=classes.shape[1], count=1, dtype="uint8",
                crs=src_profile["crs"], transform=src_profile["transform"], compress="deflate", tiled=True,
                blockxsize=512, blockysize=512, predictor=2)
    tmp = path.with_suffix(".tmp.tif")
    with rasterio.open(tmp, "w", **prof) as dst:
        dst.write(classes.astype(np.uint8), 1)
        dst.set_band_description(1, "ocm: 0 clear, 1 thick, 2 thin, 3 shadow")
    tmp.replace(path)


def process_dir(
    d: Path,
    device: str,
    overwrite: bool,
    batch: int,
    frame_shard: str = "0/1",
) -> list[dict]:
    from omnicloudmask import predict_from_array

    meta = json.loads((d / "meta.json").read_text())
    fi, fn = (int(v) for v in frame_shard.split("/"))
    if fn < 1 or fi < 0 or fi >= fn:
        raise SystemExit(f"bad --frame-shard {frame_shard}")
    rows = []
    for fr in list(meta["frames"])[fi::fn]:
        src_path = (d / fr["path"]).resolve()
        out = src_path.with_name(f"{Path(fr['path']).stem}_ocm.tif")
        t0 = time.time()
        if out.is_file() and not overwrite:
            with rasterio.open(out) as s:
                classes = s.read(1)
            status = "existing"
        else:
            with rasterio.open(src_path) as src:
                stack = src.read([1, 2, 4])  # red, green, NIR (bands stored B04, B03, B02, B08)
                prof = src.profile
            refl = np.clip(stack.astype(np.float32) / 10000.0, 0.0, 1.5)
            pred = predict_from_array(refl, inference_device=device, batch_size=batch, patch_size=1000,
                                      patch_overlap=300)
            classes = np.asarray(pred).squeeze().astype(np.uint8)
            del refl, stack
            write_mask(out, classes, prof)
            status = "written"
        valid = classes.size
        rows.append({"dir": str(d), "frame": fr["path"], "mask": out.name, "status": status,
                     "cloud_frac": float(np.isin(classes, (1, 2)).mean()), "shadow_frac": float((classes == 3).mean()),
                     "seconds": round(time.time() - t0, 1)})
        print(f"  {d.name}/{fr['path']}: {status} cloud {rows[-1]['cloud_frac']:.3f} shadow {rows[-1]['shadow_frac']:.3f} "
              f"({rows[-1]['seconds']}s)", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dirs", nargs="*", default=None, help="Granule dirs; default: nested-study parents")
    ap.add_argument("--national", nargs="*", default=[], help="Extra dirs, e.g. data/s2_revisits/national_2025_v2/32VNM")
    ap.add_argument("--shard", default="0/1", help="i/n round-robin over dirs")
    ap.add_argument(
        "--frame-shard",
        default="0/1",
        help="i/n round-robin over frames inside each dir (one GPU per shard)",
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--log", type=Path, default=None)
    args = ap.parse_args()
    dirs = [Path(x) for x in args.dirs] if args.dirs else parent_dirs()
    dirs += [ROOT / x if not Path(x).is_absolute() else Path(x) for x in args.national]
    i, n = (int(v) for v in args.shard.split("/"))
    mine = dirs[i::n]
    rows = []
    for k, d in enumerate(mine, 1):
        print(f"[{k}/{len(mine)}] {d}", flush=True)
        rows += process_dir(d, args.device, args.overwrite, args.batch, args.frame_shard)
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
