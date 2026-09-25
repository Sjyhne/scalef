#!/usr/bin/env python3
"""Build a frame screen for ``optimize.py --frame_screen`` from the full-window OmniCloudMask maps.

A non-base frame is excluded when, over the full LR512 window, either
  * OmniCloudMask flags more than ``--max_masked`` of valid pixels (thick + thin + shadow), or
  * more than ``--max_white`` of valid pixels are uniformly bright (red, green and blue surface
    reflectance all above ``--white_level`` after the BOA offset), which catches snow and
    overcast frames that OmniCloudMask leaves clear.
Base frames are never excluded (the refits pin them with ``--force_base_date``); their flagged
pixels are masked through the new class maps like every other kept frame.

Input: ``paper/results/ocm_full_window_{set}.json`` from ``scripts/recompute_ocm_masks.py``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from s2_dataset import boa_add_offset  # noqa: E402

MASK_SUFFIX = "_lr512_ocm"


def white_fraction(stack: Path, frame: dict, window: Window, level: float) -> float:
    with rasterio.open(stack / frame["path"]) as src:
        rgb = src.read([1, 2, 3], window=window).astype(np.float32)
    valid = np.all(rgb > 0, axis=0)
    refl = rgb / 10000.0 - boa_add_offset(frame["stac_id"])
    white = np.all(refl > level, axis=0) & valid
    return float(white.sum() / max(int(valid.sum()), 1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default="named")
    ap.add_argument("--max_masked", type=float, default=0.15)
    ap.add_argument("--max_white", type=float, default=0.10)
    ap.add_argument("--white_level", type=float, default=0.2)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    stats = json.loads((ROOT / f"paper/results/ocm_full_window_{args.set}.json").read_text())
    out_path = args.out or ROOT / f"paper/results/frame_screen_v1_{args.set}.json"
    by_stack: dict[str, list[dict]] = {}
    for rec in stats["frames"]:
        by_stack.setdefault(rec["stack"], []).append(rec)

    stacks: dict[str, dict] = {}
    for rel, recs in sorted(by_stack.items()):
        stack = ROOT / rel
        meta = json.loads((stack / "meta.json").read_text())
        frames = {fr["path"]: fr for fr in meta["frames"]}
        aw = meta["aoi_window"]
        window = Window(int(aw["col_off"]), int(aw["row_off"]), int(aw["width"]), int(aw["height"]))
        exclude, reasons = [], {}
        for rec in recs:
            fr = frames[rec["frame"]]
            masked = rec["full"]["cloud"] + rec["full"]["shadow"]
            white = white_fraction(stack, fr, window, args.white_level)
            why = []
            if masked > args.max_masked:
                why.append(f"masked={masked:.3f}")
            if white > args.max_white:
                why.append(f"white={white:.3f}")
            rec_out = {"date": rec["date"], "base": rec["base"], "masked": round(masked, 4), "white": round(white, 4)}
            if why and not rec["base"]:
                exclude.append(rec["frame"])
                reasons[rec["frame"]] = {**rec_out, "why": why}
            elif why:
                reasons[rec["frame"]] = {**rec_out, "why": why, "kept_as_base": True}
        stacks[stack.name] = {
            "stack": rel,
            "n_frames": len(recs),
            "n_excluded": len(exclude),
            "exclude": sorted(exclude),
            "flagged": reasons,
            "cloud_mask_suffix": MASK_SUFFIX,
        }
        print(f"{stack.name:28s} {len(recs):3d} frames, exclude {len(exclude):2d}  "
              + ", ".join(f"{reasons[f]['date']}({'/'.join(reasons[f]['why'])})" for f in sorted(exclude)))

    out = {
        "schema": "frame_screen_v1",
        "rule": {"max_masked": args.max_masked, "max_white": args.max_white, "white_level": args.white_level,
                 "masked_classes": "thick+thin+shadow", "window": "full LR512 aoi_window", "base_frames": "never excluded"},
        "source": str(Path("paper/results") / f"ocm_full_window_{args.set}.json"),
        "stacks": stacks,
    }
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}  excluded {sum(s['n_excluded'] for s in stacks.values())} / "
          f"{sum(s['n_frames'] for s in stacks.values())}")


if __name__ == "__main__":
    main()
