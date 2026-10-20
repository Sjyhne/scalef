#!/usr/bin/env python3
"""Expose full-granule OmniCloudMask files to nested and complete-field stacks, and write a
masks-only frame screen for them.

``scripts/ocm_granules.py`` writes ``<stem>_ocm.tif`` next to each parent granule. Nested stacks
(``<city>_pXX_YY``, ``<city>_nest<side>_...``) hold per-frame symlinks into the parent dir, and
``--frame_screen`` requires the mask to exist inside the stack dir, so this adds one symlink per
frame. The screen it writes excludes no frame and only switches ``cloud_mask`` to the ``_ocm``
suffix, matching the named-site masks-only arm.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NESTED_FULL = ROOT / "single_samples/sweep_results/nested_floatval_v3_full_manifest.json"
SUFFIX = "_ocm"


def nested_stacks() -> list[Path]:
    jobs = json.loads(NESTED_FULL.read_text())["jobs"]
    return sorted({Path(j["command"][j["command"].index("--s2-dir") + 1]) for j in jobs})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--screen_out", type=Path, default=ROOT / "paper/results/frame_screen_nested_masks_only.json")
    args = ap.parse_args()
    stacks = nested_stacks()
    entries, linked, missing = {}, 0, []
    for d in stacks:
        meta = json.loads((d / "meta.json").read_text())
        parent = Path(meta["parent_s2_dir"])
        for fr in meta["frames"]:
            stem = Path(fr["path"]).stem
            src = parent / f"{stem}{SUFFIX}.tif"
            dst = d / f"{stem}{SUFFIX}.tif"
            if not src.is_file():
                missing.append(str(src))
                continue
            if not dst.is_symlink() and not dst.exists():
                dst.symlink_to(src)
                linked += 1
        entries[d.name] = {"stack": str(d.relative_to(ROOT)), "n_frames": len(meta["frames"]), "n_excluded": 0,
                           "exclude": [], "flagged": {}, "cloud_mask_suffix": SUFFIX}
    args.screen_out.write_text(json.dumps({
        "schema": "frame_screen.v1", "rule": "masks only: no exclusions; full-granule OmniCloudMask class maps",
        "source": "scripts/ocm_granules.py + scripts/link_granule_masks.py", "stacks": entries}, indent=1) + "\n")
    print(f"{len(stacks)} stacks, {linked} new symlinks, {len(set(missing))} missing masks -> {args.screen_out}")
    if missing:
        print("e.g.", sorted(set(missing))[:3])


if __name__ == "__main__":
    main()
