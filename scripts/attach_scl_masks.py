#!/usr/bin/env python3
"""Attach per-frame SCL cloud masks onto an existing S2 revisit stack.

Does not re-download RGB. Writes ``{stem}_scl.tif`` and ``{stem}_aoi_cloud.tif``
(1 = cloudy) on the same grid as each RGB GeoTIFF, then points ``meta.json``
``cloud_mask`` at the aoi_cloud file so ``s2_dataset`` can AND it into the loss.

Example
-------
    python scripts/attach_scl_masks.py \\
      --src data/s2_revisits/national_2025/32VNM \\
      --propagate-tiles
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.s2_cloud_mask import (  # noqa: E402
    cloudy_from_scl,
    read_scl_10m,
    reproject_scl_to_grid,
)
from scripts.fetch_s2_revisits import L2A, catalog, query_items, write_mask  # noqa: E402
from scripts.make_granule_tiles import _symlink  # noqa: E402


def _get_item(stac_id: str):
    try:
        item = catalog().get_collection(L2A).get_item(str(stac_id))
        if item is not None:
            return item
    except Exception:
        pass
    return None


def _index_items(meta: dict) -> dict[str, object]:
    """One STAC search for the stack's date window, keyed by item id."""
    frames = list(meta.get("frames") or [])
    mgrs = None
    if frames:
        mgrs = frames[0].get("mgrs_tile") or meta.get("mgrs_tile")
    date_range = meta.get("date_range")
    if not date_range:
        dates = []
        for fr in frames:
            dt = str(fr.get("datetime") or "")[:10]
            if dt:
                dates.append(dt)
        if dates:
            date_range = f"{min(dates)}/{max(dates)}"
    by_id: dict[str, object] = {}
    if date_range and mgrs:
        items = query_items(
            L2A,
            None,
            str(date_range),
            max_cloud=100.0,
            max_items=500,
            mgrs_tile=str(mgrs),
        )
        by_id = {it.id: it for it in items}
    return by_id


def _resolve_item(frame: dict, by_id: dict[str, object]):
    stac_id = frame.get("stac_id")
    if not stac_id:
        raise RuntimeError(f"frame {frame.get('path')}: missing stac_id")
    item = by_id.get(stac_id)
    if item is not None:
        return item
    item = _get_item(str(stac_id))
    if item is not None:
        return item
    day = str(frame.get("datetime") or "")[:10]
    mgrs = frame.get("mgrs_tile")
    if day and mgrs:
        items = query_items(
            L2A,
            None,
            f"{day}/{day}",
            max_cloud=100.0,
            max_items=20,
            mgrs_tile=str(mgrs),
        )
        for it in items:
            if it.id == stac_id:
                return it
    raise RuntimeError(f"STAC miss {stac_id}")


def attach_frame(
    s2_dir: Path,
    frame: dict,
    item,
    *,
    include_shadow: bool,
    force: bool,
) -> str:
    rgb_name = str(frame["path"])
    stem = Path(rgb_name).stem
    mask_name = f"{stem}_aoi_cloud.tif"
    scl_name = f"{stem}_scl.tif"
    mask_path = s2_dir / mask_name
    scl_path = s2_dir / scl_name
    rgb_path = s2_dir / rgb_name
    if not rgb_path.is_file():
        raise FileNotFoundError(rgb_path)
    if not force and mask_path.is_file() and scl_path.is_file():
        frame["cloud_mask"] = mask_name
        frame["scl_path"] = scl_name
        frame["mask_source"] = "SCL"
        return "exists"

    with rasterio.open(rgb_path) as src:
        profile = src.profile
        dst_transform = src.transform
        dst_crs = src.crs
        height = int(src.height)
        width = int(src.width)

    scl, scl_profile = read_scl_10m(item)
    scl = reproject_scl_to_grid(
        scl,
        scl_profile,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        height=height,
        width=width,
    )
    cloudy = cloudy_from_scl(scl, include_shadow=include_shadow)
    write_mask(mask_path, cloudy.astype(np.uint8), profile)
    write_mask(scl_path, scl, profile)
    frame["cloud_mask"] = mask_name
    frame["scl_path"] = scl_name
    frame["mask_source"] = "SCL"
    frame["cloud_frac"] = float(cloudy.mean()) if cloudy.size else 1.0
    return "wrote"


def propagate_tiles(parent_dir: Path, parent_meta: dict, extra_roots: list[Path]) -> int:
    """Symlink SCL sidecars into existing LR512 cell dirs and patch their metas."""
    prefix = f"{parent_dir.name}_t"
    parent_by_path = {str(fr.get("path")): fr for fr in parent_meta.get("frames") or []}
    seen: set[Path] = set()
    n = 0
    roots = [parent_dir.parent, *extra_roots]
    for root in roots:
        if not root.is_dir():
            continue
        for dest in sorted(root.glob(f"{prefix}*")):
            dest = dest.resolve()
            if dest in seen or not dest.is_dir():
                continue
            seen.add(dest)
            tmeta_path = dest / "meta.json"
            if not tmeta_path.is_file():
                continue
            tmeta = json.loads(tmeta_path.read_text())
            changed = False
            for fr in tmeta.get("frames") or []:
                pfr = parent_by_path.get(str(fr.get("path")))
                if pfr is None:
                    continue
                for key in ("cloud_mask", "scl_path", "mask_source", "cloud_frac"):
                    if pfr.get(key) is not None and fr.get(key) != pfr.get(key):
                        fr[key] = pfr[key]
                        changed = True
                for key in ("cloud_mask", "scl_path"):
                    name = fr.get(key)
                    if name:
                        src = parent_dir / str(name)
                        if src.is_file():
                            _symlink(src, dest / str(name))
            if changed:
                tmeta_path.write_text(json.dumps(tmeta, indent=2) + "\n")
                n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Parent S2 revisit directory with meta.json + RGB frames.",
    )
    ap.add_argument(
        "--include-shadow",
        action="store_true",
        help="Treat SCL class 3 (cloud shadow) as cloudy. Default off.",
    )
    ap.add_argument("--force", action="store_true", help="Overwrite existing SCL sidecars.")
    ap.add_argument(
        "--propagate-tiles",
        action="store_true",
        help="Symlink masks into sibling {name}_t512_* cell dirs and patch metas.",
    )
    ap.add_argument(
        "--tiles-root",
        type=Path,
        action="append",
        default=None,
        help="Extra root to search for cell dirs (repeatable). Default: src parent.",
    )
    args = ap.parse_args()

    src = args.src if args.src.is_absolute() else ROOT / args.src
    meta_path = src / "meta.json"
    if not meta_path.is_file():
        raise SystemExit(f"missing {meta_path}")
    meta = json.loads(meta_path.read_text())
    frames = list(meta.get("frames") or [])
    if not frames:
        raise SystemExit(f"{meta_path} has no frames")

    include_shadow = bool(args.include_shadow or meta.get("include_shadow"))
    print(f"attach SCL → {src}  frames={len(frames)}", flush=True)
    by_id = _index_items(meta)
    print(f"  STAC index {len(by_id)} items", flush=True)

    n_wrote = n_exist = n_fail = 0
    for fr in frames:
        try:
            item = _resolve_item(fr, by_id)
            status = attach_frame(
                src, fr, item, include_shadow=include_shadow, force=bool(args.force)
            )
            if status == "wrote":
                n_wrote += 1
            else:
                n_exist += 1
            print(
                f"  {status} {fr.get('path')}  cloud_frac={fr.get('cloud_frac')}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            n_fail += 1
            print(f"  FAIL {fr.get('path')}: {exc}", flush=True)

    meta["frames"] = frames
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"meta wrote={n_wrote} exists={n_exist} fail={n_fail} → {meta_path}", flush=True)
    if n_fail:
        raise SystemExit(f"attach failed on {n_fail} frame(s)")

    if args.propagate_tiles:
        extra = []
        if args.tiles_root:
            extra = [p if p.is_absolute() else ROOT / p for p in args.tiles_root]
        n_tiles = propagate_tiles(src, meta, extra)
        print(f"propagated {n_tiles} cell dir(s)", flush=True)


if __name__ == "__main__":
    main()
