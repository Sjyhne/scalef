#!/usr/bin/env python3
"""Score the nested field-size ladder from float outputs with masked metrics and the BOA offset removed.

For every shared LR64 window and every field size (plus bilinear) this records LPIPS, SSIM, and PSNR
against the window-matched reference (the coinciding LR64 child's reference, matched over 640 m) and
the parent-matched reference (the LR512 parent's reference, matched over 5.12 km). SSIM and LPIPS are
averaged only over windows/feature cells inside the evaluation mask (``eval.masked_metrics``).

The fits divided digital numbers by 10000 without removing BOA_ADD_OFFSET. Per-frame standardization
cancels a constant offset, and prediction, bilinear, and both references are expressed in the base
frame's gauge, so removing the offset subtracts 0.1 from all of them for baseline >= 04.00 parents.
Windows whose covering fits chose different base dates are excluded.
"""
from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import lpips
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.modules.setdefault("optimize", types.SimpleNamespace(get_lpips_model=None))
from eval.masked_metrics import mask_bbox_slices, masked_lpips, masked_psnr, masked_ssim  # noqa: E402
from s2_dataset import boa_add_offset  # noqa: E402
from scripts.rescore_nested_parent_reference import (  # noqa: E402
    NEST_ROOT, SIDES, _covering_tile, _index_manifests, _run_dir, crop_for, read,
)
from scripts.run_complete_patch_size_ladder import _project_of_city  # noqa: E402
from scripts.run_nested_float_validation import FULL_MANIFEST, MANIFEST, RUN_TAG  # noqa: E402

RESULTS = ROOT / "paper" / "results"


def base_date(run_dir: Path) -> str:
    return str(json.loads((run_dir / "metrics.json").read_text())["base_frame"]["date"])[:10]


def parent_offset(s2_dir: Path, base_date: str) -> float:
    meta = json.loads((s2_dir / "meta.json").read_text())
    frames = [f for f in meta["frames"] if str(f["datetime"])[:10] == base_date]
    offsets = {boa_add_offset(f["stac_id"]) for f in meta["frames"]}
    if len(offsets) != 1:
        raise ValueError(f"{s2_dir}: stack mixes processing baselines across the offset change")
    if not frames:
        raise ValueError(f"{s2_dir}: no frame on base date {base_date}")
    return offsets.pop()


def tensor(img: np.ndarray, device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).unsqueeze(0).to(device)


def score(fn, pred: np.ndarray, gt: np.ndarray, mask: np.ndarray, device) -> dict:
    y0, y1, x0, x1 = mask_bbox_slices(mask)
    p, g = tensor(pred, device)[:, :, y0:y1, x0:x1], tensor(gt, device)[:, :, y0:y1, x0:x1]
    m = torch.from_numpy(mask[y0:y1, x0:x1]).to(device)
    with torch.no_grad():
        return {"lpips": masked_lpips(fn, p, g, m), "ssim": masked_ssim(p, g, m), "psnr": masked_psnr(p, g, m)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-parents", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    device = torch.device(args.device)
    torch.set_num_threads(16)
    fn = lpips.LPIPS(net="vgg", verbose=False).eval().to(device)
    manifest = FULL_MANIFEST if args.all_parents else MANIFEST
    parents = set(json.loads(manifest.read_text())["parents"]) if not args.all_parents else {
        j["parent_tile_id"] for j in json.loads(manifest.read_text())["jobs"]}
    index = _index_manifests()
    children = [c for c in json.loads((NEST_ROOT / "nested_lr64_manifest.json").read_text())["tiles"]
                if c["parent_tile_id"] in parents]
    out = args.out or RESULTS / f"nested_v4_scores{'_full' if args.all_parents else ''}.json"

    rows, excluded, cache = [], [], {}

    def load(p: Path) -> np.ndarray:
        if p not in cache:
            if len(cache) > 64:
                cache.clear()
            cache[p] = read(p)
        return cache[p]

    for i, child in enumerate(children, 1):
        cov = {s: _covering_tile(child, s, index) for s in SIDES}
        dirs = {s: _run_dir(tt["parent_city"], tt["tile_id"], s, RUN_TAG) for s, tt in cov.items()}
        dates = {s: base_date(d) for s, d in dirs.items()}
        if len(set(dates.values())) != 1:
            excluded.append({"child": child["tile_id"], "parent": child["parent_tile_id"], "base_dates": dates})
            continue
        offset = parent_offset(ROOT / child["s2_dir"], str(dates[64])[:10])
        gt_w = load(dirs[64] / "qgis" / "hr_gt.tif")
        h, w = gt_w.shape[:2]
        gt_p = crop_for(512, child, load(dirs[512] / "qgis" / "hr_gt.tif"))[:h, :w]
        refs = {}
        for name, gt in (("window", gt_w), ("parent", gt_p)):
            mask = np.all(gt > 1e-6, axis=-1)
            refs[name] = (np.where(mask[..., None], gt - offset, 0.0).astype(np.float32), mask)
        preds = {s: crop_for(s, child, load(dirs[s] / "qgis" / "sr_pred.tif"))[:h, :w] - offset for s in SIDES}
        preds["bilinear"] = load(dirs[64] / "qgis" / "s2_bilinear.tif")[:h, :w] - offset
        for s, pred in preds.items():
            row = {"child": child["tile_id"], "parent": child["parent_tile_id"],
                   "project": _project_of_city(child["parent_city"]), "side": s, "boa_offset": offset}
            for name, (gt, mask) in refs.items():
                for k, v in score(fn, pred.astype(np.float32), gt, mask, device).items():
                    row[f"{k}_{name}"] = v
            rows.append(row)
        if i % 100 == 0:
            print(f"{i}/{len(children)}", flush=True)

    out.write_text(json.dumps({"run_tag": RUN_TAG, "manifest": str(manifest.relative_to(ROOT)),
                               "n_parents": len(parents), "n_children": len(children),
                               "excluded_base_date_mismatch": excluded, "rows": rows}, indent=1) + "\n")
    print(f"wrote {out}: {len(rows)} rows, {len(excluded)} windows excluded")


if __name__ == "__main__":
    main()
