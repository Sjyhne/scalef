#!/usr/bin/env python3
"""Audit how many invalid pixels the SSIM/LPIPS bounding box contains for the 17-site runs.

Rebuilds each site's HR evaluation mask on CPU through the production data loader, using
the exact confirmatory_v3_halo command, and compares its valid fraction with metrics.json.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# tinycudann cannot initialise without a GPU; the audit only needs the parser and data loader.
sys.modules.setdefault("tinycudann", types.ModuleType("tinycudann"))
from data import get_dataset  # noqa: E402
from eval.masked_metrics import mask_bbox_slices  # noqa: E402
from optimize import get_argparser  # noqa: E402

MANIFEST = ROOT / "paper" / "results" / "run_manifests" / "confirmatory_v3_halo__run_manifest.json"
OUT = ROOT / "paper" / "results" / "eval_mask_bbox_audit.json"
MASK_DIR = ROOT / "paper" / "results" / "eval_masks_v3"


def find(d, key):
    if isinstance(d, dict):
        if key in d:
            return d[key]
        for v in d.values():
            r = find(v, key)
            if r is not None:
                return r
    return None


def main() -> None:
    jobs = [j for j in json.loads(MANIFEST.read_text())["jobs"] if j["seed"] == 6]
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for job in sorted(jobs, key=lambda j: j["city"]):
        args = get_argparser().parse_args(job["command"][2:])
        ds = get_dataset(args=args, name=args.dataset, training_device=torch.device("cpu"))
        mask = ds.get_hr_eval_mask().cpu().numpy().astype(bool)
        np.save(MASK_DIR / f"{job['city']}.npy", mask)
        y0, y1, x0, x1 = mask_bbox_slices(mask)
        box = mask[y0:y1, x0:x1]
        recorded = find(json.loads(Path(job["expected_metrics_path"]).read_text()), "valid_fraction")
        rows.append({
            "city": job["city"], "valid_fraction": float(mask.mean()), "recorded_valid_fraction": recorded,
            "masked_eval": bool(ds.use_masked_eval), "bbox": [y0, y1, x0, x1],
            "invalid_fraction_in_bbox": float((~box).mean()),
        })
        print(json.dumps(rows[-1]), flush=True)
    OUT.write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
