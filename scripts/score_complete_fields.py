#!/usr/bin/env python3
"""Score the 85 complete-reference LR512 fields from float outputs under the corrected protocol.

The fields are the LR512 fits of the full-parent nested rerun (``floatval_v3``: K=4 windows with
the blur halo, patience 8, per-parent frozen correction). Each field's HR evaluation mask and
histogram-matched reference are rebuilt on CPU through the production loader from the exact run
command. Prediction, bilinear, and reference have the base frame's BOA_ADD_OFFSET removed, and
SSIM/LPIPS are averaged over valid positions only (eval.masked_metrics).
"""
from __future__ import annotations

import argparse
import json
import shlex
import sys
import types
from pathlib import Path

import numpy as np
import rasterio
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# tinycudann cannot initialise without a GPU; scoring only needs the parser and data loader.
sys.modules.setdefault("tinycudann", types.ModuleType("tinycudann"))
import lpips  # noqa: E402

from data import get_dataset  # noqa: E402
from eval.masked_metrics import compute_masked_image_metrics  # noqa: E402
from optimize import get_argparser  # noqa: E402
from scripts.run_nested_float_validation import FULL_MANIFEST  # noqa: E402

OUT = ROOT / "paper" / "results" / "complete_fields_v2.json"


def load_chw(path: Path) -> torch.Tensor:
    with rasterio.open(path) as src:
        return torch.from_numpy(src.read((1, 2, 3)).astype(np.float32))[None]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=OUT)
    args_cli = ap.parse_args()
    device = torch.device(args_cli.device)
    lpips_fn = lpips.LPIPS(net="vgg", verbose=False).to(device).eval()
    jobs = [j for j in json.loads(FULL_MANIFEST.read_text())["jobs"] if j["side"] == 512]
    rows = []
    for n, job in enumerate(jobs, 1):
        cmd = shlex.split(job["command"]) if isinstance(job["command"], str) else list(job["command"])
        args = get_argparser().parse_args([*cmd[2:], "--s2_boa_offset", "remove"])
        ds = get_dataset(args=args, name=args.dataset, training_device=torch.device("cpu"))
        mask = ds.get_hr_eval_mask().cpu().numpy().astype(bool)
        offset = float(ds.eval_reflectance_offset)
        run = Path(job["metrics"]).parent
        m = json.loads(Path(job["metrics"]).read_text())
        gt = torch.clamp(ds.get_original_hr().permute(2, 0, 1)[None].float(), 0, 1)
        gt = torch.where(gt.amax(dim=1, keepdim=True) > 0, gt - offset, gt)
        mask_t = torch.from_numpy(mask)
        row = {"tile_id": job["tile_id"], "city": args.dataset, "boa_offset": offset,
               "valid_fraction": float(mask.mean()), "time_s": m.get("training_time_seconds"),
               "base_date": (m.get("base_frame") or {}).get("date"),
               "stopped_early": (m.get("checkpoint") or {}).get("stopped_early")}
        pred = load_chw(run / "qgis" / "sr_pred.tif") - offset
        bil = load_chw(run / "qgis" / "s2_bilinear.tif") - offset
        met = compute_masked_image_metrics(pred.to(device), gt.to(device), bil.to(device), mask_t,
                                           device=device, lpips_fn=lpips_fn)
        row.update({f"{k}_{who}": float(met[f"{who}_{k}"]) for k in ("psnr", "ssim", "lpips")
                    for who in ("model", "bilinear")})
        rows.append(row)
        print(f"[{n}/{len(jobs)}] {row['tile_id']} lpips {row['lpips_model']:.4f} bil {row['lpips_bilinear']:.4f}",
              flush=True)
        args_cli.out.write_text(json.dumps({"source_manifest": str(FULL_MANIFEST.relative_to(ROOT)),
                                            "protocol": "surface reflectance, masked SSIM/LPIPS", "rows": rows},
                                           indent=1) + "\n")


if __name__ == "__main__":
    main()
