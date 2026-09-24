#!/usr/bin/env python3
"""Rescore saved HR fields onto a shared center window.

Use this to check the bilinear-invariant on historical size-ladder outputs
before a fresh B17 run. It does not replace the confirmatory rerun.

Example
-------
python scripts/rescore_common_footprint.py \\
  --target-hw 256 256 \\
  --pred path/to/lr512_pred.npy --gt path/to/gt.npy --bilinear path/to/bil.npy \\
  --pred path/to/lr64_pred.npy --gt path/to/lr64_gt.npy --bilinear path/to/lr64_bil.npy
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.common_footprint import bilinear_scores_are_invariant, score_common_footprint


def _as_bchw(path: Path) -> torch.Tensor:
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
    else:
        import cv2

        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f"could not read {path}")
        arr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(np.asarray(arr, dtype=np.float32))
    if tensor.ndim == 3 and tensor.shape[0] in (1, 3) and tensor.shape[-1] not in (1, 3):
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 3 and tensor.shape[-1] in (1, 3):
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    elif tensor.ndim != 4:
        raise SystemExit(f"{path}: expected HWC, CHW, or BCHW, got {tuple(tensor.shape)}")
    return tensor


def _as_mask(path: Path | None, hw: tuple[int, int]) -> torch.Tensor:
    if path is None:
        return torch.ones(hw, dtype=torch.bool)
    arr = np.load(path)
    mask = torch.from_numpy(np.asarray(arr, dtype=bool))
    if mask.ndim == 3:
        mask = mask.squeeze(0)
    return mask


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", type=Path, action="append", required=True)
    ap.add_argument("--gt", type=Path, action="append", required=True)
    ap.add_argument("--bilinear", type=Path, action="append", required=True)
    ap.add_argument("--mask", type=Path, action="append", default=None)
    ap.add_argument("--target-hw", type=int, nargs=2, default=None, metavar=("H", "W"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    n = len(args.pred)
    if len(args.gt) != n or len(args.bilinear) != n:
        raise SystemExit("--pred, --gt, and --bilinear must be repeated the same number of times")
    masks = args.mask or [None] * n
    if len(masks) != n:
        raise SystemExit("--mask count must match --pred when provided")

    target_hw = tuple(args.target_hw) if args.target_hw else None
    scores = []
    for pred_path, gt_path, bil_path, mask_path in zip(args.pred, args.gt, args.bilinear, masks, strict=True):
        pred = _as_bchw(pred_path)
        gt = _as_bchw(gt_path)
        bilinear = _as_bchw(bil_path)
        mask = _as_mask(mask_path, (int(gt.shape[-2]), int(gt.shape[-1])))
        score = score_common_footprint(pred, gt, bilinear, mask, target_hw=target_hw)
        score["pred"] = str(pred_path)
        scores.append(score)
        print(
            f"{pred_path.name}: bilinear PSNR {score['bilinear_psnr']:.4f}  "
            f"model PSNR {score['model_psnr']:.4f}  window {score['hr_hw']}"
        )

    if len(scores) >= 2:
        bilinear_scores_are_invariant(scores)
        print("bilinear invariant: ok")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(scores, indent=2) + "\n")


if __name__ == "__main__":
    main()
