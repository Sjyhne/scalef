#!/usr/bin/env python3
"""Estimate one independent HR-eval correction per LR512 patch manifest row."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_hr_lr_spatial_alignment import (  # noqa: E402
    audit_s2_dir,
    merge_into_spatial_alignment,
)

DEFAULT_MANIFEST = (
    ROOT
    / "data"
    / "s2_revisits"
    / "map"
    / "patch_grid_lr512"
    / "complete_patch_tiles_manifest.json"
)
DEFAULT_OUT_DIR = ROOT / "single_samples" / "spatial_audit" / "lr512_patch_fields"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _audit_one(
    tile: dict,
    *,
    refine_radius_hr: float,
    refine_step_hr: float,
    phase_upsample: int,
    manifest_sha256: str,
) -> dict:
    s2_dir = ROOT / tile["s2_dir"]
    rec = audit_s2_dir(
        s2_dir,
        city=str(tile["parent_city"]),
        refine_radius_hr=refine_radius_hr,
        refine_step_hr=refine_step_hr,
        phase_upsample=phase_upsample,
    )
    rec.update(
        {
            "tile_id": str(tile["tile_id"]),
            "parent_city": str(tile["parent_city"]),
            "alignment_scope": "lr512_patch_exact",
            "status": "accepted",
            "input_manifest_sha256": manifest_sha256,
            "s2_meta_sha256": _sha256(s2_dir / "meta.json"),
        }
    )
    return rec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--merge-into", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--refine-radius-hr", type=float, default=2.0)
    parser.add_argument("--refine-step-hr", type=float, default=0.25)
    parser.add_argument("--phase-upsample", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--allow-failed",
        action="store_true",
        help="Merge accepted records while retaining failed rows in the audit summary.",
    )
    args = parser.parse_args()

    manifest = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    payload = json.loads(manifest.read_text())
    tiles = list(payload["tiles"])
    manifest_sha256 = _sha256(manifest)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    pending: list[dict] = []
    for tile in tiles:
        report_path = out_dir / f"{tile['tile_id']}_spatial_alignment.json"
        if args.resume and report_path.is_file():
            results.append(json.loads(report_path.read_text()))
        else:
            pending.append(tile)

    failures: list[dict] = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        future_to_tile = {
            pool.submit(
                _audit_one,
                tile,
                refine_radius_hr=args.refine_radius_hr,
                refine_step_hr=args.refine_step_hr,
                phase_upsample=args.phase_upsample,
                manifest_sha256=manifest_sha256,
            ): tile
            for tile in pending
        }
        for future in as_completed(future_to_tile):
            tile = future_to_tile[future]
            try:
                rec = future.result()
            except Exception as exc:  # noqa: BLE001
                failure = {"tile_id": tile["tile_id"], "error": repr(exc)}
                failures.append(failure)
                print(f"FAIL {tile['tile_id']}: {exc}", flush=True)
                continue
            results.append(rec)
            (out_dir / f"{tile['tile_id']}_spatial_alignment.json").write_text(
                json.dumps(rec, indent=2) + "\n"
            )
            shift = rec["shift_apply_to_hr_for_eval"]
            print(
                f"{len(results):3d}/{len(tiles)} {tile['tile_id']:28s} "
                f"dy={shift['dy']:+.2f} dx={shift['dx']:+.2f}",
                flush=True,
            )

    results.sort(key=lambda row: row["tile_id"])
    summary = {
        "schema": "scalef.patch_alignment_audit.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(manifest.relative_to(ROOT)),
        "source_manifest_sha256": manifest_sha256,
        "source_git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "estimator": {
            "method": "hr_phase_corr + local_hr_mse_refine",
            "phase_upsample": args.phase_upsample,
            "refine_radius_hr_px": args.refine_radius_hr,
            "refine_step_hr_px": args.refine_step_hr,
        },
        "n_expected": len(tiles),
        "n_accepted": len(results),
        "n_failed": len(failures),
        "failures": failures,
        "results": results,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {summary_path}: {len(results)}/{len(tiles)} accepted", flush=True)

    if (failures or len(results) != len(tiles)) and not args.allow_failed:
        raise SystemExit("not merging: audit is incomplete")
    if args.merge_into is not None:
        merge_path = args.merge_into if args.merge_into.is_absolute() else ROOT / args.merge_into
        merge_into_spatial_alignment(
            results,
            alignment_path=merge_path,
            refine_radius_hr=args.refine_radius_hr,
            refine_step_hr=args.refine_step_hr,
        )


if __name__ == "__main__":
    main()
