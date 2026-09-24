#!/usr/bin/env python3
"""Materialize single-frame and repeated-frame Sentinel-2 control stacks."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s2_dataset import (
    _base_frame_index_for_nib,
    _frame_acquisition_date,
    _parse_iso_date,
)


def resolve_nib_date(meta: dict, explicit: str | date | None = None) -> date:
    """Use the same metadata fallback used by ``S2NIBRevisitDataset``."""
    value = explicit or meta.get("nib_acquisition_date") or meta.get("center_date")
    if value is None:
        frames = list(meta.get("frames") or [])
        if not frames:
            raise ValueError("meta.json has no frames and no NIB date")
        return _frame_acquisition_date(frames[0])
    return _parse_iso_date(value)


def _link_or_copy(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(source, destination)
    else:
        destination.symlink_to(os.path.relpath(source, destination.parent))


def materialize_control(
    source_dir: Path,
    output_dir: Path,
    *,
    repeats: int,
    nib_date: str | date | None = None,
    mode: str = "symlink",
) -> dict:
    """Create a controlled stack containing the NIB-nearest S2 frame T times."""
    source_dir = Path(source_dir).resolve()
    output_dir = Path(output_dir)
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    if mode not in {"symlink", "copy"}:
        raise ValueError("mode must be 'symlink' or 'copy'")
    meta_path = source_dir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"{meta_path} does not exist")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"{output_dir} is not empty")

    meta = json.loads(meta_path.read_text())
    frames = list(meta.get("frames") or [])
    selected_date = resolve_nib_date(meta, nib_date)
    base_index = _base_frame_index_for_nib(frames, selected_date)
    base = frames[base_index]
    output_dir.mkdir(parents=True, exist_ok=True)

    controlled_frames = []
    linked: dict[str, str] = {}
    for i in range(repeats):
        frame = copy.deepcopy(base)
        source_frame = source_dir / base["path"]
        suffix = source_frame.suffix
        frame_name = f"{i + 1:03d}_{_frame_acquisition_date(base):%Y%m%d}{suffix}"
        _link_or_copy(source_frame, output_dir / frame_name, mode)
        linked[frame_name] = str(source_frame)
        frame["index"] = i + 1
        frame["path"] = frame_name
        frame["control_repeat_index"] = i

        cloud_mask = base.get("cloud_mask")
        if cloud_mask:
            source_mask = source_dir / cloud_mask
            if source_mask.is_file():
                mask_name = f"{i + 1:03d}_{_frame_acquisition_date(base):%Y%m%d}_aoi_cloud{source_mask.suffix}"
                _link_or_copy(source_mask, output_dir / mask_name, mode)
                linked[mask_name] = str(source_mask)
                frame["cloud_mask"] = mask_name
        controlled_frames.append(frame)

    out_meta = copy.deepcopy(meta)
    out_meta["frames"] = controlled_frames
    out_meta["out"] = str(output_dir.resolve())
    out_meta["control"] = {
        "kind": "single_base" if repeats == 1 else "repeated_base",
        "repeats": repeats,
        "source_dir": str(source_dir),
        "base_frame_original_index": base_index,
        "base_frame_original_path": base["path"],
        "base_frame_date": _frame_acquisition_date(base).isoformat(),
        "nib_acquisition_date": selected_date.isoformat(),
        "selection": "minimum absolute acquisition-date difference; first metadata frame breaks ties",
        "materialization": mode,
        "linked_files": linked,
    }
    (output_dir / "meta.json").write_text(json.dumps(out_meta, indent=2) + "\n")
    return out_meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Existing data/s2_revisits/<city> directory")
    parser.add_argument("output", type=Path)
    parser.add_argument("--repeats", type=int, default=1, help="T identical base frames")
    parser.add_argument("--nib-date", help="Override NIB date (YYYY-MM-DD)")
    parser.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    args = parser.parse_args()
    meta = materialize_control(
        args.source, args.output, repeats=args.repeats, nib_date=args.nib_date, mode=args.mode
    )
    control = meta["control"]
    print(
        f"Wrote {args.output}: {control['kind']}, T={control['repeats']}, "
        f"base={control['base_frame_original_path']} ({control['base_frame_date']})"
    )


if __name__ == "__main__":
    main()
