#!/usr/bin/env python3
"""Discover MuS2 and optionally prepare per-band ScaleF revisit datasets.

Dry-run is the default. Pass ``--execute`` to write GeoTIFF adapters and the
manifest. The script never downloads MuS2; see docs/MUS2.md for the official
Harvard Dataverse archives and their approximately 3.1 GB total size.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.mus2 import (  # noqa: E402
    MUS2_BAND_PAIRS,
    acquisition_date,
    discover_scenes,
    find_mask,
    read_grayscale,
)

PROVENANCE = {
    "dataset": "MuS2: A Benchmark for Sentinel-2 Multi-Image Super-Resolution",
    "dataset_doi": "https://doi.org/10.7910/DVN/1JMRAT",
    "dataset_version": "2.0",
    "dataset_license": "CC0-1.0",
    "dataset_license_url": "https://creativecommons.org/publicdomain/zero/1.0/legalcode",
    "paper_doi": "https://doi.org/10.1038/s41597-023-02538-9",
    "official_code": "https://codeocean.com/capsule/8131193/tree/v2",
    "public_code_mirror": "https://github.com/pk94/WVS2Benchmark",
    "public_code_license": None,
    "code_license_note": "The public benchmark repository contains no license file; no code is copied.",
    "source_sensors": ["Sentinel-2 Level-2A", "WorldView-2 European Cities"],
}


def _write_replicated_geotiff(
    source: Path,
    destination: Path,
    *,
    pixel_size: float,
) -> tuple[int, int]:
    """Write a one-band MuS2 image as a three-channel ScaleF input."""
    import rasterio
    from rasterio.transform import from_origin

    image = read_grayscale(source)
    height, width = image.shape
    destination.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 3,
        "dtype": image.dtype,
        "crs": "EPSG:3857",
        "transform": from_origin(0.0, height * pixel_size, pixel_size, pixel_size),
        "compress": "deflate",
    }
    with rasterio.open(destination, "w", **profile) as dst:
        for index in range(1, 4):
            dst.write(image, index)
    return height, width


def build_manifest(
    source_root: Path,
    output_root: Path,
    *,
    bands: list[str],
    mask_root: Path | None,
    mask_mode: str,
) -> dict:
    """Build a manifest from an extracted official MuS2 dataset."""
    scenes = discover_scenes(source_root, bands)
    records = []
    for scene in scenes:
        band_records = {}
        for band, entry in scene["bands"].items():
            prepared_dir = output_root / scene["id"] / band
            mask = find_mask(mask_root, scene["id"], band, mask_mode)
            band_records[band] = {
                **entry,
                "mask": str(mask) if mask is not None else None,
                "mask_mode": mask_mode if mask is not None else "none",
                "prepared_dir": str(prepared_dir),
                "prepared_hr": str(prepared_dir / "hr_worldview2.tif"),
                "scale_factor": 3,
            }
        records.append(
            {
                "id": scene["id"],
                "source_dir": scene["source_dir"],
                "bands": band_records,
            }
        )
    return {
        "schema": "scalef.mus2-manifest.v1",
        "source_root": str(source_root.resolve()),
        "output_root": str(output_root.resolve()),
        "provenance": PROVENANCE,
        "mask_convention": "MuS2 nonzero/white pixels are excluded from evaluation",
        "scenes": records,
    }


def prepare_manifest(manifest: dict) -> dict:
    """Materialize all manifest records as ScaleF-compatible datasets."""
    for scene in manifest["scenes"]:
        for band, entry in scene["bands"].items():
            output_dir = Path(entry["prepared_dir"])
            lr_files = [Path(path) for path in entry["lr_files"]]
            if not lr_files:
                raise ValueError(f"{scene['id']} {band} has no LR revisits")

            frame_records = []
            lr_shape = None
            for index, source in enumerate(lr_files):
                relative = Path("frames") / f"{index:02d}_{source.stem}.tif"
                shape = _write_replicated_geotiff(
                    source,
                    output_dir / relative,
                    pixel_size=10.0,
                )
                if lr_shape is not None and shape != lr_shape:
                    raise ValueError(
                        f"{scene['id']} {band} LR shape mismatch: {shape} != {lr_shape}"
                    )
                lr_shape = shape
                frame_records.append(
                    {
                        "id": source.stem,
                        "path": str(relative),
                        "datetime": acquisition_date(source) + "T00:00:00+00:00",
                        "source": str(source),
                    }
                )

            hr_shape = _write_replicated_geotiff(
                Path(entry["hr_file"]),
                Path(entry["prepared_hr"]),
                pixel_size=10.0 / 3.0,
            )
            assert lr_shape is not None
            expected_hr = (lr_shape[0] * 3, lr_shape[1] * 3)
            if hr_shape != expected_hr:
                raise ValueError(
                    f"{scene['id']} {band}: HR shape {hr_shape} != 3x LR {expected_hr}"
                )

            meta = {
                "schema": "scalef.s2-revisits.v1",
                "adapter": "MuS2 single-band replicated to RGB",
                "scene_id": scene["id"],
                "mus2_band": band,
                "worldview2_band": f"mul_band_{MUS2_BAND_PAIRS[band]}",
                "resolution_m": 10.0,
                "aoi_window": {
                    "row_off": 0,
                    "col_off": 0,
                    "height": lr_shape[0],
                    "width": lr_shape[1],
                },
                "center_date": frame_records[0]["datetime"][:10],
                "nib_acquisition_date": frame_records[0]["datetime"][:10],
                "frames": frame_records,
                "provenance": manifest["provenance"],
            }
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Extracted MuS2 dataset root.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "mus2_scalef",
        help="Prepared adapter root (large output; gitignored under data/).",
    )
    parser.add_argument("--mask-root", type=Path, default=None)
    parser.add_argument(
        "--mask-mode",
        choices=["none", "final", "relevance", "difference_newest", "perceptual"],
        default="final",
    )
    parser.add_argument(
        "--bands",
        nargs="+",
        choices=sorted(MUS2_BAND_PAIRS),
        default=sorted(MUS2_BAND_PAIRS),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Output manifest path (default: OUTPUT/manifest.json).",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Write prepared rasters and manifest. Without this flag, only print the plan.",
    )
    args = parser.parse_args()

    manifest_path = args.manifest or args.output / "manifest.json"
    manifest = build_manifest(
        args.source,
        args.output,
        bands=args.bands,
        mask_root=args.mask_root or args.source,
        mask_mode=args.mask_mode,
    )
    if not manifest["scenes"]:
        raise SystemExit(
            f"No MuS2 scenes found below {args.source}. Expected scene/hr_resized/mul_band_*.tiff."
        )

    if not args.execute:
        print(json.dumps(manifest, indent=2))
        print(f"DRY RUN: {len(manifest['scenes'])} scenes; pass --execute to write {args.output}")
        return

    prepare_manifest(manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {len(manifest['scenes'])} scenes; manifest={manifest_path}")


if __name__ == "__main__":
    main()
