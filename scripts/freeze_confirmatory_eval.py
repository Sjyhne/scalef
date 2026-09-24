#!/usr/bin/env python3
"""Generate or validate the immutable confirmatory evaluation manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = ROOT / "eval" / "spatial_alignment.json"
DEFAULT_OUTPUT = ROOT / "eval" / "confirmatory_eval_manifest.v1.json"
DEFAULT_OUTPUT_V2 = ROOT / "eval" / "confirmatory_eval_manifest.v2.json"

REQUIRED_CITY_FIELDS = (
    "df",
    "s2_dir_name",
    "lr_size",
    "hr_size",
    "base_frame_index",
    "base_frame_date",
    "nib_acquisition_date",
    "hr_shift_hr_px",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frame_identity(meta: dict[str, Any]) -> str:
    frames = []
    for frame in meta.get("frames") or []:
        frames.append(
            {
                key: frame.get(key)
                for key in ("id", "path", "datetime", "s2:mgrs_tile")
                if frame.get(key) is not None
            }
        )
    payload = json.dumps(frames, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def build_frozen_manifest(
    source: Path = DEFAULT_SOURCE, data_root: Path | None = None
) -> dict[str, Any]:
    """Return the deterministic, minimal eval contract derived from source."""
    raw = json.loads(source.read_text(encoding="utf-8"))
    cities = raw.get("cities")
    if not isinstance(cities, dict) or not cities:
        raise ValueError(f"{source} has no non-empty 'cities' object")

    frozen_cities: dict[str, dict[str, Any]] = {}
    for city in sorted(cities):
        entry = cities[city]
        missing = [field for field in REQUIRED_CITY_FIELDS if field not in entry]
        if missing:
            raise ValueError(f"{source}: city {city!r} missing fields: {', '.join(missing)}")
        frozen = {field: entry[field] for field in REQUIRED_CITY_FIELDS}
        if data_root is not None:
            # Confirmatory comparisons use one LR512 spatial unit even when
            # the original alignment package was a native ~256 px crop.
            dirname = f"{city}_lr512"
            meta_path = data_root / dirname / "meta.json"
            if not meta_path.is_file():
                raise FileNotFoundError(
                    f"{meta_path} missing; materialize LR512 before freezing"
                )
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            frames = list(meta.get("frames") or [])
            if not frames:
                raise ValueError(f"{meta_path} contains no frames")
            tiles = raw.get("tiles") or {}
            tile_entry = tiles.get(dirname) if isinstance(tiles, dict) else None
            if isinstance(tile_entry, dict):
                missing_tile = [field for field in REQUIRED_CITY_FIELDS if field not in tile_entry]
                if missing_tile:
                    raise ValueError(
                        f"{source}: tile {dirname!r} missing fields: {', '.join(missing_tile)}"
                    )
                frozen = {field: tile_entry[field] for field in REQUIRED_CITY_FIELDS}
                frozen["s2_dir_name"] = dirname
            frozen.update(
                {
                    "confirmatory_s2_dir_name": dirname,
                    "requested_frames": min(16, len(frames)),
                    "available_frames": len(frames),
                    "s2_meta_sha256": _sha256(meta_path),
                    "frame_identity_sha256": _frame_identity(meta),
                }
            )
        frozen_cities[city] = frozen

    return {
        "schema": "scalef.confirmatory_eval.v1",
        "source": str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source),
        "source_sha256": _sha256(source),
        "source_version": raw.get("version"),
        "source_method": raw.get("method"),
        "cities": frozen_cities,
    }


def validate_frozen_manifest(
    frozen: Path = DEFAULT_OUTPUT,
    source: Path = DEFAULT_SOURCE,
    data_root: Path | None = None,
) -> dict[str, Any]:
    """Fail if the frozen file is malformed or differs from its source."""
    actual = json.loads(frozen.read_text(encoding="utf-8"))
    expected = build_frozen_manifest(source, data_root=data_root)
    if actual != expected:
        actual_sha = actual.get("source_sha256", "<missing>") if isinstance(actual, dict) else "<invalid>"
        raise ValueError(
            f"{frozen} is stale or edited (records source {actual_sha}, "
            f"current source is {expected['source_sha256']}); regenerate deliberately"
        )
    return actual


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--generate", action="store_true", help="Write the frozen manifest.")
    mode.add_argument("--validate", action="store_true", help="Validate it against the source.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "Frozen manifest path. Keep v1 as historical evidence; write "
            f"{DEFAULT_OUTPUT_V2.name} for the LR512-aligned B17 protocol."
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Optional data/s2_revisits root; freezes LR512 frame identities/counts.",
    )
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    data_root = args.data_root.resolve() if args.data_root is not None else None
    if args.generate:
        manifest = build_frozen_manifest(source, data_root=data_root)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {output} ({len(manifest['cities'])} frozen sites)")
    else:
        manifest = validate_frozen_manifest(output, source, data_root=data_root)
        print(f"Validated {output} ({len(manifest['cities'])} frozen sites)")


if __name__ == "__main__":
    main()
