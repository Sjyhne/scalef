#!/usr/bin/env python3
"""Generate (or explicitly execute) matched Phase 2 control commands."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build_commands(
    *,
    city: str,
    source_dir: Path,
    single_dir: Path,
    repeated_dir: Path,
    repeats: int,
    seeds: list[int],
    device: int,
    iters: int,
    extra_args: list[str] | None = None,
) -> dict:
    variants = {
        "full_revisits": Path(source_dir),
        "single_base": Path(single_dir),
        f"repeated_base_t{repeats}": Path(repeated_dir),
    }
    commands = []
    for seed in seeds:
        for variant, s2_dir in variants.items():
            run_name = f"phase2_{variant}_{city}_seed{seed}"
            command = [
                sys.executable,
                str(ROOT / "optimize.py"),
                "--dataset", city,
                "--s2-dir", str(s2_dir),
                "--num_samples", str(repeats if variant.startswith("repeated") else 1)
                if variant != "full_revisits" else "0",
                "--run_name", run_name,
                "--device", str(device),
                "--iters", str(iters),
                "--seed", str(seed),
                "--no_qgis_export",
            ]
            command.extend(extra_args or [])
            commands.append({"variant": variant, "seed": seed, "command": command})
    return {
        "city": city,
        "commands": commands,
        "frozen_alignment_control": {
            "status": "blocked",
            "reason": (
                "optimize.py has no flag to freeze every per-frame affine. "
                "--no_base_frame does not provide a global frozen-alignment control; "
                "adding that flag must be coordinated with the optimize.py owner."
            ),
            "required_flag_contract": "--freeze_all_affines (or equivalent)",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("city")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--single-dir", type=Path, required=True)
    parser.add_argument("--repeated-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--iters", type=int, default=5000)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--execute", action="store_true", help="Run GPU jobs; default is command-only")
    args, extra = parser.parse_known_args()
    manifest = build_commands(
        city=args.city,
        source_dir=args.source_dir,
        single_dir=args.single_dir,
        repeated_dir=args.repeated_dir,
        repeats=args.repeats,
        seeds=args.seeds,
        device=args.device,
        iters=args.iters,
        extra_args=extra,
    )
    text = json.dumps(manifest, indent=2) + "\n"
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(text)
    for job in manifest["commands"]:
        print(shlex.join(job["command"]))
        if args.execute:
            subprocess.run(job["command"], cwd=ROOT, check=True)
    print("BLOCKED frozen alignment:", manifest["frozen_alignment_control"]["reason"])


if __name__ == "__main__":
    main()
