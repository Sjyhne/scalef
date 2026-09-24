#!/usr/bin/env python3
"""Build a reproducible command for an external original-SuperF checkout."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

EXPECTED_SOURCE_COMMIT = "dff884c9ed61a6227737537c834a2604792af478"


def build_command(
    checkout: Path,
    srdata: Path,
    *,
    entrypoint: str = "optimize.py",
    dataset_flag: str = "--dataset",
    extra_args: list[str] | None = None,
) -> list[str]:
    checkout = Path(checkout).resolve()
    srdata = Path(srdata).resolve()
    target = checkout / entrypoint
    if not target.is_file():
        raise FileNotFoundError(f"SuperF entrypoint not found: {target}")
    required = ("transform_log.json", "hr_ground_truth.png")
    missing = [name for name in required if not (srdata / name).is_file()]
    if missing:
        raise FileNotFoundError(f"SRData export missing: {', '.join(missing)}")
    return [sys.executable, str(target), dataset_flag, str(srdata), *(extra_args or [])]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    parser.add_argument("srdata", type=Path)
    parser.add_argument("--entrypoint", default="optimize.py")
    parser.add_argument("--dataset-flag", default="--dataset")
    parser.add_argument("--execute", action="store_true", help="Execute; default only prints")
    args, extra = parser.parse_known_args()
    command = build_command(
        args.checkout,
        args.srdata,
        entrypoint=args.entrypoint,
        dataset_flag=args.dataset_flag,
        extra_args=extra,
    )
    print(f"Expected original source commit: {EXPECTED_SOURCE_COMMIT}")
    print(shlex.join(command))
    if args.execute:
        subprocess.run(command, cwd=args.checkout, check=True)


if __name__ == "__main__":
    main()
