#!/usr/bin/env python3
"""Cache the Sentinel-2 L2A processing baseline of every frame used by the S2 stacks.

Baseline >= 04.00 products carry BOA_ADD_OFFSET = -1000 DN, which the loader removes. Planetary
Computer items expose the baseline (``s2:processing_baseline``) but not the offset itself.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "s2_revisits" / "processing_baselines.json"


def stac_ids(roots: list[Path]) -> set[str]:
    ids: set[str] = set()
    for root in roots:
        for meta in root.rglob("meta.json"):
            try:
                frames = json.loads(meta.read_text()).get("frames") or []
            except (OSError, json.JSONDecodeError):
                continue
            ids.update(f["stac_id"] for f in frames if f.get("stac_id"))
    return ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="*", type=Path, default=[ROOT / "data" / "s2_revisits"])
    args = ap.parse_args()
    from pystac_client import Client

    cache = json.loads(CACHE.read_text()) if CACHE.is_file() else {}
    todo = sorted(stac_ids(args.roots) - set(cache))
    client = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    for i in range(0, len(todo), 100):
        for item in client.search(collections=["sentinel-2-l2a"], ids=todo[i:i + 100]).items():
            cache[item.id] = item.properties.get("s2:processing_baseline")
    missing = [s for s in todo if s not in cache]
    CACHE.write_text(json.dumps(dict(sorted(cache.items())), indent=1) + "\n")
    print(f"cached {len(cache)} baselines; {len(missing)} ids not found")
    if missing:
        print("\n".join(missing[:20]))


if __name__ == "__main__":
    main()
