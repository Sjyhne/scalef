#!/usr/bin/env python3
"""Freeze manifest (v4) for the focused-revision results.

Records the source snapshot (git commit, dirty diff, and a tarball of all Python sources),
per-run arguments and metric hashes for every run namespace used in the revision, the
alignment registry and evaluation manifest, all ``paper/results`` JSONs, and the
generated/static LaTeX tables and figures. The previous manifest is referenced as the parent.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "paper" / "results"
OVERLEAF = ROOT / "ScaleF_Overleaf"

RUN_NAMESPACES = {
    "confirmatory_v5_boa": "single_samples/*/confirmatory_v5_boa/*/metrics.json",
    "confirmatory_v3_halo": "single_samples/*/confirmatory_v3_halo/*/metrics.json",
    "fourier_diag_v1": "single_samples/*/fourier_diag_v1/*/metrics.json",
    "update_eff_v1": "single_samples/*/update_eff_v1/*/metrics.json",
    "nested_floatval_v3": "single_samples/*/sample/prod_*floatval_v3_*/metrics.json",
}
RUN_MANIFESTS = (
    "single_samples/sweep_results/confirmatory_v3_halo/run_manifest.json",
    "single_samples/sweep_results/confirmatory_v2_lr512align/run_manifest.json",
    "single_samples/sweep_results/fourier_diag_v1/run_manifest.json",
    "single_samples/sweep_results/fourier_diag_v1/extension_manifest.json",
    "single_samples/sweep_results/nested_floatval_v3_manifest.json",
)
PROVENANCE = (
    "eval/confirmatory_eval_manifest.v2.json",
    "eval/spatial_alignment.json",
    "data/s2_revisits/aois.json",
    "data/s2_revisits/processing_baselines.json",
)
DIRECT_RUN_MANIFESTS = ("paper/results/run_manifests/confirmatory_v5_boa__run_manifest.json",)
SOURCE_DIRS = ("", "models", "eval", "scripts", "training", "production")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    path = path.resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def file_record(path: Path) -> dict:
    return {"path": rel(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def git(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout


def git_state(repo: Path) -> dict:
    status = git("status", "--porcelain", cwd=repo)
    return {
        "commit": git("rev-parse", "HEAD", cwd=repo).strip(),
        "dirty": bool(status.strip()),
        "n_changed_paths": len(status.splitlines()),
    }


def source_files() -> list[Path]:
    files: list[Path] = []
    for d in SOURCE_DIRS:
        base = ROOT / d
        pattern = "*.py" if d == "" else "**/*.py"
        files.extend(p for p in base.glob(pattern) if p.is_file() and "__pycache__" not in p.parts)
    return sorted(set(files))


def write_source_snapshot(out: Path, files: list[Path]) -> None:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz", compresslevel=9) as tar:
        for p in files:
            info = tar.gettarinfo(str(p), arcname=str(p.relative_to(ROOT)))
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            with p.open("rb") as f:
                tar.addfile(info, f)
    out.write_bytes(buf.getvalue())


def run_records(pattern: str) -> list[dict]:
    records = []
    for m in sorted(ROOT.glob(pattern)):
        data = json.loads(m.read_text())
        prov = data.get("provenance", {})
        records.append({
            "run_dir": str(m.parent.relative_to(ROOT)),
            "metrics_sha256": sha256_file(m),
            "args": prov.get("args"),
            "base_frame": data.get("base_frame"),
        })
    return records


def software() -> dict:
    info = {"python": platform.python_version()}
    try:
        import torch

        info["pytorch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    try:
        from importlib.metadata import version

        info["tinycudann"] = version("tinycudann")
    except Exception:
        pass
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=RESULTS / "MANIFEST.v4.json")
    ap.add_argument("--parent", type=Path, default=RESULTS / "MANIFEST.v3.json")
    ap.add_argument("--snapshot", type=Path, default=RESULTS / "source_snapshot_v4.tar.gz")
    ap.add_argument("--diff", type=Path, default=RESULTS / "source_snapshot_v4.diff")
    args = ap.parse_args()

    srcs = source_files()
    write_source_snapshot(args.snapshot, srcs)
    args.diff.write_text(git("diff", "HEAD", "--", "*.py"))

    run_manifests_dir = RESULTS / "run_manifests"
    run_manifests_dir.mkdir(exist_ok=True)
    copied = []
    for manifest_rel in RUN_MANIFESTS:
        src = ROOT / manifest_rel
        if src.is_file():
            dst = run_manifests_dir / manifest_rel.replace("single_samples/sweep_results/", "").replace("/", "__")
            dst.write_bytes(src.read_bytes())
            copied.append({"source": manifest_rel, **file_record(dst)})
    copied.extend({"source": m, **file_record(ROOT / m)} for m in DIRECT_RUN_MANIFESTS if (ROOT / m).is_file())

    skip = {args.out.name, args.snapshot.name, args.diff.name}
    results = [file_record(p) for p in sorted(RESULTS.glob("*.json"))
               if p.name not in skip and not p.name.startswith("MANIFEST")]
    tex_assets = sorted(
        [*OVERLEAF.glob("generated/tables/*.tex"), *OVERLEAF.glob("generated/figures/*"), *OVERLEAF.glob("tables/*.tex"),
         *OVERLEAF.glob("tables/layout/*.tex")]
    )
    runs = {ns: run_records(pat) for ns, pat in RUN_NAMESPACES.items()}

    manifest = {
        "schema_version": 4,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Freeze for the ScaleF focused-revision diagnostics and regenerated tables/figures.",
        "generator": "scripts/build_freeze_manifest.py",
        "parent_manifest": file_record(args.parent),
        "git": {"code": git_state(ROOT), "paper": git_state(OVERLEAF)},
        "software": software(),
        "source_snapshot": {
            "tarball": file_record(args.snapshot),
            "diff_vs_head": file_record(args.diff),
            "files": {str(p.relative_to(ROOT)): sha256_file(p) for p in srcs},
        },
        "provenance": [file_record(ROOT / p) for p in PROVENANCE if (ROOT / p).is_file()],
        "run_manifests": copied,
        "runs": {ns: {"n": len(r), "records": r} for ns, r in runs.items()},
        "results": results,
        "latex_assets": [
            {"path": str(p.relative_to(OVERLEAF)), "sha256": sha256_file(p), "bytes": p.stat().st_size}
            for p in tex_assets if p.is_file()
        ],
    }
    args.out.write_text(json.dumps(manifest, indent=1, sort_keys=False) + "\n")
    print(f"{rel(args.out)} sha256={sha256_file(args.out)}")
    print({ns: len(r) for ns, r in runs.items()}, "results", len(results), "assets", len(manifest["latex_assets"]))


if __name__ == "__main__":
    main()
