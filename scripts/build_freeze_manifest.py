#!/usr/bin/env python3
"""Freeze manifest (v6) for the focused-revision results.

Records the source snapshot (git commit, dirty diff, and a tarball of all Python sources),
per-run arguments and metric hashes for every run namespace used in the revision, the
alignment registry and evaluation manifest, all ``paper/results`` JSONs, the frame-screening
inputs (full-window OmniCloudMask maps, rebuilt stack metadata, replacement frames), and the
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
    "fourier_diag_v2": "single_samples/*/fourier_diag_v2/*/metrics.json",
    "update_eff_v2": "single_samples/*/update_eff_v2/*/metrics.json",
    "figure_float_v5": "single_samples/*/figure_float_v5/*/metrics.json",
    "figure_float_v5b": "single_samples/*/figure_float_v5b/*/metrics.json",
    "nested_floatval_v3": "single_samples/*/sample/prod_*floatval_v3_*/metrics.json",
    "confirmatory_v6_masks": "single_samples/*/confirmatory_v6_masks/*/metrics.json",
    "confirmatory_v6_screen": "single_samples/*/confirmatory_v6_screen/*/metrics.json",
    "confirmatory_v6_repl": "single_samples/*/confirmatory_v6_repl/*/metrics.json",
}
RUN_MANIFESTS = (
    "single_samples/sweep_results/confirmatory_v3_halo/run_manifest.json",
    "single_samples/sweep_results/confirmatory_v2_lr512align/run_manifest.json",
    "single_samples/sweep_results/fourier_diag_v1/run_manifest.json",
    "single_samples/sweep_results/fourier_diag_v1/extension_manifest.json",
    "single_samples/sweep_results/fourier_diag_v2/run_manifest.json",
    "single_samples/sweep_results/fourier_diag_v2/extension_manifest.json",
    "single_samples/sweep_results/nested_floatval_v3_full_manifest.json",
    "single_samples/sweep_results/nested_floatval_v3_manifest.json",
)
PROVENANCE = (
    "eval/confirmatory_eval_manifest.v2.json",
    "eval/spatial_alignment.json",
    "data/s2_revisits/aois.json",
    "data/s2_revisits/processing_baselines.json",
)
DIRECT_RUN_MANIFESTS = (
    "paper/results/run_manifests/confirmatory_v5_boa__run_manifest.json",
    "paper/results/run_manifests/confirmatory_v6_masks__run_manifest.json",
    "paper/results/run_manifests/confirmatory_v6_screen__run_manifest.json",
    "paper/results/run_manifests/confirmatory_v6_repl__run_manifest.json",
)
SCREENING_INPUTS = (
    "data/s2_revisits/*_lr512/*_lr512_ocm.tif",
    "data/s2_revisits_screen_v1/*_lr512/meta.json",
    "data/s2_revisits_screen_v1/*_lr512/r*.tif",
)
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


CLEAR_PIXEL_RESULTS = (
    "paper/results/b0_17_v7_clear_summary.json",
    "paper/results/b0_17_v7_clear_paired.json",
    "paper/results/b0_17_v7_clear_ssim.json",
    "paper/results/complete_fields_clear_v4.json",
    "paper/results/nested_v4_scores_clear_v4.json",
)
CLEAR_PIXEL_RUNS = "single_samples/*/confirmatory_v7_clear/*/metrics.json"


def _correction(tiles: dict, stack_name: str) -> dict | None:
    rec = tiles.get(stack_name)
    if not rec:
        return None
    return {
        "registry_key": stack_name,
        "method": "hr_phase_corr + local_hr_mse_refine",
        "hr_shift_hr_px": rec.get("hr_shift_hr_px"),
        "base_frame_date": rec.get("base_frame_date"),
        "nib_acquisition_date": rec.get("nib_acquisition_date"),
        "valid_fraction": rec.get("valid_fraction"),
    }


def _stack_record(s2_dir: Path, screen: dict | None, tiles: dict) -> dict:
    meta_path = s2_dir / "meta.json"
    meta = json.loads(meta_path.read_text())
    suffix = (screen or {}).get("cloud_mask_suffix") or ""
    frames = []
    for fr in meta.get("frames") or []:
        stem = Path(fr["path"]).stem
        ocm = s2_dir / f"{stem}{suffix}.tif" if suffix else None
        frames.append({
            "date": str(fr.get("datetime") or "")[:10],
            "stac_id": fr.get("stac_id"),
            "mgrs_tile": fr.get("mgrs_tile") or meta.get("mgrs_tile"),
            "path": fr.get("path"),
            "ocm": file_record(ocm) if ocm is not None and ocm.is_file() else None,
        })
    return {
        "s2_dir": rel(s2_dir),
        "meta": file_record(meta_path),
        "spatial_correction": _correction(tiles, s2_dir.name),
        "frame_screen": None if screen is None else {
            "n_frames": screen.get("n_frames"),
            "n_excluded": screen.get("n_excluded"),
            "exclude": screen.get("exclude") or [],
            "cloud_mask_suffix": suffix,
        },
        "frames": frames,
    }


def _production_audit() -> dict:
    summary_path = ROOT / "single_samples/sweep_results/production_32VNM_lr512.json"
    mosaic_side = ROOT / "production/mosaics/32VNM_icm_ramp_sr_2p5m.tif.json"
    mosaic = ROOT / "production/mosaics/32VNM_icm_ramp_sr_2p5m.tif"
    summary = json.loads(summary_path.read_text())
    side = json.loads(mosaic_side.read_text())
    tiles = summary["tiles"]
    times = [t["training_time_s"] for t in tiles if t.get("training_time_s") is not None]
    layout = None
    if mosaic.is_file():
        import rasterio

        with rasterio.open(mosaic) as src:
            layout = src.tags(ns="IMAGE_STRUCTURE").get("LAYOUT")
            overviews = src.overviews(1)
    else:
        overviews = []
    return {
        "mosaic": {
            "path": rel(mosaic),
            "bytes": mosaic.stat().st_size if mosaic.is_file() else None,
            "layout": layout,
            "overviews": overviews,
            "sidecar": side,
            "sidecar_predates_cog_rewrite": True,
        },
        "training_summary": file_record(summary_path),
        "n_tiles": len(tiles),
        "n_ok": summary["aggregate"]["n_ok"],
        "n_fail": summary["aggregate"]["n_fail"],
        "n_skipped_on_that_invocation": summary["aggregate"]["n_skipped"],
        "sum_recorded_train_s": sum(times),
        "mean_recorded_train_s": (sum(times) / len(times)) if times else None,
        "created_utc": summary.get("created_utc"),
        "manifest": summary.get("manifest"),
        "figure_provenance": (
            "Paper qualitative panels are the confirmatory_v7_clear float GeoTIFFs, "
            "not crops of this mosaic."
        ),
        "intentional_differences_from_clear_pixel_benchmark": {
            "reference": "No NIB orthophoto, so no per-field spatial correction and no histogram match.",
            "reflectance": "Fits keep the processing-baseline +0.1 offset. The benchmark removes it at export and score time.",
            "statistics": "Per-frame mean and std use every valid pixel. The benchmark uses clear pixels only.",
            "masks": "Loss masks are the stack SCL cloud masks. The benchmark uses full-field OmniCloudMask thick+thin+shadow.",
            "identity": "One base date per cell, then a 128 HR-pixel ramp on date cuts (90 edges). The benchmark pins each field's own base date and does not mosaic.",
            "geometry": "438 abutting 512-pixel cells, no overlap. The settled production join is a 5% overlap with a midpoint cut.",
        },
    }


def _national_prep() -> dict:
    pilot = ROOT / "production/32VNM_ovl10/training_run.json"
    pilot_agg = json.loads(pilot.read_text())["aggregate"] if pilot.is_file() else {}
    mgrs_path = ROOT / "production/cloud_availability/norway_mgrs_list.json"
    mgrs = json.loads(mgrs_path.read_text()) if mgrs_path.is_file() else []
    if isinstance(mgrs, dict):
        mgrs = list(mgrs.get("mgrs") or mgrs.get("tiles") or [])
    stacks = ROOT / "data/s2_revisits/national_2025_v2"
    present = [m for m in mgrs if (stacks / m / "meta.json").is_file()]
    return {
        "status": "not_started",
        "recipe": {
            "overlap_frac": 0.05,
            "stride_lr_px": 486,
            "overlap_lr_px": 26,
            "join": "midpoint hard cut",
            "cloud": "OmniCloudMask thick+thin+shadow, admit thick+thin <= 15% and valid >= 85%",
            "identity": "primary 2025-07-12 with 5% neighbour slack",
            "export": "BOA +0.1 removed, COG",
            "reference_alignment": "none; production has no orthophoto",
        },
        "pilot_32VNM_10pct_measured": {
            "summary": rel(pilot) if pilot.is_file() else None,
            "n_ok": pilot_agg.get("n_ok"),
            "n_fail": pilot_agg.get("n_fail"),
            "mean_train_s": pilot_agg.get("mean_train_s"),
            "sum_train_s": pilot_agg.get("sum_train_s"),
        },
        "pilot_32VNM_5pct_grid": {
            "full_grid": "23x23 = 529",
            "land_cells": 524,
            "source": "make_granule_tiles.py --overlap-frac 0.05 --mainland-only --dry-run",
        },
        "unique_reconstructed_area": (
            "Not the sum of window areas. Overlap is discarded at the midpoint, so the "
            "delivered area is the union of the land-touching windows. On 32VNM that union "
            "is the granule footprint that meets the mainland outline."
        ),
        "acquisition": {
            "mgrs_list": rel(mgrs_path) if mgrs_path.is_file() else None,
            "n_listed": len(mgrs),
            "n_with_local_stack": len(present),
            "missing_stacks": [m for m in mgrs if m not in set(present)],
        },
        "reconstruction": {
            "started": False,
            "failures": 0,
            "retries": 0,
            "seconds_per_cell_from_10pct_pilot": pilot_agg.get("mean_train_s"),
        },
        "delivery": {"mosaic": None, "cog": True},
    }


def write_clear_pixel_freeze(out: Path) -> None:
    """Archive the clear-pixel benchmark with masks, acquisitions, and per-field shifts.

    The fits record ``dbe9f23`` with a dirty tree. Later edits to this checkout
    are listed so the manifest is not read as a byte-identical source archive.
    """
    metrics = sorted(ROOT.glob(CLEAR_PIXEL_RUNS))
    if len(metrics) != 51:
        raise SystemExit(f"expected 51 confirmatory_v7_clear metrics, found {len(metrics)}")
    commits = set()
    newest = 0.0
    records = []
    for path in metrics:
        data = json.loads(path.read_text())
        git_info = (data.get("provenance") or {}).get("git") or {}
        commits.add((git_info.get("commit"), bool(git_info.get("dirty"))))
        newest = max(newest, path.stat().st_mtime)
        args = (data.get("provenance") or {}).get("args") or {}
        records.append({
            "run_dir": str(path.parent.relative_to(ROOT)),
            "metrics_sha256": sha256_file(path),
            "lr_stats_pixels": args.get("lr_stats_pixels"),
            "hr_harmonize_pixels": args.get("hr_harmonize_pixels"),
            "s2_boa_offset": args.get("s2_boa_offset"),
            "lr_tile_halo": args.get("lr_tile_halo"),
        })
    parent = RESULTS / "MANIFEST.v6.json"
    later = []
    for rel_path in (
        "optimize.py", "s2_dataset.py", "eval/export_geotiff.py",
        "scripts/build_complete_tiles_table.py", "scripts/mosaic_seam_ramp.py",
        "scripts/render_paper_figures.py", "scripts/run_nested_float_validation.py",
        "scripts/run_production.py", "scripts/run_screened_refits.py",
        "scripts/score_complete_fields.py", "scripts/score_nested_v4.py",
        "scripts/rescore_clear_pixels.py", "scripts/site_retarget_nib.py",
    ):
        path = ROOT / rel_path
        if path.is_file() and path.stat().st_mtime > newest + 60:
            later.append(rel_path)
    alignment = json.loads((ROOT / "eval/spatial_alignment.json").read_text())
    tiles = alignment.get("tiles") or {}
    named_screen = json.loads((ROOT / "paper/results/frame_screen_v1_named_masks_only.json").read_text())["stacks"]
    field_screen = json.loads((ROOT / "paper/results/frame_screen_nested_masks_only.json").read_text())["stacks"]
    by_city: dict[str, list[dict]] = {}
    for rec in records:
        city = Path(rec["run_dir"]).parts[1]
        by_city.setdefault(city, []).append(rec)
    named_fields = []
    missing_named = []
    for city, runs in sorted(by_city.items()):
        metrics = json.loads((ROOT / runs[0]["run_dir"] / "metrics.json").read_text())
        s2_dir = Path((metrics.get("provenance") or {}).get("args", {}).get("s2_dir"))
        stack = _stack_record(s2_dir, named_screen.get(s2_dir.name), tiles)
        if stack["spatial_correction"] is None:
            missing_named.append(s2_dir.name)
        named_fields.append({"city": city, "seeds": runs, "base_frame": metrics.get("base_frame"), **stack})

    full = json.loads((ROOT / "single_samples/sweep_results/nested_clear_v4_full_manifest.json").read_text())
    lr512_jobs = [j for j in full["jobs"] if j.get("side") == 512]
    complete = []
    missing_fields = []
    for job in lr512_jobs:
        cmd = job["command"] if isinstance(job["command"], str) else " ".join(job["command"])
        s2_dir = Path(cmd.split("--s2-dir", 1)[1].split()[0])
        metrics_path = Path(job["metrics"])
        stack = _stack_record(s2_dir, field_screen.get(s2_dir.name), tiles)
        if stack["spatial_correction"] is None:
            missing_fields.append(job["tile_id"])
        complete.append({
            "tile_id": job["tile_id"],
            "parent_tile_id": job.get("parent_tile_id"),
            "metrics": file_record(metrics_path) if metrics_path.is_file() else {"path": str(metrics_path), "missing": True},
            "force_base_date": cmd.split("--force_base_date", 1)[1].split()[0] if "--force_base_date" in cmd else None,
            **stack,
        })
    parents = json.loads((ROOT / "single_samples/sweep_results/nested_floatval_v3_manifest.json").read_text())["parents"]
    parent_ids = set(parents)
    manifest = {
        "schema_version": 7,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Clear-pixel benchmark archive: 17 named sites, 85 complete LR512 fields, "
            "and the 11 nested parents, with masks, acquisition metadata, and the "
            "independent spatial correction for each LR512 field. Not a clean source archive."
        ),
        "generator": "scripts/build_freeze_manifest.py --clear-pixel",
        "parent_manifest": file_record(parent) if parent.is_file() else None,
        "spatial_alignment": file_record(ROOT / "eval/spatial_alignment.json"),
        "runs_recorded_git": [{"commit": c, "dirty": d} for c, d in sorted(commits)],
        "protocol_flags_uniform": {
            "lr_stats_pixels": sorted({r["lr_stats_pixels"] for r in records}),
            "hr_harmonize_pixels": sorted({r["hr_harmonize_pixels"] for r in records}),
            "s2_boa_offset": sorted({r["s2_boa_offset"] for r in records}),
            "lr_tile_halo": sorted({r["lr_tile_halo"] for r in records}),
        },
        "checkout_at_manifest": git_state(ROOT),
        "source_files_modified_after_metrics": later,
        "results": [file_record(ROOT / p) for p in CLEAR_PIXEL_RESULTS if (ROOT / p).is_file()],
        "named_sites": {"n_sites": len(named_fields), "n_runs": len(records), "fields": named_fields},
        "complete_fields": {
            "n": len(complete),
            "source_manifest": "single_samples/sweep_results/nested_clear_v4_full_manifest.json",
            "run_tag": full.get("run_tag"),
            "fields": complete,
        },
        "nested_parents": {
            "n": len(parents),
            "scores": "paper/results/nested_v4_scores_clear_v4.json",
            "run_tag": "clear_v4",
            "parent_ids": parents,
            "fields": [f for f in complete if f["tile_id"] in parent_ids],
        },
        "missing_spatial_corrections": {"named_sites": missing_named, "complete_fields": missing_fields},
    }
    out.write_text(json.dumps(manifest, indent=1) + "\n")
    audit_path = RESULTS / "production_audit_32VNM.json"
    audit_path.write_text(json.dumps(_production_audit(), indent=1) + "\n")
    prep_path = RESULTS / "national_production_prep.json"
    prep_path.write_text(json.dumps(_national_prep(), indent=1) + "\n")
    print(
        f"{rel(out)} sites={len(named_fields)} fields={len(complete)} "
        f"parents={len(parents)} missing_corrections={len(missing_named)+len(missing_fields)}"
    )
    print(f"audit {rel(audit_path)}")
    print(f"prep {rel(prep_path)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clear-pixel", action="store_true",
                    help="Write the clear-pixel result freeze instead of MANIFEST.v6.")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--parent", type=Path, default=RESULTS / "MANIFEST.v5.json")
    ap.add_argument("--snapshot", type=Path, default=RESULTS / "source_snapshot_v6.tar.gz")
    ap.add_argument("--diff", type=Path, default=RESULTS / "source_snapshot_v6.diff")
    args = ap.parse_args()
    if args.clear_pixel:
        write_clear_pixel_freeze(args.out or (RESULTS / "MANIFEST.v7.json"))
        return
    if args.out is None:
        args.out = RESULTS / "MANIFEST.v6.json"

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
    results = [file_record(p) for p in sorted([*RESULTS.glob("*.json"), *RESULTS.glob("ssim_failure/*"),
                                               *RESULTS.glob("screened_refits/*.json")])
               if p.name not in skip and not p.name.startswith("MANIFEST")]
    screening = [file_record(p) for pat in SCREENING_INPUTS for p in sorted(ROOT.glob(pat)) if p.is_file()]
    tex_assets = sorted(
        [*OVERLEAF.glob("generated/tables/*.tex"), *OVERLEAF.glob("generated/figures/*"), *OVERLEAF.glob("tables/*.tex"),
         *OVERLEAF.glob("tables/layout/*.tex")]
    )
    runs = {ns: run_records(pat) for ns, pat in RUN_NAMESPACES.items()}

    manifest = {
        "schema_version": 6,
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
        "screening_inputs": screening,
        "latex_assets": [
            {"path": str(p.relative_to(OVERLEAF)), "sha256": sha256_file(p), "bytes": p.stat().st_size}
            for p in tex_assets if p.is_file()
        ],
    }
    args.out.write_text(json.dumps(manifest, indent=1, sort_keys=False) + "\n")
    print(f"{rel(args.out)} sha256={sha256_file(args.out)}")
    print({ns: len(r) for ns, r in runs.items()}, "results", len(results), "screening", len(screening),
          "assets", len(manifest["latex_assets"]))


if __name__ == "__main__":
    main()
