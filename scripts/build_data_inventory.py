#!/usr/bin/env python3
"""Data and evaluation inventory for the paper (JSON + appendix tables).

Everything is read from the frozen inputs: ``data/s2_revisits/aois.json``,
per-stack ``meta.json``, the NIB package manifests, the confirmatory v2
manifest and the nested patch-grid manifests. Nested exclusions are recomputed
from the saved ``metrics.json`` files of the lr512align_v2 runs.
Optionally (``--query-stac``) records ``s2:processing_baseline`` per frame.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s2_dataset import DEFAULT_MAX_BASE_CLOUD_FRAC  # noqa: E402

S2 = ROOT / "data" / "s2_revisits"
NEST = S2 / "map" / "patch_grid_nested"
V2_MANIFEST = ROOT / "eval" / "confirmatory_eval_manifest.v2.json"
NIB_MANIFESTS = (ROOT / "data/nib_focus_1m_worldcover/package_manifest.csv",
                 ROOT / "data/nib_new_1m_worldcover/package_manifest.csv")
SITE_ORDER = ("asker", "vennesla", "trondheim", "bergen", "rana", "tromso", "amli", "stavanger", "algard",
              "naerbo", "flekkefjord", "rafsbotn", "nittedal", "melhus", "alta", "karasjok", "kautokeino")
LABEL = {"tromso": r"Troms{\o}", "amli": r"{\AA}mli", "algard": r"{\AA}lg{\aa}rd", "naerbo": r"N{\ae}rb{\o}"}
BASELINE_04_START = date(2022, 1, 25)

PROCESSING = {
    "s2_source": "Microsoft Planetary Computer STAC, collection sentinel-2-l2a (MGRS tile, 10 m)",
    "s2_bands": ["B04", "B03", "B02"],
    "s2_scaling": "L2A digital numbers divided by 10000, NaN->0, clipped to [0, 1.5]; "
                  "BOA_ADD_OFFSET is not applied (see processing_baseline)",
    "cloud_method": "OmniCloudMask on the AOI window; clear = not thick, thin or shadow",
    "frame_admission": {"max_cloud_frac": 0.15, "min_valid_frac": 0.85, "max_stac_cloud": 100.0,
                        "search_window_days": "+-90 around the NIB date (+-150 for two sparse retries)",
                        "frames_requested": 16},
    "base_frame_rule": f"closest in date to the NIB date among frames with cloud fraction <= "
                       f"{DEFAULT_MAX_BASE_CLOUD_FRAC}; else clearest, then closest",
    "lr_standardization": "per frame and band, mean/std over all pixels of the LR window "
                          "(--lr_stats_pixels all); outputs destandardized with base-frame statistics",
    "nib_date_rule": "norgeibilder photoDate of the newest non-CIR RGB project covering the AOI "
                     "(pixel size <= 0.25 m, bbox cover >= 0.80); focus exports use the named NIB project; "
                     "the export XML <Dato> is the export date and is not used",
    "nib_resampling": "rasterio reproject, bilinear, 1 m NIB -> 2.5 m S2-aligned grid (df = 4); "
                      "coverage by nearest; 8-bit values divided by 255",
    "hr_colour_harmonization": "per-band histogram matching of NIB to the base S2 frame "
                               "(skimage match_histograms, NIB zeros kept as nodata, clipped to [0, 1])",
    "hr_spatial_alignment": "per-field frozen sub-pixel shift of the reference (hr_phase_corr + "
                            "local_hr_mse_refine; eval/spatial_alignment.json); children inherit their "
                            "parent's shift; applied to reference and mask only",
    "hr_eval_mask": "NIB coverage AND valid S2 (upsampled) AND all RGB > 0, eroded by 1 HR pixel",
    "metrics": {"psnr": "masked MSE over mask pixels, data range 1",
                "lpips": "VGG LPIPS on the mask bounding box (<= 2048 px, centre crop), inputs x*2-1",
                "prediction": "denormalized with base-frame mean/std; not clipped",
                "bilinear": "cv2 INTER_LINEAR of the unstandardized base frame, clipped to [0, 1]"},
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def nib_exports() -> dict[str, dict]:
    out = {}
    for path in NIB_MANIFESTS:
        with path.open() as fh:
            for row in csv.DictReader(fh):
                archive = row.get("export_archive_id") or row.get("source_archive") or ""
                out[row["project_folder"]] = {
                    "export_id": archive.removesuffix(".zip").removesuffix("_1").removeprefix("eksport_"),
                    "export_archive": archive,
                    "package_label": row.get("project") or row.get("normalized_archive"),
                    "valid_area_km2": float(row.get("valid_area_km2") or row.get("valid_area_km2_approx") or 0),
                    "package": path.parent.name,
                }
    return out


def frame_date(frame: dict) -> str:
    return str(frame["datetime"])[:10]


def processing_baselines(stac_ids: list[str]) -> dict[str, str | None]:
    import planetary_computer as pc  # noqa: F401
    from pystac_client import Client

    client = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    out: dict[str, str | None] = {}
    for i in range(0, len(stac_ids), 100):
        chunk = stac_ids[i:i + 100]
        for item in client.search(collections=["sentinel-2-l2a"], ids=chunk).items():
            out[item.id] = item.properties.get("s2:processing_baseline")
    return {sid: out.get(sid) for sid in stac_ids}


def site_record(city: str, area: dict, v2: dict, exports: dict) -> dict:
    meta_path = S2 / f"{city}_lr512" / "meta.json"
    meta = json.loads(meta_path.read_text())
    frames = meta["frames"]
    skipped = meta.get("skipped") or []
    reasons: dict[str, int] = {}
    for s in skipped:
        reasons[s.get("reason", "unknown")] = reasons.get(s.get("reason", "unknown"), 0) + 1
    project = area.get("project_folder")
    nib_date = area["date"]
    base_date = v2["base_frame_date"]
    run_metrics = sorted(ROOT.glob(f"single_samples/{city}/confirmatory_v3_halo/*__{city}__seed6/metrics.json"))
    base_cloud = (json.loads(run_metrics[0].read_text())["base_frame"]["cloud_frac"] if run_metrics else None)
    return {
        "site": city,
        "nib": {
            "package": area.get("nib_package"),
            "project_folder": project,
            "nib_project": area.get("nib_project"),
            **{k: v for k, v in exports.get(project, {}).items() if k != "package"},
            "date": nib_date,
            "date_provenance": area.get("acquisition_date_note"),
            "date_uncertainty": "photoDate is a project-level date; individual NIB strips may be flown on "
                                "other days of the campaign",
        },
        "s2": {
            "mgrs_tile": meta.get("mgrs_tile"),
            "crs": meta.get("crs"),
            "date_range": meta.get("date_range"),
            "aoi_window_lr512": meta["aoi_window"],
            "center_lonlat": [area.get("center_lon"), area.get("center_lat")],
            "center_kind": area.get("center_kind"),
            "n_frames": len(frames),
            "n_skipped": len(skipped),
            "skipped_by_reason": reasons,
            "frames": [
                {"stac_id": f["stac_id"], "date": frame_date(f), "eo_cloud_cover": f.get("eo:cloud_cover"),
                 "aoi_cloud_frac": f.get("cloud_frac"), "aoi_valid_frac": f.get("valid_frac")}
                for f in frames
            ],
            "first_date": min(frame_date(f) for f in frames),
            "last_date": max(frame_date(f) for f in frames),
        },
        "evaluation": {
            "base_frame_date": base_date,
            "base_minus_nib_days": (date.fromisoformat(base_date) - date.fromisoformat(nib_date)).days,
            "base_frame_lr_cloud_frac": base_cloud,
            "hr_shift_hr_px": v2.get("hr_shift_hr_px"),
            "lr_size": v2.get("lr_size"),
            "hr_size": v2.get("hr_size"),
            "s2_meta_sha256": v2.get("s2_meta_sha256"),
            "frame_identity_sha256": v2.get("frame_identity_sha256"),
        },
        "meta_path": str(meta_path.relative_to(ROOT)),
    }


def nested_exclusions(run_tag: str) -> dict:
    from scripts.rescore_nested_common_footprint import (  # noqa: E402
        SIDES, _base_date, _covering_tile, _index_manifests, _run_dir)

    index = _index_manifests()
    children = json.loads((NEST / "nested_lr64_manifest.json").read_text())["tiles"]
    mismatched, missing = [], []
    for child in children:
        covering = {side: _covering_tile(child, side, index) for side in SIDES}
        if any(t is None for t in covering.values()):
            missing.append(child["tile_id"])
            continue
        dirs = {side: _run_dir(t["parent_city"], t["tile_id"], side, run_tag) for side, t in covering.items()}
        if not all((d / "metrics.json").is_file() for d in dirs.values()):
            missing.append(child["tile_id"])
            continue
        dates = {side: _base_date(d) for side, d in dirs.items()}
        if len(set(dates.values())) != 1:
            mismatched.append({"child": child["tile_id"], "parent_tile_id": child["parent_tile_id"],
                               "project_folder": child["project_folder"],
                               "base_dates": {str(k): v for k, v in dates.items()}})

    base = json.loads((ROOT / json.loads((NEST / "nested_lr512_manifest.json").read_text())["base_manifest"])
                      .read_text())
    base_ids = {t["tile_id"] for t in base["tiles"]}
    nested_ids = {t["tile_id"] for t in json.loads((NEST / "nested_lr512_manifest.json").read_text())["tiles"]}
    dropped = sorted(base_ids - nested_ids)
    return {
        "run_tag": run_tag,
        "n_children": len(children),
        "base_date_mismatch": mismatched,
        "n_base_date_mismatch": len(mismatched),
        "n_missing": len(missing),
        "complete_tiles": len(base_ids),
        "nested_parents": len(nested_ids),
        "parents_dropped": dropped,
        "parents_dropped_reason": "empty HR evaluation mask after erosion and S2-validity masking",
    }


def nested_projects() -> list[dict]:
    tiles = json.loads((NEST / "nested_lr512_manifest.json").read_text())["tiles"]
    by: dict[str, list[dict]] = {}
    for t in tiles:
        by.setdefault(t["project_folder"], []).append(t)
    return [{"project_folder": p, "parent_city": ts[0]["parent_city"], "n_parents": len(ts),
             "parent_tile_ids": sorted(t["tile_id"] for t in ts)} for p, ts in sorted(by.items())]


def tex_site_table(sites: list[dict]) -> str:
    rows = []
    for s in sites:
        nib, s2, ev = s["nib"], s["s2"], s["evaluation"]
        rows.append(
            f"    {LABEL.get(s['site'], s['site'].capitalize())} & {nib.get('export_id') or '--'} & {nib['date']} & "
            f"{s2['mgrs_tile']} & {s2['first_date']}--{s2['last_date']} & {s2['n_frames']} & "
            f"{ev['base_frame_date']} & {ev['base_minus_nib_days']:+d} \\\\")
    return "\n".join([
        "% AUTO-GENERATED by scripts/build_data_inventory.py; DO NOT EDIT.",
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Named-site data. NIB export ID and project date (norgeibilder \texttt{photoDate}; the",
        r"  export XML date is the download date and is not used). S2 L2A frames are the admitted revisits",
        r"  (AOI cloud fraction $\le 0.15$, valid fraction $\ge 0.85$); the base frame is the frame closest to",
        r"  the NIB date with cloud fraction $\le 0.02$. $\Delta$ is base minus NIB date in days. Product IDs,",
        r"  per-frame cloud fractions, AOI windows and reference shifts are in \texttt{data\_inventory.json}.}",
        r"  \label{tab:data-inventory}",
        r"  \scriptsize",
        r"  \setlength{\tabcolsep}{2.6pt}",
        r"  \begin{tabular}{lrrlcrrr}",
        r"    \toprule",
        r"    Site & NIB export & NIB date & MGRS & S2 frame dates & $n$ & Base frame & $\Delta$ \\",
        r"    \midrule",
        *rows,
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-tag", default="lr512align_v2")
    ap.add_argument("--query-stac", action="store_true")
    ap.add_argument("--out-json", type=Path, default=ROOT / "paper/results/data_inventory.json")
    ap.add_argument("--out-tex", type=Path, default=ROOT / "ScaleF_Overleaf/generated/tables/data_inventory.tex")
    args = ap.parse_args()

    aois_path = S2 / "aois.json"
    aois = json.loads(aois_path.read_text())
    areas = {a["id"]: a for a in aois["areas"]}
    v2 = json.loads(V2_MANIFEST.read_text())
    exports = nib_exports()
    sites = [site_record(c, areas[c], v2["cities"][c], exports) for c in SITE_ORDER]

    baseline_note = None
    if args.query_stac:
        ids = [f["stac_id"] for s in sites for f in s["s2"]["frames"]]
        pb = processing_baselines(ids)
        for s in sites:
            for f in s["s2"]["frames"]:
                f["processing_baseline"] = pb.get(f["stac_id"])
            vals = {f["processing_baseline"] for f in s["s2"]["frames"]}
            s["s2"]["processing_baselines"] = sorted(v for v in vals if v)
            s["s2"]["mixed_offset_regime"] = len({(v or "0") >= "04.00" for v in vals}) > 1
        baseline_note = "queried from Planetary Computer"
    for s in sites:
        after = {date.fromisoformat(f["date"]) >= BASELINE_04_START for f in s["s2"]["frames"]}
        s["s2"]["straddles_baseline_04_00_by_date"] = len(after) > 1

    out = {
        "schema": "scalef.data_inventory.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "aois": {"path": str(aois_path.relative_to(ROOT)), "sha256": sha256(aois_path)},
            "confirmatory_manifest": {"path": str(V2_MANIFEST.relative_to(ROOT)), "sha256": sha256(V2_MANIFEST)},
            "spatial_alignment": {"path": v2["source"], "sha256": v2["source_sha256"]},
            "nib_package_manifests": [str(p.relative_to(ROOT)) for p in NIB_MANIFESTS],
            "nested_manifests": [str(p.relative_to(ROOT)) for p in sorted(NEST.glob("nested_lr*_manifest.json"))],
        },
        "processing": PROCESSING,
        "aois_notes": aois.get("notes"),
        "nib_projects": aois.get("projects"),
        "processing_baseline_source": baseline_note,
        "sites": sites,
        "nested": {"projects": nested_projects(), "exclusions": nested_exclusions(args.run_tag)},
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=1))
    args.out_tex.parent.mkdir(parents=True, exist_ok=True)
    args.out_tex.write_text(tex_site_table(sites))

    ex = out["nested"]["exclusions"]
    print(f"sites={len(sites)} nested_projects={len(out['nested']['projects'])} "
          f"mismatch={ex['n_base_date_mismatch']} missing={ex['n_missing']} "
          f"complete={ex['complete_tiles']} nested={ex['nested_parents']} dropped={ex['parents_dropped']}")
    for s in sites:
        print(s["site"], s["nib"].get("export_id"), s["nib"]["date"], s["s2"]["mgrs_tile"], s["s2"]["n_frames"],
              s["evaluation"]["base_frame_date"], s["evaluation"]["base_minus_nib_days"],
              "straddle" if s["s2"]["straddles_baseline_04_00_by_date"] else "",
              s["s2"].get("processing_baselines", ""))


if __name__ == "__main__":
    main()
