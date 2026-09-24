#!/usr/bin/env python3
"""Compare candidate seasons for full-scale Norway SR.

National SR wants a *shared* date window that jointly balances:

  1. Clouds — enough clear revisits (≥6–8) per mainland LR512 cell
  2. Snow  — SCL class 11 low on land (winter/early spring poison the RGB prior)
  3. Phenology — growing season green-up (SCL vegetation / mid-summer),
                 without locking only to peak NDVI if clouds dominate

Layer A (fast, all mainland MGRS): Planetary Computer STAC ``eo:cloud_cover``
counts per window — same spirit as ``map_norway_s2_availability.py``.

Layer B (heavier, latitudinal probe tiles): SCL and B04/B08 stats on
one-per-day scenes — cloud, shadow, snow, vegetation, and clear-land NDVI
stability. Results are checkpointed after every tile and can be resumed.

Example
-------
    # STAC-only season table (network; resumes checkpoints by default):
    python scripts/compare_norway_season_windows.py --mode stac

    # + full-granule SCL/B04/B08 probe on default S/M/N tiles (network-heavy):
    python scripts/compare_norway_season_windows.py --mode both --scl-max-days 24
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fetch_s2_revisits import L2A, catalog, item_mgrs  # noqa: E402
from scripts.map_lr512_clear_counts import (  # noqa: E402
    SCL_CIRRUS,
    SCL_CLOUD_HIGH,
    SCL_CLOUD_MED,
    SCL_NODATA,
    SCL_SHADOW,
    _one_per_day,
    _sign,
    _search_mgrs,
    date_window,
)

SCL_VEG = 4
SCL_NOT_VEG = 5
SCL_WATER = 6
SCL_SNOW = 11

DEFAULT_WINDOWS = [
    # (name, center, days_before, days_after, notes)
    ("apr_pm30", "2025-04-15", 30, 30, "Early spring; snow risk N/mountains"),
    ("may_pm30", "2025-05-15", 30, 30, "Green-up south; snow still likely north"),
    ("jun_pm30", "2025-06-15", 30, 30, "NBS summer cube start; growing season"),
    ("jul_pm45", "2025-07-15", 45, 45, "Current national clear-map pilot"),
    ("jul_pm30", "2025-07-15", 30, 30, "Tighter mid-summer"),
    ("jul_pm60", "2025-07-15", 60, 60, "Wider mid-summer; May snowmelt + early Sept"),
    ("aug_pm30", "2025-08-15", 30, 30, "Often snow-free mountains; late growth"),
    ("sep_pm30", "2025-09-15", 30, 30, "Late season; shorter days / more cloud N"),
    ("jja", "2025-07-15", 45, 46, "Jun–Aug-ish (NBS mainland summer cubes)"),
]

# Latitudinal SCL probe (MGRS that exist in norway_mgrs_list).
DEFAULT_PROBE = [
    ("south", "32VKL"),   # ~Agder / south
    ("south", "32VNM"),   # ~Oslo / Asker region
    ("mid", "32VPM"),     # ~Trøndelag-ish
    ("mid", "33VUH"),
    ("north", "33WXP"),   # Troms / north
    ("north", "34WDB"),
]


def _load_mgrs(path: Path) -> list[str]:
    payload = json.loads(path.read_text())
    return list(payload.get("mgrs") or [])


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)


def _stac_clear_days(mgrs: str, datetime_range: str, *, cloud_lt: float, max_items: int) -> dict:
    kwargs = {
        "collections": [L2A],
        "datetime": datetime_range,
        "max_items": int(max_items),
        "query": {"s2:mgrs_tile": {"eq": mgrs}},
    }
    last_err = None
    items = []
    for attempt in range(4):
        try:
            items = [it for it in catalog().search(**kwargs).items() if item_mgrs(it) == mgrs]
            break
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(3 * (attempt + 1))
    if last_err and not items:
        return {"mgrs_tile": mgrs, "error": str(last_err), "n_scenes": 0, "clear_days": 0}

    by_day: dict[str, float] = {}
    footprint = None
    bbox = None
    lats = []
    for it in items:
        if it.datetime is None:
            continue
        day = it.datetime.astimezone(timezone.utc).strftime("%Y-%m-%d")
        cc = float(it.properties.get("eo:cloud_cover", 100.0))
        if day not in by_day or cc < by_day[day]:
            by_day[day] = cc
        if footprint is None and getattr(it, "geometry", None):
            footprint = it.geometry
        if bbox is None and getattr(it, "bbox", None):
            bbox = list(it.bbox)
            lats = [bbox[1], bbox[3]]
    clear_days = sum(1 for cc in by_day.values() if cc < cloud_lt)
    lat_c = float(statistics.mean(lats)) if lats else None
    return {
        "mgrs_tile": mgrs,
        "n_scenes": len(items),
        "n_days": len(by_day),
        "clear_days": clear_days,
        "cloud_median": float(statistics.median(by_day.values())) if by_day else None,
        "lat_center": lat_c,
        "bbox": bbox,
    }


def run_stac(
    mgrs_list: list[str],
    windows: list[tuple],
    *,
    cloud_lt: float,
    max_items: int,
    min_clear: int,
    checkpoint_dir: Path | None = None,
    resume: bool = False,
) -> dict:
    out_windows = []
    for name, center, db, da, note in windows:
        dr = date_window(center, db, da)
        print(f"\n=== STAC {name} {dr} ===", flush=True)
        rows = []
        for i, mgrs in enumerate(mgrs_list):
            checkpoint = (
                None
                if checkpoint_dir is None
                else checkpoint_dir
                / f"stac_cloud{cloud_lt:g}_items{max_items}"
                / f"{name}_{dr.replace('/', '_')}"
                / f"{mgrs}.json"
            )
            if (
                resume
                and checkpoint is not None
                and checkpoint.is_file()
                and "error" not in json.loads(checkpoint.read_text())
            ):
                row = json.loads(checkpoint.read_text())
            else:
                row = _stac_clear_days(mgrs, dr, cloud_lt=cloud_lt, max_items=max_items)
                if checkpoint is not None:
                    _write_json_atomic(checkpoint, row)
            rows.append(row)
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(mgrs_list)}", flush=True)
        clears = [r["clear_days"] for r in rows if "error" not in r]
        meets = sum(1 for c in clears if c >= min_clear)
        by_band = {"south": [], "mid": [], "north": []}
        for r in rows:
            lat = r.get("lat_center")
            if lat is None:
                continue
            band = "south" if lat < 62 else ("mid" if lat < 67 else "north")
            by_band[band].append(r["clear_days"])
        summary = {
            "name": name,
            "center_date": center,
            "days_before": db,
            "days_after": da,
            "date_range": dr,
            "note": note,
            "n_tiles": len(rows),
            "mean_clear_days": float(statistics.mean(clears)) if clears else None,
            "median_clear_days": float(statistics.median(clears)) if clears else None,
            "frac_tiles_ge_min_clear": meets / max(len(clears), 1),
            "n_tiles_ge_min_clear": meets,
            "by_lat_band_mean_clear": {
                b: (float(statistics.mean(v)) if v else None) for b, v in by_band.items()
            },
            "tiles": rows,
        }
        out_windows.append(summary)
        print(
            f"  mean clear={summary['mean_clear_days']:.1f}  "
            f"≥{min_clear}: {meets}/{len(clears)} ({100*summary['frac_tiles_ge_min_clear']:.0f}%)  "
            f"S/M/N={summary['by_lat_band_mean_clear']}",
            flush=True,
        )
    return {"layer": "stac_eo_cloud_cover", "cloud_lt": cloud_lt, "min_clear": min_clear, "windows": out_windows}


def _scl_day_stats(scl: np.ndarray, *, include_shadow: bool) -> dict:
    valid = scl != SCL_NODATA
    if not valid.any():
        return {
            "valid_frac": 0.0,
            "cloud_frac": 1.0,
            "shadow_frac": 0.0,
            "snow_frac": 0.0,
            "veg_frac": 0.0,
        }
    cloud_ids = {SCL_CLOUD_MED, SCL_CLOUD_HIGH, SCL_CIRRUS}
    if include_shadow:
        cloud_ids.add(SCL_SHADOW)
    v = valid
    cloud = np.isin(scl, list(cloud_ids))
    shadow = scl == SCL_SHADOW
    snow = scl == SCL_SNOW
    veg = scl == SCL_VEG
    # Usable land-ish for RGB SR: valid, not cloud, not snow (water OK but rare in score)
    usable = v & ~cloud & ~snow
    return {
        "valid_frac": float(v.mean()),
        "cloud_frac": float(cloud[v].mean()),
        "shadow_frac": float(shadow[v].mean()),
        "snow_frac": float(snow[v].mean()),
        "veg_frac": float(veg[v].mean()),
        "usable_frac": float(usable.mean()),
    }


def _read_probe_arrays(item) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read SCL/B04/B08 on B04's 10 m grid without downloading full products."""
    signed = _sign(item)
    missing = [band for band in ("SCL", "B04", "B08") if band not in signed.assets]
    if missing:
        raise RuntimeError(f"{item.id}: missing {missing}")
    with rasterio.open(signed.assets["B04"].href) as red_src:
        red = red_src.read(1).astype(np.float32)
        shape = (int(red_src.height), int(red_src.width))
        transform = red_src.transform
        crs = red_src.crs

    def aligned(band: str, resampling: Resampling) -> np.ndarray:
        with rasterio.open(signed.assets[band].href) as src:
            out = np.zeros(shape, dtype=np.uint8 if band == "SCL" else np.float32)
            reproject(
                source=src.read(1),
                destination=out,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=crs,
                resampling=resampling,
            )
            return out

    scl = aligned("SCL", Resampling.nearest)
    nir = aligned("B08", Resampling.bilinear)
    return scl, red, nir


def _probe_day_stats(
    scl: np.ndarray,
    red: np.ndarray,
    nir: np.ndarray,
    *,
    include_shadow: bool,
) -> dict:
    out = _scl_day_stats(scl, include_shadow=include_shadow)
    clear_land = (
        (scl != SCL_NODATA)
        & ~np.isin(scl, [SCL_CLOUD_MED, SCL_CLOUD_HIGH, SCL_CIRRUS, SCL_SHADOW, SCL_SNOW])
        & (scl != SCL_WATER)
    )
    denom = nir + red
    good = clear_land & np.isfinite(denom) & (denom > 0)
    if good.any():
        ndvi = np.clip((nir[good] - red[good]) / denom[good], -1.0, 1.0)
        out["ndvi_clear_land_mean"] = float(np.mean(ndvi))
        out["ndvi_clear_land_median"] = float(np.median(ndvi))
    else:
        out["ndvi_clear_land_mean"] = None
        out["ndvi_clear_land_median"] = None
    out["ndvi_valid_frac"] = float(good.mean())
    return out


def run_scl_probe(
    probes: list[tuple[str, str]],
    windows: list[tuple],
    *,
    max_days: int,
    include_shadow: bool,
    cloud_lt_prefilter: float,
    checkpoint_dir: Path | None = None,
    resume: bool = False,
) -> dict:
    out = []
    for name, center, db, da, note in windows:
        dr = date_window(center, db, da)
        print(f"\n=== SCL probe {name} {dr} ===", flush=True)
        window_rows = []
        for band, mgrs in probes:
            checkpoint = (
                None
                if checkpoint_dir is None
                else checkpoint_dir
                / (
                    f"scl_days{max_days}_shadow{int(include_shadow)}"
                    f"_prefilter{cloud_lt_prefilter:g}"
                )
                / f"{name}_{dr.replace('/', '_')}"
                / f"{mgrs}.json"
            )
            if resume and checkpoint is not None and checkpoint.is_file():
                row = json.loads(checkpoint.read_text())
                window_rows.append(row)
                print(f"  resume {band:5} {mgrs}: {row['n_days_scored']} days", flush=True)
                continue
            items = _one_per_day(_search_mgrs(mgrs, dr, max_items=200))
            # Prefer low STAC cloud for SCL read budget.
            items = sorted(
                items,
                key=lambda it: float(it.properties.get("eo:cloud_cover", 100.0)),
            )[: max(1, int(max_days))]
            day_stats = []
            for it in items:
                cc = float(it.properties.get("eo:cloud_cover", 100.0))
                if cc >= cloud_lt_prefilter and len(day_stats) >= 8:
                    continue
                try:
                    scl, red, nir = _read_probe_arrays(it)
                    st = _probe_day_stats(
                        scl,
                        red,
                        nir,
                        include_shadow=include_shadow,
                    )
                    st["day"] = it.datetime.astimezone(timezone.utc).strftime("%Y-%m-%d")
                    st["eo_cloud_cover"] = cc
                    day_stats.append(st)
                except Exception as exc:  # noqa: BLE001
                    print(f"  skip {mgrs} {it.id}: {exc}", flush=True)
            def _m(key: str) -> float | None:
                xs = [d[key] for d in day_stats if d.get(key) is not None]
                return float(statistics.mean(xs)) if xs else None

            ndvi_means = [
                float(d["ndvi_clear_land_mean"])
                for d in sorted(day_stats, key=lambda x: x["day"])
                if d.get("ndvi_clear_land_mean") is not None
            ]
            ndvi_deltas = [
                abs(b - a) for a, b in zip(ndvi_means, ndvi_means[1:])
            ]

            row = {
                "band": band,
                "mgrs_tile": mgrs,
                "n_days_scored": len(day_stats),
                "mean_cloud_frac": _m("cloud_frac"),
                "mean_shadow_frac": _m("shadow_frac"),
                "mean_snow_frac": _m("snow_frac"),
                "mean_veg_frac": _m("veg_frac"),
                "mean_usable_frac": _m("usable_frac"),
                "mean_ndvi_clear_land": _m("ndvi_clear_land_mean"),
                "ndvi_temporal_std": (
                    float(statistics.pstdev(ndvi_means)) if len(ndvi_means) >= 2 else None
                ),
                "ndvi_successive_abs_delta_mean": (
                    float(statistics.mean(ndvi_deltas)) if ndvi_deltas else None
                ),
                "n_days_snow_lt_05": sum(1 for d in day_stats if d.get("snow_frac", 1) < 0.05),
                "n_days_usable_ge_80": sum(1 for d in day_stats if d.get("usable_frac", 0) >= 0.80),
                "days": day_stats,
            }
            if checkpoint is not None:
                _write_json_atomic(checkpoint, row)
            window_rows.append(row)
            print(
                f"  {band:5} {mgrs}: snow={row['mean_snow_frac']} "
                f"cloud={row['mean_cloud_frac']} veg={row['mean_veg_frac']} "
                f"usable_days≥0.8={row['n_days_usable_ge_80']}/{row['n_days_scored']}",
                flush=True,
            )
        out.append(
            {
                "name": name,
                "center_date": center,
                "date_range": dr,
                "note": note,
                "probes": window_rows,
            }
        )
    return {
        "layer": "scl_b04_b08_probe",
        "sampling": "one scene per day, lowest eo:cloud_cover days first, capped by scl_max_days",
        "ndvi_stability": "population stddev and mean successive absolute delta of clear non-water land mean NDVI",
        "windows": out,
    }


def _rank_and_recommend(stac: dict | None, scl: dict | None) -> dict:
    """Simple multi-criterion rank for writing the season note."""
    ranks = []
    stac_by = {w["name"]: w for w in (stac or {}).get("windows") or []}
    scl_by = {w["name"]: w for w in (scl or {}).get("windows") or []}
    names = list(stac_by.keys()) or list(scl_by.keys())
    for name in names:
        s = stac_by.get(name)
        p = scl_by.get(name)
        snow_n = None
        usable = None
        if p:
            north = [x for x in p["probes"] if x["band"] == "north"]
            allp = p["probes"]
            snow_n = statistics.mean(
                [x["mean_snow_frac"] for x in north if x.get("mean_snow_frac") is not None]
                or [1.0]
            )
            usable = statistics.mean(
                [x["n_days_usable_ge_80"] for x in allp if x.get("n_days_usable_ge_80") is not None]
                or [0]
            )
        ranks.append(
            {
                "name": name,
                "frac_tiles_ge_min_clear": (s or {}).get("frac_tiles_ge_min_clear"),
                "mean_clear_days": (s or {}).get("mean_clear_days"),
                "north_mean_clear": ((s or {}).get("by_lat_band_mean_clear") or {}).get("north"),
                "probe_mean_snow_north": snow_n,
                "probe_mean_usable_days": usable,
                "note": (s or p or {}).get("note"),
            }
        )
    # Prefer high clear coverage, low northern snow, high usable days.
    def score(r: dict) -> float:
        clear = float(r.get("frac_tiles_ge_min_clear") or 0)
        snow = float(r.get("probe_mean_snow_north") if r.get("probe_mean_snow_north") is not None else 0.5)
        use = float(r.get("probe_mean_usable_days") or 0) / 20.0
        return 2.0 * clear + use - 1.5 * snow

    ranks.sort(key=score, reverse=True)
    return {"ranked": ranks, "top": ranks[0]["name"] if ranks else None, "score_fn": "2*clear_frac + usable/20 - 1.5*north_snow"}


def write_markdown(path: Path, payload: dict) -> None:
    lines = [
        "# Norway SR season window comparison",
        "",
        f"Generated: `{payload.get('created_utc')}`",
        "",
        "Goal: pick a **shared** date window for mainland Norway production that",
        "balances clear revisits (clouds), snow-free land, and growing-season appearance.",
        "",
        "## Criteria",
        "",
        "- **Clouds:** STAC `eo:cloud_cover` clear-day counts per MGRS (Layer A).",
        "- **Snow:** SCL class 11 on probe tiles (Layer B). Note: the July LR512 clear map",
        "  excluded snow only when generated with `--max-snow-frac`.",
        "- **NDVI stability:** temporal population standard deviation and mean successive",
        "  absolute change of clear, non-water land mean NDVI from B04/B08.",
        "- **Growing season:** prefer Jun–Aug phenology (NBS mainland summer cubes are JJA);",
        "  SCL vegetation fraction is a coarse proxy only.",
        "",
    ]
    rec = payload.get("recommendation") or {}
    if rec.get("top"):
        lines += [
            f"**Current top by heuristic score:** `{rec['top']}`",
            f"(score = `{rec.get('score_fn')}` — refine after Layer B finishes).",
            "",
        ]
    stac = payload.get("stac") or {}
    if stac.get("windows"):
        lines += [
            "## Layer A — STAC clear days (all mainland MGRS)",
            "",
            f"Rules: `eo:cloud_cover` < {stac.get('cloud_lt')}, one-best-per-day; "
            f"tile “ok” if clear days ≥ {stac.get('min_clear')}.",
            "",
            "| Window | Range | Mean clear days | % tiles ≥ min | S / M / N mean |",
            "|--|--|--:|--:|--|",
        ]
        for w in stac["windows"]:
            b = w.get("by_lat_band_mean_clear") or {}
            smn = "/".join(
                f"{(b.get(k) if b.get(k) is not None else float('nan')):.1f}" for k in ("south", "mid", "north")
            )
            lines.append(
                f"| `{w['name']}` | {w['date_range']} | {w['mean_clear_days']:.1f} | "
                f"{100*w['frac_tiles_ge_min_clear']:.0f}% | {smn} |"
            )
        lines.append("")
    scl = payload.get("scl") or {}
    if scl.get("windows"):
        lines += [
            "## Layer B — SCL probe (snow / usable)",
            "",
            "| Window | Probe mean snow (N tiles) | Mean usable days (frac≥0.8) |",
            "|--|--:|--:|",
        ]
        for w in scl["windows"]:
            north = [p for p in w["probes"] if p["band"] == "north"]
            snow = statistics.mean(
                [p["mean_snow_frac"] for p in north if p.get("mean_snow_frac") is not None] or [float("nan")]
            )
            use = statistics.mean([p["n_days_usable_ge_80"] for p in w["probes"]])
            lines.append(f"| `{w['name']}` | {snow:.3f} | {use:.1f} |")
        lines.append("")
    lines += [
        "## Interpretation notes",
        "",
        "- **April–May:** often more clear sky inland but snow (esp. north / mountains) "
        "makes RGB MISR look like winter — usually wrong for a “national summer map”.",
        "- **June–August:** NBS publishes mainland L2A summer datacubes for this reason; "
        "best phenology / snow-free compromise for most of the mainland.",
        "- **September:** can still be snow-free in the south; north loses light and often gains cloud.",
        "- **Year matters:** 2025 is the pilot year for the clear map; re-score the winning window "
        "on 2023/2024 before locking multi-year production.",
        "",
        "## Next steps",
        "",
        "1. Finish Layer B if only STAC ran.",
        "2. Re-run `map_lr512_clear_counts.py` for the top 1–2 windows with "
        "`--max-snow-frac 0.05` so LR512 heatmaps match production intent.",
        "3. Pick the shared window, then re-fetch a contiguous block with `--min-clear 6–8`.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["stac", "scl", "both"], default="stac")
    ap.add_argument(
        "--mgrs-list",
        type=Path,
        default=ROOT / "production/cloud_availability/norway_mgrs_list.json",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "production/cloud_availability/season_compare_2025",
    )
    ap.add_argument(
        "--cloud-lt",
        type=float,
        default=15.0,
        help="Scene eo:cloud_cover threshold for Layer A; also informs the Layer B prefilter.",
    )
    ap.add_argument("--min-clear", type=int, default=6, help="Layer A clear-day target.")
    ap.add_argument("--max-items", type=int, default=200, help="STAC result cap per tile/window.")
    ap.add_argument(
        "--scl-max-days",
        type=int,
        default=20,
        help="Layer B scene-read cap per probe/window after lowest eo:cloud_cover ranking.",
    )
    ap.add_argument(
        "--include-shadow",
        action="store_true",
        help="Count SCL class 3 as cloud; shadow fraction is reported either way.",
    )
    ap.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse per-window/per-MGRS checkpoint JSON (default: enabled).",
    )
    args = ap.parse_args()

    mgrs_list = _load_mgrs(args.mgrs_list)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    payload: dict = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "year": 2025,
        "mgrs_list": str(args.mgrs_list.relative_to(ROOT)),
        "n_mgrs": len(mgrs_list),
        "windows_defined": [
            {"name": n, "center": c, "before": a, "after": b, "note": note}
            for n, c, a, b, note in DEFAULT_WINDOWS
        ],
    }

    stac = None
    scl = None
    if args.mode in ("stac", "both"):
        stac = run_stac(
            mgrs_list,
            DEFAULT_WINDOWS,
            cloud_lt=args.cloud_lt,
            max_items=args.max_items,
            min_clear=args.min_clear,
            checkpoint_dir=args.out_dir / "checkpoints",
            resume=bool(args.resume),
        )
        payload["stac"] = stac
    if args.mode in ("scl", "both"):
        scl = run_scl_probe(
            DEFAULT_PROBE,
            DEFAULT_WINDOWS,
            max_days=args.scl_max_days,
            include_shadow=args.include_shadow,
            cloud_lt_prefilter=max(args.cloud_lt * 2, 40.0),
            checkpoint_dir=args.out_dir / "checkpoints",
            resume=bool(args.resume),
        )
        payload["scl"] = scl

    payload["recommendation"] = _rank_and_recommend(stac, scl)
    out_json = args.out_dir / "season_compare.json"
    out_json.write_text(json.dumps(payload, indent=2) + "\n")
    write_markdown(args.out_dir / "README.md", payload)
    print(f"\nWrote {out_json}", flush=True)
    print(f"Top heuristic: {payload['recommendation'].get('top')}", flush=True)


if __name__ == "__main__":
    main()
