#!/usr/bin/env python3
"""Nested size-ladder table from the float-output validation subset (floatval_v3).

Quality comes from ``paper/results/nested_float_validation.json`` (float_v3 route, with the
published png_v2 LPIPS alongside). Cost and early stopping come from the floatval_v3 run
metrics: serial optimization seconds summed over all fits covering one LR512 parent.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_nested_float_validation import MANIFEST as RUN_MANIFEST  # noqa: E402

VALIDATION = ROOT / "paper" / "results" / "nested_float_validation.json"
OUT_JSON = ROOT / "paper" / "results" / "nested_float_ladder.json"
OUT_TEX = ROOT / "ScaleF_Overleaf" / "generated" / "tables" / "nested_float_ladder.tex"
PARENT_KM2 = 26.2144


def project_mean(rows: list[dict], side: int, key: str) -> float:
    by_parent: dict[tuple[str, str], list[float]] = {}
    for r in rows:
        if r["side"] == side:
            by_parent.setdefault((r["project_folder"], r["parent_tile_id"]), []).append(r[key])
    by_project: dict[str, list[float]] = {}
    for (project, _), vals in by_parent.items():
        by_project.setdefault(project, []).append(statistics.fmean(vals))
    return statistics.fmean(statistics.fmean(v) for v in by_project.values())


def main() -> None:
    val = json.loads(VALIDATION.read_text())
    jobs = json.loads(RUN_MANIFEST.read_text())["jobs"]
    cost: dict[int, dict[str, float]] = {}
    stops: dict[int, list[bool]] = {}
    for j in jobs:
        m = json.loads(Path(j["metrics"]).read_text())
        side = int(j["side"])
        cost.setdefault(side, {}).setdefault(j["parent_tile_id"], 0.0)
        cost[side][j["parent_tile_id"]] += float(m["training_time_seconds"])
        stops.setdefault(side, []).append(bool(m["early_stop"]["stopped"]))

    rows = []
    for side in sorted(cost):
        gpu_s_parent = statistics.fmean(cost[side].values())
        n_windows = sum(1 for r in val["rows"]["float_v3"] if r["side"] == side)
        rows.append({
            "lr_side": side,
            "n_parents": len(cost[side]),
            "n_fits": len(stops[side]),
            "n_windows": n_windows,
            "lpips": project_mean(val["rows"]["float_v3"], side, "lpips"),
            "lpips_bilinear": project_mean(val["rows"]["float_v3"], side, "lpips_bilinear"),
            "psnr": project_mean(val["rows"]["float_v3"], side, "psnr"),
            "psnr_bilinear": project_mean(val["rows"]["float_v3"], side, "psnr_bilinear"),
            "lpips_png_v2": project_mean(val["rows"]["png_v2"], side, "lpips"),
            "early_stop_frac": sum(stops[side]) / len(stops[side]),
            "gpu_s_per_parent": gpu_s_parent,
            "gpu_s_per_km2": gpu_s_parent / PARENT_KM2,
            "km2_per_gpu_h": PARENT_KM2 * 3600 / gpu_s_parent,
        })
    OUT_JSON.write_text(json.dumps({"source": str(VALIDATION.relative_to(ROOT)),
                                    "decision": val["decision"], "rows": rows}, indent=1) + "\n")

    n_proj = len({r["project_folder"] for r in val["rows"]["float_v3"]})
    body = "\n".join(
        f"    {r['lr_side']} & {r['lpips']:.3f} & {r['lpips_bilinear']:.3f} & {r['psnr']:.2f} & "
        f"{r['lpips_png_v2']:.3f} & {100 * r['early_stop_frac']:.0f} & {r['gpu_s_per_km2']:.1f} & "
        f"{r['km2_per_gpu_h']:.0f} \\\\"
        for r in rows
    )
    tol = val["decision"]["lpips_tolerance"]
    OUT_TEX.write_text(rf"""\begin{{table}}[htbp]
  \centering
  \caption{{Nested field-size ladder scored from raw floating-point outputs.
  One LR512 parent per NIB project ($n={n_proj}$, lowest tile identifier, fixed before scoring) was refitted at LR64/128/256/512 with the blur halo; each parent's own LR512 correction is shared by its children and by bilinear.
  Scores use {rows[0]['n_windows']} shared LR64 windows, averaged within parents and then over projects with equal weight.
  LR64/128 use full-field updates and LR256/512 use $K=4$ windows, so differences combine field size with update policy.
  ``PNG'' is the display-PNG rescoring route of the earlier $85$-parent table on the same windows; it differs from the float scores by more than the predeclared ${tol}$ LPIPS tolerance at LR256/512, so that table is not reported.}}
  \label{{tab:nested-ladder}}
  \small
  \begin{{tabular}}{{rrrrrrrr}}
    \toprule
    LR side & LPIPS $\downarrow$ & Bilinear & PSNR & PNG & Stop (\%) & GPU-s/km$^2$ & km$^2$/GPU-h \\
    \midrule
{body}
    \bottomrule
  \end{{tabular}}\\[0.4em]
  {{\raggedright\scriptsize Bilinear PSNR is {rows[0]['psnr_bilinear']:.2f}\,dB at every size. Stop: fits ended early by the LR rule. Cost: serial optimization time summed over all fits covering one parent ($26.2144\,\mathrm{{km}}^2$), one job per H100.\par}}
\end{{table}}
""")
    for r in rows:
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()})


if __name__ == "__main__":
    main()
