#!/usr/bin/env python3
"""Build the 17-site table and paired statistics from a confirmatory summary.

Also reports the paired difference to the pre-halo lr512align_v2 benchmark,
which has identical inputs, masks, alignment and evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SITE_ORDER = ("asker", "vennesla", "trondheim", "bergen", "rana", "tromso", "amli", "stavanger", "algard",
              "naerbo", "flekkefjord", "rafsbotn", "nittedal", "melhus", "alta", "karasjok", "kautokeino")
SEVEN = ("asker", "bergen", "rana", "tromso", "amli", "vennesla", "trondheim")
LABEL = {"tromso": r"Troms{\o}", "amli": r"{\AA}mli", "algard": r"{\AA}lg{\aa}rd", "naerbo": r"N{\ae}rb{\o}"}
BOOT = dict(seed=20260914, n=200_000)


def site_means(summary: dict, key: str) -> dict[str, float]:
    by: dict[str, list[float]] = {}
    for r in summary["results"]:
        by.setdefault(r["city"], []).append(float(r[key]))
    return {c: float(np.mean(v)) for c, v in by.items()}


def seed_sd(summary: dict, key: str) -> float:
    by: dict[str, list[float]] = {}
    for r in summary["results"]:
        by.setdefault(r["city"], []).append(float(r[key]))
    return float(np.mean([np.std(v, ddof=1) for v in by.values()]))


def boot_ci(values: np.ndarray) -> list[float]:
    rng = np.random.default_rng(BOOT["seed"])
    idx = rng.integers(0, len(values), size=(BOOT["n"], len(values)))
    return [float(v) for v in np.quantile(values[idx].mean(axis=1), [0.025, 0.975])]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", type=Path, default=ROOT / "paper/results/b0_17_v3_halo_summary.json")
    ap.add_argument("--previous", type=Path, default=ROOT / "paper/results/b0_17_lr512align_v2_summary.json")
    ap.add_argument("--out-json", type=Path, default=ROOT / "paper/results/b0_17_v3_halo_paired.json")
    ap.add_argument("--out-tex", type=Path, default=ROOT / "ScaleF_Overleaf/generated/tables/frozen_17_aoi.tex")
    ap.add_argument("--ssim-json", type=Path, default=None,
                    help="Per-site SSIM gains (scripts output b0_17_*_ssim.json) to add SSIM columns.")
    ap.add_argument("--layout-tex", type=Path, default=None,
                    help="Also write the journal-layout copy ([H] float, adjustbox).")
    ap.add_argument("--caption-note", default="")
    args = ap.parse_args()
    ssim = json.loads(args.ssim_json.read_text()) if args.ssim_json else None
    s = json.loads(args.summary.read_text())
    prev = json.loads(args.previous.read_text())

    lp, bil = site_means(s, "lpips"), site_means(s, "lpips_bilinear")
    ps, psb = site_means(s, "psnr"), site_means(s, "psnr_bilinear")
    t = site_means(s, "training_time_s")
    lp_prev, ps_prev = site_means(prev, "lpips"), site_means(prev, "psnr")
    sites = [c for c in SITE_ORDER if c in lp]
    gain = np.array([bil[c] - lp[c] for c in sites])
    dpsnr = np.array([ps[c] - psb[c] for c in sites])
    halo = np.array([lp[c] - lp_prev[c] for c in sites])
    halo_ps = np.array([ps[c] - ps_prev[c] for c in sites])
    out = {
        "summary": str(args.summary.relative_to(ROOT)), "previous": str(args.previous.relative_to(ROOT)),
        "n_sites": len(sites), "bootstrap": BOOT,
        "lpips_mean": float(np.mean([lp[c] for c in sites])), "lpips_bilinear_mean": float(np.mean([bil[c] for c in sites])),
        "lpips_site_sd": float(np.std([lp[c] for c in sites], ddof=1)),
        "lpips_bilinear_site_sd": float(np.std([bil[c] for c in sites], ddof=1)),
        "lpips_gain": {"mean": float(gain.mean()), "site_sd": float(gain.std(ddof=1)), "ci95": boot_ci(gain),
                       "n_sites_improved": int((gain > 0).sum())},
        "lpips_within_site_seed_sd": seed_sd(s, "lpips"),
        "psnr_mean": float(np.mean([ps[c] for c in sites])), "psnr_bilinear_mean": float(np.mean([psb[c] for c in sites])),
        "psnr_gain": {"mean": float(dpsnr.mean()), "site_sd": float(dpsnr.std(ddof=1)), "ci95": boot_ci(dpsnr),
                      "n_sites_higher": int((dpsnr > 0).sum()),
                      "min_site": min(sites, key=lambda c: ps[c] - psb[c]),
                      "min_value": float(dpsnr.min())},
        "psnr_within_site_seed_sd": seed_sd(s, "psnr"),
        "time_s_mean": float(np.mean([t[c] for c in sites])), "time_s_site_sd": float(np.std([t[c] for c in sites], ddof=1)),
        "seven_site": {"lpips": float(np.mean([lp[c] for c in SEVEN])), "bilinear": float(np.mean([bil[c] for c in SEVEN])),
                       "time_s": float(np.mean([t[c] for c in SEVEN]))},
        "vs_previous": {"lpips_delta_mean": float(halo.mean()), "lpips_delta_ci95": boot_ci(halo),
                        "lpips_delta_max_abs": float(np.abs(halo).max()),
                        "psnr_delta_mean": float(halo_ps.mean()), "psnr_delta_ci95": boot_ci(halo_ps),
                        "time_s_previous": float(np.mean(list(site_means(prev, "training_time_s").values())))},
        "per_site": {c: {"lpips": lp[c], "bilinear": bil[c], "psnr": ps[c], "psnr_bilinear": psb[c], "time_s": t[c],
                         "lpips_previous": lp_prev[c]} for c in sites},
    }
    args.out_json.write_text(json.dumps(out, indent=1))

    if ssim:
        out["ssim"] = {k: v for k, v in ssim.items()}
        sm, sb = ssim["per_site_model"], ssim["per_site_bilinear"]
        rows = [f"    {LABEL.get(c, c.capitalize())} & {lp[c]:.3f} & {bil[c]:.3f} & {sm[c]:.3f} & {sb[c]:.3f} & {t[c]:.0f} \\\\"
                for c in sites]
        cols, head = "lrrrrr", r"    Site & LPIPS $\downarrow$ & Bil.\ & SSIM $\uparrow$ & Bil.\ & Time (s) \\"
        s_mean = f" & {ssim['ssim_mean']:.3f} & {ssim['ssim_bilinear_mean']:.3f}"
        s_seven = (f" & {np.mean([sm[c] for c in SEVEN]):.3f} & {np.mean([sb[c] for c in SEVEN]):.3f}")
    else:
        rows = [f"    {LABEL.get(c, c.capitalize())} & {lp[c]:.3f} & {bil[c]:.3f} & {t[c]:.0f} \\\\" for c in sites]
        cols, head, s_mean, s_seven = "lrrr", r"    Site & LPIPS $\downarrow$ & Bil.\ & Time (s) \\", "", ""
    tex = "\n".join([
        "% AUTO-GENERATED by scripts/build_frozen17_table.py; DO NOT EDIT.",
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Named-site results under the selected ScaleF configuration (\Cref{tab:default-config}),",
        r"  averaged over seeds $6/7/8$ (the seeds vary the partial-update sampling order). Each LR512 field",
        r"  uses its independently estimated, frozen reference correction, shared by ScaleF and bilinear.",
        r"  Time is optimization only (5\,000 steps; the LR-only rule restored the best checkpoint but",
        r"  terminated no run early)." + (" " + args.caption_note if args.caption_note else "") + "}",
        r"  \label{tab:frozen-17}",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{3.2pt}",
        f"  \\begin{{tabular}}{{{cols}}}",
        r"    \toprule",
        head,
        r"    \midrule",
        *rows,
        r"    \midrule",
        f"    Mean $\\pm$ site SD (17) & ${out['lpips_mean']:.3f}\\pm{out['lpips_site_sd']:.3f}$ & "
        f"${out['lpips_bilinear_mean']:.3f}\\pm{out['lpips_bilinear_site_sd']:.3f}${s_mean} & "
        f"${out['time_s_mean']:.0f}\\pm{out['time_s_site_sd']:.0f}$ \\\\",
        f"    Mean (seven sites) & {out['seven_site']['lpips']:.3f} & {out['seven_site']['bilinear']:.3f}{s_seven} & "
        f"{out['seven_site']['time_s']:.0f} \\\\",
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]) + "\n"
    args.out_tex.write_text(tex)
    if args.layout_tex:
        layout = (tex.replace(r"\begin{table}[htbp]", r"\begin{table}[H]")
                  .replace(f"  \\begin{{tabular}}{{{cols}}}", f"  \\begin{{adjustbox}}{{max width=\\linewidth}}\n\\begin{{tabular}}{{{cols}}}")
                  .replace(r"  \end{tabular}", "  \\end{tabular}\n\\end{adjustbox}"))
        args.layout_tex.write_text("% Layout copy of generated/tables/frozen_17_aoi.tex; numerical entries preserved.\n" + layout)
    print(json.dumps({k: v for k, v in out.items() if k != "per_site"}, indent=1))


if __name__ == "__main__":
    main()
