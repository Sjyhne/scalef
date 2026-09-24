#!/usr/bin/env python3
"""Separate stopping effects from update efficiency: K=4 windows vs full-field updates.

Arms (Asker/Bergen/Rana LR512, seeds 6/7/8, 5,000 steps, no early termination):
  full       fourier_diag_v1 lr512_grid (full-field updates)
  k4         update_eff_v1 lr512_k4 (K=4 LR128 windows, 2-px context halo)
  k4_nohalo  update_eff_v1 lr512_k4_nohalo (windows without halo, the pre-fix path)

For each run: step time, LR-selected checkpoint and stop iteration from replaying
the frozen LR-only rule, HR quality at fixed step budgets, at matched elapsed time
and at the LR-selected checkpoint. HR-optimal checkpoints are retrospective only.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_fourier_diagnostic import FROZEN_STOP, replay_lr_stop  # noqa: E402

SITES = ("asker", "bergen", "rana")
SEEDS = (6, 7, 8)
ARMS = {
    "full": ("fourier_diag_v1", "lr512_grid"),
    "k4": ("update_eff_v1", "lr512_k4"),
    "k4_nohalo": ("update_eff_v1", "lr512_k4_nohalo"),
}
ARM_LABEL = {"full": "Full field", "k4": "K=4, halo", "k4_nohalo": "K=4, no halo"}
FIXED = (1000, 2000, 3000, 5000)
OUT_JSON = ROOT / "paper" / "results" / "update_efficiency.json"
TABLE = ROOT / "ScaleF_Overleaf" / "generated" / "tables" / "update_efficiency.tex"
FIG = ROOT / "ScaleF_Overleaf" / "generated" / "figures" / "update_efficiency_time"


def load(arm: str, site: str, seed: int) -> dict | None:
    ns, tag = ARMS[arm]
    p = ROOT / "single_samples" / site / ns / f"{ns}__{tag}__{site}__seed{seed}" / "metrics.json"
    return json.loads(p.read_text()) if p.is_file() else None


def interp_at_time(times, values, t):
    """Value of the last checkpoint reached by elapsed time t (step function)."""
    best = None
    for ti, v in zip(times, values):
        if ti <= t:
            best = v
    return best


def summarise(arm, site, seed, m) -> dict:
    h = m["training"]["history"]
    it = h["iterations"]
    es = m["early_stop"]["score_history"]
    rep = replay_lr_stop(es["iterations"], es["values"], **FROZEN_STOP)
    g = lambda k, i: h[k][it.index(i)] if i in it else None  # noqa: E731
    sel = rep["selected_iter"]
    oracle = min(it, key=lambda i: g("model_lpips", i))
    return {
        "arm": arm, "site": site, "seed": seed, "run_name": m["run_name"],
        "lr_tile_halo": m.get("lr_tile_halo", 0),
        "avg_step_s": m["training_instrumentation"]["average_step_time_seconds"],
        "lr_pixels_per_step": m["training_instrumentation"]["queried"]["average_per_step"]["lr_pixels"],
        "peak_alloc_gb": m["gpu_memory"]["torch_peak_allocated_gb"],
        "fixed": {str(i): {"lpips": g("model_lpips", i), "psnr": g("psnr", i), "val": g("val_loss", i),
                           "elapsed_step_s": g("elapsed_step_seconds", i)} for i in FIXED},
        "lr_selected": {"iter": sel, "stop_iter": rep["stop_iter"], "stopped": rep["stopped"],
                        "lpips": g("model_lpips", sel), "psnr": g("psnr", sel),
                        "elapsed_step_at_stop_s": g("elapsed_step_seconds", rep["stop_iter"])},
        "hr_oracle_retrospective": {"iter": oracle, "lpips": g("model_lpips", oracle)},
        "curve": {k: h[k] for k in ("iterations", "elapsed_step_seconds", "model_lpips", "psnr", "val_loss")},
    }


def s(xs):
    xs = [x for x in xs if x is not None]
    return {"mean": statistics.fmean(xs), "sd": statistics.stdev(xs) if len(xs) > 1 else 0.0,
            "n": len(xs), "values": xs} if xs else None


def main() -> None:
    rows = [summarise(a, site, seed, m) for a in ARMS for site in SITES for seed in SEEDS
            if (m := load(a, site, seed)) is not None]
    by = {(r["arm"], r["site"], r["seed"]): r for r in rows}

    for r in rows:
        f = by.get(("full", r["site"], r["seed"]))
        if f is None:
            continue
        t = r["fixed"]["5000"]["elapsed_step_s"]
        r["full_at_same_time"] = {
            "elapsed_step_s": t,
            "lpips": interp_at_time(f["curve"]["elapsed_step_seconds"], f["curve"]["model_lpips"], t),
            "psnr": interp_at_time(f["curve"]["elapsed_step_seconds"], f["curve"]["psnr"], t),
        }

    agg = {}
    for arm in ARMS:
        for seed in SEEDS:
            rs = [by[(arm, site, seed)] for site in SITES if (arm, site, seed) in by]
            if not rs:
                continue
            e = {
                "avg_step_s": s([r["avg_step_s"] for r in rs]),
                "lr_selected_iter": s([r["lr_selected"]["iter"] for r in rs]),
                "lr_stop_iter": s([r["lr_selected"]["stop_iter"] for r in rs]),
                "time_to_stop_s": s([r["lr_selected"]["elapsed_step_at_stop_s"] for r in rs]),
                "lpips_lr_selected": s([r["lr_selected"]["lpips"] for r in rs]),
                "psnr_lr_selected": s([r["lr_selected"]["psnr"] for r in rs]),
                "lpips_hr_oracle": s([r["hr_oracle_retrospective"]["lpips"] for r in rs]),
            }
            for i in FIXED:
                e[f"lpips_{i}"] = s([r["fixed"][str(i)]["lpips"] for r in rs])
                e[f"psnr_{i}"] = s([r["fixed"][str(i)]["psnr"] for r in rs])
            if arm != "full":
                e["full_lpips_at_k4_5000_time"] = s([r.get("full_at_same_time", {}).get("lpips") for r in rs])
            agg[f"{arm}/seed{seed}"] = e

    pm = lambda x, nd=3: "--" if x is None else f"{x['mean']:.{nd}f}$\\pm${x['sd']:.{nd}f}"  # noqa: E731
    lines = [
        r"\begin{table*}[t]\centering\small",
        r"\caption{Update efficiency on Asker, Bergen and Rana (LR512, grid encoder, 5\,000 steps without "
        r"early termination). Full-field LPIPS; mean$\pm$SD across sites, seeds 6/7/8 pooled per row only "
        r"for step time. \emph{Stop}/\emph{Sel.}: stop and selected iteration from replaying the frozen "
        r"LR-only rule; \emph{Full @ t}: full-field run at the elapsed training time the K=4 arm needed "
        r"for 5\,000 steps.}",
        r"\label{tab:update-efficiency}",
        r"\begin{tabular}{llccccccc}\toprule",
        r"Arm & Seed & s/step & LPIPS@1k & LPIPS@5k & Stop & Sel. & LPIPS LR-sel. & Full @ t \\ \midrule",
    ]
    for arm in ARMS:
        for seed in SEEDS:
            e = agg.get(f"{arm}/seed{seed}")
            if not e:
                continue
            lines.append(
                f"{ARM_LABEL[arm]} & {seed} & {e['avg_step_s']['mean']:.4f} & {pm(e['lpips_1000'])} & "
                f"{pm(e['lpips_5000'])} & {e['lr_stop_iter']['mean']:.0f} & {e['lr_selected_iter']['mean']:.0f} & "
                f"{pm(e['lpips_lr_selected'])} & {pm(e.get('full_lpips_at_k4_5000_time'))} \\\\")
        lines.append(r"\midrule" if arm != list(ARMS)[-1] else r"\bottomrule")
    lines += [r"\end{tabular}", r"\end{table*}"]
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    TABLE.write_text("\n".join(lines) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"full": "#1b9e77", "k4": "#d95f02", "k4_nohalo": "#7570b3"}
    fig, axes = plt.subplots(2, len(SITES), figsize=(10, 5.5), sharex=True, squeeze=False)
    for j, site in enumerate(SITES):
        for arm in ARMS:
            for seed in SEEDS:
                r = by.get((arm, site, seed))
                if r is None:
                    continue
                c = r["curve"]
                lab = ARM_LABEL[arm] if seed == SEEDS[0] else None
                axes[0][j].plot(c["elapsed_step_seconds"], c["model_lpips"], color=colors[arm], lw=0.9, label=lab)
                axes[1][j].plot(c["elapsed_step_seconds"], c["val_loss"], color=colors[arm], lw=0.9)
                sel = r["lr_selected"]
                axes[0][j].plot(r["curve"]["elapsed_step_seconds"][r["curve"]["iterations"].index(sel["iter"])],
                                sel["lpips"], "o", ms=3, color=colors[arm])
        axes[0][j].set_title(f"{site.capitalize()} LR512", fontsize=9)
        axes[1][j].set_xlabel("training step time (s)")
    axes[0][0].set_ylabel("HR LPIPS ↓")
    axes[1][0].set_ylabel("LR holdout loss ↓")
    axes[0][0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{FIG}.{ext}", dpi=200)
    plt.close(fig)

    OUT_JSON.write_text(json.dumps({
        "arms": {k: {"namespace": v[0], "tag": v[1], "label": ARM_LABEL[k]} for k, v in ARMS.items()},
        "protocol": {"iters": 5000, "early_termination": False, "lr_stop_rule": FROZEN_STOP,
                     "hr_oracle_note": "retrospective only; deployment selection is LR-only",
                     "timing": "CUDA-event step time, evaluation excluded"},
        "n_runs": len(rows), "aggregate": agg, "runs": rows,
        "outputs": [str(TABLE.relative_to(ROOT)), f"{FIG.relative_to(ROOT)}.pdf"],
    }, indent=1))
    print(f"{len(rows)} runs -> {OUT_JSON.relative_to(ROOT)}")
    for k, e in agg.items():
        print(k, f"s/step {e['avg_step_s']['mean']:.4f}", "lp1k", pm(e["lpips_1000"]), "lp5k", pm(e["lpips_5000"]),
              "stop", e["lr_stop_iter"]["values"], "sel", pm(e["lpips_lr_selected"]),
              "oracle", pm(e["lpips_hr_oracle"]), "full@t", pm(e.get("full_lpips_at_k4_5000_time")),
              "t_stop", pm(e["time_to_stop_s"], 0))


if __name__ == "__main__":
    main()
