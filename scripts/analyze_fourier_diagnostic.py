#!/usr/bin/env python3
"""Analyse the fourier_diag_v1 encoding diagnostic.

Per run: encoder/decoder parameters, step time, peak memory, and quality on the
shared LR64 geographic window (scored against the LR512 parent reference) at
fixed checkpoints, at the checkpoint the frozen LR-only stopping rule would have
selected (replayed on the recorded holdout scores), and at the retrospective
HR-optimal checkpoint (reported for diagnosis only, never for selection).

Writes paper/results/fourier_diagnostic.json, two LaTeX tables and
quality-vs-time figures under ScaleF_Overleaf/generated/.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMESPACE = "fourier_diag_v1"
SITES = ("asker", "bergen", "rana")
SIDES = (64, 256, 512)
SEEDS = (6, 7, 8)
FIXED_CHECKPOINTS = (1000, 2000, 3000, 5000)
ENCODING_ORDER = ("grid", "fourier_s2", "fourier_s5", "fourier_s10", "fourier_bw10")
ENCODING_LABEL = {
    "grid": "Grid (dense tcnn)",
    "fourier_s2": "Fourier $s{=}2$",
    "fourier_s5": "Fourier $s{=}5$",
    "fourier_s10": "Fourier $s{=}10$",
    "fourier_bw10": "Fourier fixed bandwidth",
}
# Frozen LR-only stopping rule (b0_17 / confirmatory_v2_lr512align provenance).
FROZEN_STOP = dict(patience=8, min_iters=1000, min_delta=0.0005, max_regression=0.01, ema_alpha=0.4)
# Fixed before inspecting trajectory slopes: extend both encodings on Asker LR512
# only if the LR validation EMA (the deployment selection signal) still improves
# by more than min_delta over the last 1000 steps in at least 2 of 3 seeds for
# either the grid or the fixed-bandwidth Fourier run.
EXTENSION_RULE = {
    "case": {"site": "asker", "side": 512, "encodings": ["grid", "fourier_bw10"]},
    "signal": "EMA(alpha=0.4) of LR holdout Charbonnier",
    "window": [4000, 5000],
    "threshold": FROZEN_STOP["min_delta"],
    "min_seeds": 2,
}

OUT_JSON = ROOT / "paper" / "results" / "fourier_diagnostic.json"
TABLE_DIR = ROOT / "ScaleF_Overleaf" / "generated" / "tables"
FIG_DIR = ROOT / "ScaleF_Overleaf" / "generated" / "figures"


def replay_lr_stop(iters, values, *, patience, min_iters, min_delta, max_regression, ema_alpha):
    """Replay eval.lr_holdout.EarlyStopState.observe on recorded raw scores."""
    ema = math.nan
    best, best_iter, bad = math.inf, None, 0
    ema_hist = []
    for it, v in zip(iters, values):
        ema = v if not math.isfinite(ema) or ema_alpha <= 0 else ema_alpha * v + (1 - ema_alpha) * ema
        ema_hist.append(ema)
        if it < min_iters:
            continue
        if ema < best - min_delta:
            best, best_iter, bad = ema, it, 0
        else:
            bad += 1
            if max_regression > 0 and math.isfinite(best) and ema > best + max_regression:
                bad = max(bad, patience)
        if patience > 0 and bad >= patience:
            return {"stopped": True, "stop_iter": it, "selected_iter": best_iter, "ema": ema_hist}
    return {"stopped": False, "stop_iter": iters[-1] if iters else None, "selected_iter": best_iter, "ema": ema_hist}


def encoder_params(side: int, audit: dict) -> int:
    for row in audit["sides"]:
        if int(row["lr_side"]) == side:
            return int(row["encoder_params_actual"])
    raise KeyError(side)


def load_runs() -> list[dict]:
    runs = []
    for site in SITES:
        for p in sorted((ROOT / "single_samples" / site / NAMESPACE).glob(f"{NAMESPACE}__*/metrics.json")):
            m = json.loads(p.read_text())
            _, tag, site_name, seed_tag = m["run_name"].split("__")
            side_tag, enc = tag.split("_", 1)
            if "_it" in enc:
                continue
            runs.append({"path": p, "m": m, "site": site_name, "side": int(side_tag[2:]),
                         "seed": int(seed_tag[4:]), "encoding": enc})
    return runs


def at(hist: dict, key: str, it: int):
    try:
        return hist[key][hist["iterations"].index(it)]
    except (KeyError, ValueError):
        return None


def summarise_run(r: dict, audit: dict) -> dict:
    m = r["m"]
    h = m["training"]["history"]
    es = m["early_stop"]["score_history"]
    rep = replay_lr_stop(es["iterations"], es["values"], **FROZEN_STOP)
    sel = rep["selected_iter"]
    oracle_it = min(h["iterations"], key=lambda it: at(h, "sub_model_lpips", it))
    args = m["provenance"]["args"]
    total = int(m["model_parameters"]["total"])
    enc_p = encoder_params(r["side"], audit) if r["encoding"] == "grid" else 0

    def q(it):
        return None if it is None else {
            "iter": it,
            "sub_lpips": at(h, "sub_model_lpips", it),
            "sub_psnr": at(h, "sub_psnr", it),
            "sub_ssim": at(h, "sub_model_ssim", it),
            "full_lpips": at(h, "model_lpips", it),
            "full_psnr": at(h, "psnr", it),
            "elapsed_step_s": at(h, "elapsed_step_seconds", it),
            "elapsed_wall_s": at(h, "elapsed_wall_seconds", it),
        }

    ema_full = replay_lr_stop(es["iterations"], es["values"], **{**FROZEN_STOP, "patience": 0})["ema"]
    ema_by_it = dict(zip(es["iterations"], ema_full))
    return {
        "run_name": m["run_name"], "site": r["site"], "side": r["side"], "seed": r["seed"],
        "encoding": r["encoding"],
        "input_projection": m["input_projection"], "decoder": m["model"],
        "decoder_depth_arg": args.get("network_depth"), "decoder_width": args.get("network_hidden_dim"),
        "fourier_scale": args.get("fourier_scale") if r["encoding"] != "grid" else None,
        "fourier_matrix": m.get("fourier_matrix"),
        "params_total": total, "params_encoder": enc_p, "params_rest": total - enc_p,
        "completed_iters": m["completed_iters"],
        "avg_step_s": m["training_instrumentation"]["average_step_time_seconds"],
        "train_step_total_s": m["training_instrumentation"]["total_step_time_seconds"],
        "train_wall_s": m["training_time_seconds"],
        "peak_alloc_gb": m["gpu_memory"]["torch_peak_allocated_gb"],
        "bilinear_sub_lpips": h["sub_bilinear_lpips"][0], "bilinear_sub_psnr": h["sub_bilinear_psnr"][0],
        "eval_reference": m["training"].get("eval_subwindow_reference",
                                           {"s2_dir": args.get("s2_dir"), "note": "own LR512 field"}),
        "fixed": {str(it): q(it) for it in FIXED_CHECKPOINTS},
        "lr_selected": {**q(sel), "stopped": rep["stopped"], "stop_iter": rep["stop_iter"],
                        "elapsed_step_at_stop_s": at(h, "elapsed_step_seconds", rep["stop_iter"])},
        "hr_oracle_retrospective": q(oracle_it),
        "lr_val_ema_4000_5000": [ema_by_it.get(4000), ema_by_it.get(5000)],
        "curve": {k: h[k] for k in ("iterations", "elapsed_step_seconds", "elapsed_wall_seconds",
                                    "sub_model_lpips", "sub_psnr", "sub_model_ssim", "model_lpips", "val_loss")},
    }


def stats(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return {"n": len(xs), "mean": statistics.fmean(xs), "sd": statistics.stdev(xs) if len(xs) > 1 else 0.0,
            "min": min(xs), "max": max(xs), "values": xs}


def aggregate(rows: list[dict]) -> dict:
    by = defaultdict(dict)
    for r in rows:
        by[(r["side"], r["encoding"], r["seed"])][r["site"]] = r
    out = {}
    for (side, enc, seed), per_site in sorted(by.items()):
        grid = {s: by.get((side, "grid", seed), {}).get(s) for s in per_site}
        entry = {"sites": sorted(per_site)}
        for label, get in (
            ("sub_lpips_5000", lambda r: r["fixed"]["5000"]["sub_lpips"]),
            ("sub_psnr_5000", lambda r: r["fixed"]["5000"]["sub_psnr"]),
            ("sub_lpips_lr_selected", lambda r: r["lr_selected"]["sub_lpips"]),
            ("sub_psnr_lr_selected", lambda r: r["lr_selected"]["sub_psnr"]),
            ("lr_selected_iter", lambda r: r["lr_selected"]["iter"]),
            ("lr_stop_iter", lambda r: r["lr_selected"]["stop_iter"]),
            ("sub_lpips_hr_oracle", lambda r: r["hr_oracle_retrospective"]["sub_lpips"]),
            ("avg_step_s", lambda r: r["avg_step_s"]),
            ("peak_alloc_gb", lambda r: r["peak_alloc_gb"]),
        ):
            entry[label] = stats([get(r) for r in per_site.values()])
        if enc != "grid":
            entry["delta_vs_grid_sub_lpips_5000"] = stats([
                per_site[s]["fixed"]["5000"]["sub_lpips"] - grid[s]["fixed"]["5000"]["sub_lpips"]
                for s in per_site if grid.get(s)])
        out[f"lr{side}/{enc}/seed{seed}"] = entry
    return out


def extension_check(rows: list[dict]) -> dict:
    c = EXTENSION_RULE["case"]
    res = {}
    for enc in c["encodings"]:
        gains = []
        for r in rows:
            if r["site"] == c["site"] and r["side"] == c["side"] and r["encoding"] == enc:
                a, b = r["lr_val_ema_4000_5000"]
                gains.append({"seed": r["seed"], "ema_gain": None if a is None else a - b})
        n_improving = sum(1 for g in gains if g["ema_gain"] is not None and g["ema_gain"] > EXTENSION_RULE["threshold"])
        res[enc] = {"per_seed": gains, "n_improving": n_improving}
    extend = any(v["n_improving"] >= EXTENSION_RULE["min_seeds"] for v in res.values())
    return {"rule": EXTENSION_RULE, "result": res, "extend": extend}


def extension_runs(audit: dict, iters: int = 10000) -> list[dict]:
    """Identical-budget extension of the predeclared case, if it was triggered and run."""
    c = EXTENSION_RULE["case"]
    out = []
    for enc in c["encodings"]:
        for seed in SEEDS:
            run = f"{NAMESPACE}__lr{c['side']}_{enc}_it{iters}__{c['site']}__seed{seed}"
            p = ROOT / "single_samples" / c["site"] / NAMESPACE / run / "metrics.json"
            if not p.is_file():
                continue
            s = summarise_run({"m": json.loads(p.read_text()), "site": c["site"], "side": c["side"],
                               "seed": seed, "encoding": enc, "path": p}, audit)
            h = s["curve"]
            s["fixed_ext"] = {str(it): {"sub_lpips": at(h, "sub_model_lpips", it), "sub_psnr": at(h, "sub_psnr", it)}
                              for it in (5000, 7500, 10000)}
            s["note"] = ("separate run with a 10k-step schedule; its step-5000 checkpoint is not the "
                         "5k-budget run because the learning-rate schedule spans the full budget")
            out.append(s)
    return out


def fmt(s, key="mean", nd=3):
    return "--" if s is None else f"{s[key]:.{nd}f}"


def pm(s, nd=3):
    return "--" if s is None else f"{s['mean']:.{nd}f}$\\pm${s['sd']:.{nd}f}"


def write_tables(rows: list[dict], agg: dict) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    cost = [
        r"\begin{table}[t]\centering\small",
        r"\caption{Encoding diagnostic: cost per configuration (full-field updates, 5\,000 steps, "
        r"identical decoder: tcnn FullyFusedMLP, 3 hidden layers of 256, fp16). "
        r"Encoder parameters are trainable grid entries; Fourier matrices are fixed buffers. "
        r"Step time excludes evaluation; values are means over 3 sites $\times$ 3 seeds. "
        r"Peak is PyTorch's allocator peak; it excludes the tiny-cuda-nn device arena (grid table, "
        r"fused-MLP workspace) and is therefore not a process-level memory comparison.}",
        r"\label{tab:fourier-cost}",
        r"\begin{tabular}{llrrrr}\toprule",
        r"LR side & Encoding & Encoder params & Other params & s/step & Torch peak (GB) \\ \midrule",
    ]
    for side in SIDES:
        for enc in ENCODING_ORDER:
            rs = [r for r in rows if r["side"] == side and r["encoding"] == enc]
            if not rs:
                continue
            scale = rs[0]["fourier_scale"]
            lab = ENCODING_LABEL[enc] + (f" ($s{{=}}{scale:g}$)" if enc == "fourier_bw10" else "")
            cost.append(
                f"{side} & {lab} & {rs[0]['params_encoder']:,} & {rs[0]['params_rest']:,} & ".replace(",", "{,}") +
                f"{statistics.fmean(r['avg_step_s'] for r in rs):.4f} & "
                f"{statistics.fmean(r['peak_alloc_gb'] for r in rs):.1f} \\\\")
        cost.append(r"\midrule" if side != SIDES[-1] else r"\bottomrule")
    cost += [r"\end{tabular}", r"\end{table}"]
    (TABLE_DIR / "fourier_diagnostic_cost.tex").write_text("\n".join(cost) + "\n")

    qual = [
        r"\begin{table*}[t]\centering\small",
        r"\caption{Encoding diagnostic: LPIPS on the shared central LR64 window, scored against the "
        r"LR512 parent reference for every field size. Mean$\pm$SD across the three sites "
        r"(Asker, Bergen, Rana), seeds kept separate. Fixed: step 5\,000 (no early termination). "
        r"LR-selected: checkpoint chosen by replaying the frozen LR-only stopping rule on the recorded "
        r"holdout scores.}",
        r"\label{tab:fourier-quality}",
        r"\begin{tabular}{llcccccc}\toprule",
        r" & & \multicolumn{3}{c}{Fixed 5\,000 steps} & \multicolumn{3}{c}{LR-selected} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
        r"LR side & Encoding & seed 6 & seed 7 & seed 8 & seed 6 & seed 7 & seed 8 \\ \midrule",
    ]
    for side in SIDES:
        for enc in ENCODING_ORDER:
            keys = [f"lr{side}/{enc}/seed{s}" for s in SEEDS]
            if not any(k in agg for k in keys):
                continue
            fixed = " & ".join(pm(agg.get(k, {}).get("sub_lpips_5000")) for k in keys)
            sel = " & ".join(pm(agg.get(k, {}).get("sub_lpips_lr_selected")) for k in keys)
            qual.append(f"{side} & {ENCODING_LABEL[enc]} & {fixed} & {sel} \\\\")
        bil = [r["bilinear_sub_lpips"] for r in rows if r["side"] == side]
        if bil:
            qual.append(f"{side} & Bilinear & \\multicolumn{{6}}{{c}}{{{statistics.fmean(bil):.3f}}} \\\\")
        qual.append(r"\midrule" if side != SIDES[-1] else r"\bottomrule")
    qual += [r"\end{tabular}", r"\end{table*}"]
    (TABLE_DIR / "fourier_diagnostic_quality.tex").write_text("\n".join(qual) + "\n")


def write_extension_table(ext_runs: list[dict]) -> None:
    if not ext_runs:
        return
    c = EXTENSION_RULE["case"]
    lines = [
        r"\begin{table}[t]\centering\small",
        rf"\caption{{Predeclared budget extension ({c['site'].capitalize()} LR{c['side']}, full-field updates, "
        r"10\,000-step schedule; triggered because the smoothed LR holdout loss still improved by more than "
        r"$5\times10^{-4}$ between steps 4\,000 and 5\,000 for at least two seeds of either encoding; "
        r"here all three seeds of both). "
        r"Shared-window LPIPS per seed. Step 5\,000 is an intermediate checkpoint of the longer schedule, "
        r"not the 5\,000-step run. \emph{LR rule}: stop/selected step and LPIPS from the frozen LR-only rule. "
        r"\emph{HR best}: retrospective reference-optimal checkpoint, reported for diagnosis only. "
        rf"Bilinear: {ext_runs[0]['bilinear_sub_lpips']:.3f}.}}",
        r"\label{tab:fourier-extension}",
        r"\begin{tabular}{llcccccc}\toprule",
        r"Encoding & Seed & @5k & @10k & LR stop & LR sel. & LPIPS LR-sel. & HR best \\ \midrule",
    ]
    for enc in c["encodings"]:
        for r in sorted((r for r in ext_runs if r["encoding"] == enc), key=lambda r: r["seed"]):
            s, h, e = r["lr_selected"], r["hr_oracle_retrospective"], r["fixed_ext"]
            lines.append(
                f"{ENCODING_LABEL[enc]} & {r['seed']} & {e['5000']['sub_lpips']:.3f} & "
                f"{e['10000']['sub_lpips']:.3f} & {s['stop_iter'] if s.get('stopped') else '--'} & {s['iter']} & "
                f"{s['sub_lpips']:.3f} & {h['sub_lpips']:.3f} ({h['iter']}) \\\\")
        lines.append(r"\midrule" if enc != c["encodings"][-1] else r"\bottomrule")
    lines += [r"\end{tabular}", r"\end{table}"]
    (TABLE_DIR / "fourier_diagnostic_extension.tex").write_text("\n".join(lines) + "\n")


def write_figures(rows: list[dict]) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    colors = {"grid": "#1b9e77", "fourier_s2": "#7570b3", "fourier_s5": "#e7298a",
              "fourier_s10": "#d95f02", "fourier_bw10": "#666666"}
    written = []
    for key, ylabel, fname in (("sub_model_lpips", "LPIPS (shared window) ↓", "fourier_diagnostic_lpips_time"),
                               ("sub_psnr", "PSNR dB (shared window) ↑", "fourier_diagnostic_psnr_time")):
        fig, axes = plt.subplots(len(SIDES), len(SITES), figsize=(10, 8.5), sharey="row", squeeze=False)
        for i, side in enumerate(SIDES):
            for j, site in enumerate(SITES):
                ax = axes[i][j]
                for r in rows:
                    if r["side"] != side or r["site"] != site:
                        continue
                    c = r["curve"]
                    ax.plot(c["elapsed_step_seconds"], c[key], color=colors[r["encoding"]], lw=0.9, alpha=0.8,
                            label=ENCODING_LABEL[r["encoding"]].replace("$", "").replace("{=}", "=")
                            if r["seed"] == SEEDS[0] else None)
                    s = r["lr_selected"]
                    y = s["sub_lpips"] if key == "sub_model_lpips" else s["sub_psnr"]
                    ax.plot(s["elapsed_step_s"], y, "o", ms=3, color=colors[r["encoding"]])
                bil = [r["bilinear_sub_lpips" if key == "sub_model_lpips" else "bilinear_sub_psnr"]
                       for r in rows if r["side"] == side and r["site"] == site]
                if bil:
                    ax.axhline(bil[0], color="k", ls=":", lw=0.8, label="Bilinear")
                ax.set_title(f"{site.capitalize()} LR{side}", fontsize=9)
                if i == len(SIDES) - 1:
                    ax.set_xlabel("training step time (s)")
                if j == 0:
                    ax.set_ylabel(ylabel)
        handles, labels = axes[-1][-1].get_legend_handles_labels()
        for row in axes:
            for ax in row:
                h, l = ax.get_legend_handles_labels()
                if len(h) > len(handles):
                    handles, labels = h, l
        fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=8, frameon=False)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        for ext in ("pdf", "png"):
            out = FIG_DIR / f"{fname}.{ext}"
            fig.savefig(out, dpi=200)
            written.append(str(out.relative_to(ROOT)))
        plt.close(fig)
    return written


def main() -> None:
    global NAMESPACE
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--namespace", default=NAMESPACE)
    args = ap.parse_args()
    NAMESPACE = args.namespace
    audit = json.loads((ROOT / "paper" / "results" / "hash_encoding_audit.json").read_text())
    rows = [summarise_run(r, audit) for r in load_runs()]
    expected = sum(len(SITES) * len(SEEDS) * (4 if s == 64 else 5) for s in SIDES)
    agg = aggregate(rows)
    ext = extension_check(rows)
    figs = [] if args.no_figures else write_figures(rows)
    write_tables(rows, agg)
    ext_runs = extension_runs(audit) if ext["extend"] else []
    write_extension_table(ext_runs)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "namespace": NAMESPACE,
        "n_runs": len(rows), "n_expected": expected,
        "protocol": {
            "updates": "full-field (lr_tile 0)", "iters": 5000, "early_termination": False,
            "eval_every": 200, "shared_window": "central LR64 window in every field",
            "reference": "LR512 parent harmonized NIB reference and mask (--eval_reference_s2_dir)",
            "fourier_bandwidth_rule": "s(L) = 10 * L / 64 (coords normalised to [0,1) per field)",
            "fourier_matrix": "B ~ N(0,1)^{128x2} drawn with torch.Generator(seed); identical per seed",
            "lr_stop_rule": FROZEN_STOP,
            "hr_oracle_note": "retrospective diagnosis only; deployment selection is LR-only",
            "timing": "CUDA-event step time, evaluation excluded; one job per GPU (H100)",
        },
        "extension_check": ext,
        "extension_runs": ext_runs,
        "aggregate": agg,
        "runs": [{k: v for k, v in r.items() if k != "curve"} | {"curve": r["curve"]} for r in rows],
        "figures": figs,
    }, indent=1))
    print(f"{len(rows)}/{expected} runs -> {OUT_JSON.relative_to(ROOT)}; extend={ext['extend']}")


if __name__ == "__main__":
    main()
