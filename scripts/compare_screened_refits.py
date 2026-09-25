#!/usr/bin/env python3
"""Compare screened refits (``run_screened_refits.py``) with the confirmatory v5 fits, site by site.

For PSNR, masked SSIM and masked LPIPS it reports per-site seed means for both namespaces, the
paired change (screened minus v5) with a site-bootstrap 95% interval, the gain over bilinear under
each, and the mean within-site seed SD as a noise reference. Writes
``paper/results/screened_refits/compare_{namespace}.json``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BOOT = dict(seed=20260925, n=200_000)
METRICS = ("psnr", "ssim", "lpips")


def site_ci(x: np.ndarray, rng: np.random.Generator) -> list[float]:
    idx = rng.integers(0, len(x), size=(BOOT["n"], len(x)))
    return [float(q) for q in np.quantile(x[idx].mean(axis=1), [0.025, 0.975])]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--namespace", default="confirmatory_v6_screen")
    args = ap.parse_args()
    manifest = json.loads((ROOT / f"paper/results/run_manifests/{args.namespace}__run_manifest.json").read_text())

    rows: dict[str, dict] = {}
    for job in manifest["jobs"]:
        new_p = Path(job["expected_metrics_path"])
        old_p = Path(str(new_p).replace(args.namespace, manifest["parent"]))
        if not new_p.is_file():
            continue
        new, old = json.loads(new_p.read_text()), json.loads(old_p.read_text())
        assert new["base_frame"]["date"] == old["base_frame"]["date"], job["run_name"]
        r = rows.setdefault(job["city"], {"n_frames_v5": job["n_frames_v5"], "n_frames": job["n_frames"],
                                          "seeds": [], **{m: {"v5": [], "screen": [], "bil": []} for m in METRICS}})
        r["seeds"].append(job["seed"])
        for m in METRICS:
            r[m]["v5"].append(float(old[m]["model"]))
            r[m]["screen"].append(float(new[m]["model"]))
            r[m]["bil"].append(float(new[m]["bilinear"]))

    sites = sorted(rows)
    rng = np.random.default_rng(BOOT["seed"])
    summary: dict = {"namespace": args.namespace, "parent": manifest["parent"], "screen": manifest["screen"],
                     "n_sites": len(sites), "n_runs": sum(len(rows[s]["seeds"]) for s in sites), "bootstrap": BOOT}
    per_site = {}
    for s in sites:
        r = rows[s]
        per_site[s] = {"n_frames_v5": r["n_frames_v5"], "n_frames": r["n_frames"], "n_seeds": len(r["seeds"]),
                       **{m: {k: float(np.mean(v)) for k, v in r[m].items()} for m in METRICS}}
    for m in METRICS:
        v5 = np.array([per_site[s][m]["v5"] for s in sites])
        sc = np.array([per_site[s][m]["screen"] for s in sites])
        bil = np.array([per_site[s][m]["bil"] for s in sites])
        sign = -1.0 if m == "lpips" else 1.0
        delta = sc - v5
        seed_sd = {k: float(np.mean([np.std(rows[s][m][k], ddof=1) for s in sites if len(rows[s][m][k]) > 1]))
                   for k in ("v5", "screen")}
        summary[m] = {
            "v5_mean": float(v5.mean()), "screen_mean": float(sc.mean()), "bilinear_mean": float(bil.mean()),
            "delta": {"mean": float(delta.mean()), "ci95": site_ci(delta, rng),
                      "n_sites_better": int((sign * delta > 0).sum()), "max_abs": float(np.abs(delta).max())},
            "gain_v5": {"mean": float((sign * (v5 - bil)).mean()), "ci95": site_ci(sign * (v5 - bil), rng),
                        "n_sites_improved": int((sign * (v5 - bil) > 0).sum())},
            "gain_screen": {"mean": float((sign * (sc - bil)).mean()), "ci95": site_ci(sign * (sc - bil), rng),
                            "n_sites_improved": int((sign * (sc - bil) > 0).sum())},
            "within_site_seed_sd": seed_sd,
        }
    summary["per_site"] = per_site
    out = ROOT / f"paper/results/screened_refits/compare_{args.namespace}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))

    print(f"{summary['n_runs']} runs over {len(sites)} sites")
    print(f"{'site':12s} {'frames':>7s} " + " ".join(f"{m + ' v5':>10s} {m + ' scr':>10s}" for m in METRICS))
    for s in sites:
        p = per_site[s]
        print(f"{s:12s} {p['n_frames_v5']:>3d}->{p['n_frames']:<3d} "
              + " ".join(f"{p[m]['v5']:10.4f} {p[m]['screen']:10.4f}" for m in METRICS))
    for m in METRICS:
        x = summary[m]
        print(f"{m}: v5 {x['v5_mean']:.4f} screen {x['screen_mean']:.4f} delta {x['delta']['mean']:+.4f} "
              f"CI {x['delta']['ci95'][0]:+.4f}..{x['delta']['ci95'][1]:+.4f} better {x['delta']['n_sites_better']}/{len(sites)} | "
              f"gain v5 {x['gain_v5']['mean']:+.4f} ({x['gain_v5']['n_sites_improved']}) -> "
              f"screen {x['gain_screen']['mean']:+.4f} ({x['gain_screen']['n_sites_improved']}) | "
              f"seed sd {x['within_site_seed_sd']['v5']:.4f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
