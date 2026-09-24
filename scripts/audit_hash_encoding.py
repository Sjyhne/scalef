#!/usr/bin/env python3
"""Audit whether the tiny-cuda-nn HashGrid encoder actually hashes at the frozen settings.

For each LR side the encoder is built exactly as ``optimize.build_projection_and_decoder``
builds it (``resolve_hash_resolutions`` + ``HashGridTcnn``). Per level we export the
resolution tcnn realises internally, the number of allocated table entries, and the
indexing mode (dense when ``resolution**2 <= 2**log2_hashmap_size``, hashed otherwise),
and check the implied parameter total against the module's actual parameter count.

Equivalence check: the same parameter vector is loaded into a tcnn ``Grid`` of type
``Dense`` with identical levels. If every level is dense, both encoders must return the
same features (and gradients) for arbitrary coordinates.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tinycudann as tcnn  # noqa: E402

from input_projections.hashgrid_tcnn import HashGridTcnn  # noqa: E402
from optimize import resolve_hash_resolutions  # noqa: E402

FROZEN_RUN = (
    ROOT / "single_samples/asker/confirmatory_v2_lr512align/"
    "confirmatory_v2_lr512align__b0_17__lr512_b0__asker__seed6/metrics.json"
)
DEFAULT_OUT = ROOT / "paper/results/hash_encoding_audit.json"


def _next_multiple(v: int, m: int) -> int:
    return (v + m - 1) // m * m


def tcnn_levels(base: int, per_level_scale: float, n_levels: int, log2_T: int) -> list[dict]:
    """Per-level layout following tiny-cuda-nn's GridEncoding (grid_scale / grid_resolution, float32)."""
    T = 1 << log2_T
    f32 = np.float32
    log2_pls = f32(np.log2(f32(per_level_scale)))
    out = []
    for level in range(n_levels):
        scale = f32(np.exp2(f32(level) * log2_pls)) * f32(base) - f32(1.0)
        res = int(np.ceil(scale)) + 1
        dense_entries = res * res
        dense = dense_entries <= T
        entries = min(_next_multiple(dense_entries, 8), T)
        out.append({
            "level": level,
            "resolution": res,
            "cell_m_at_10m_gsd": None,
            "dense_entries": dense_entries,
            "allocated_entries": entries,
            "indexing": "dense" if dense else "hash",
            "fill_fraction_of_table": dense_entries / T,
        })
    return out


def audit_side(side: int, frozen: dict, device: torch.device, n_eq: int) -> dict:
    args = SimpleNamespace(**{**frozen, "lr_height": side, "lr_width": side})
    base_h, max_h, base_w, max_w = resolve_hash_resolutions(args)
    enc = HashGridTcnn(
        input_dim=2,
        n_levels=frozen["hash_n_levels"],
        n_features_per_level=frozen["hash_n_features_per_level"],
        log2_hashmap_size=frozen["hash_log2_hashmap_size"],
        base_resolution=base_w,
        max_resolution=max_w,
        interpolation=frozen["hash_interpolation"],
        device=device,
    )
    cfg = dict(enc._encoding.encoding_config) if hasattr(enc._encoding, "encoding_config") else {}
    pls = math.exp(math.log(max_w / base_w) / (frozen["hash_n_levels"] - 1))
    levels = tcnn_levels(base_w, pls, frozen["hash_n_levels"], frozen["hash_log2_hashmap_size"])
    for lv in levels:
        lv["cell_m_at_10m_gsd"] = 10.0 * side / lv["resolution"]
    F = frozen["hash_n_features_per_level"]
    implied = sum(lv["allocated_entries"] for lv in levels) * F
    actual = sum(p.numel() for p in enc.parameters())
    full_hash = frozen["hash_n_levels"] * (1 << frozen["hash_log2_hashmap_size"]) * F

    eq = None
    if all(lv["indexing"] == "dense" for lv in levels):
        dense_cfg = {
            "otype": "Grid", "type": "Dense",
            "n_levels": frozen["hash_n_levels"], "n_features_per_level": F,
            "base_resolution": base_w, "per_level_scale": pls,
            "interpolation": "Linear" if frozen["hash_interpolation"] == "linear" else "Smoothstep",
        }
        dense = tcnn.Encoding(2, dense_cfg).to(device)
        hp, dp = list(enc._encoding.parameters())[0], list(dense.parameters())[0]
        same_layout = hp.numel() == dp.numel()
        eq = {"dense_param_count": int(dp.numel()), "same_param_layout": bool(same_layout)}
        if same_layout:
            torch.manual_seed(0)
            with torch.no_grad():
                hp.uniform_(-1e-1, 1e-1)
                dp.copy_(hp)
            x = torch.rand(n_eq, 2, device=device) * (1 - 1e-6)
            yh, yd = enc._encoding(x), dense(x)
            g = torch.randn_like(yh.float())
            (yh.float() * g).sum().backward()
            (yd.float() * g).sum().backward()
            gh, gd = hp.grad.float().clone(), dp.grad.float()
            hp.grad = None
            (enc._encoding(x).float() * g).sum().backward()
            g_repeat = hp.grad.float()
            eq.update({
                "n_points": n_eq,
                "max_abs_feature_diff": float((yh.float() - yd.float()).abs().max().detach()),
                "max_abs_grad_diff": float((gh - gd).abs().max()),
                "max_abs_grad": float(gh.abs().max()),
                "rel_grad_diff_l2": float((gh - gd).norm() / gh.norm().clamp_min(1e-30)),
                "rel_grad_diff_l2_hash_vs_hash_repeat": float((gh - g_repeat).norm() / gh.norm().clamp_min(1e-30)),
            })
    return {
        "lr_side": side,
        "base_resolution": base_w,
        "max_resolution": max_w,
        "per_level_scale": pls,
        "tcnn_encoding_config": cfg,
        "levels": levels,
        "n_dense_levels": sum(lv["indexing"] == "dense" for lv in levels),
        "n_hashed_levels": sum(lv["indexing"] == "hash" for lv in levels),
        "encoder_params_implied": implied,
        "encoder_params_actual": actual,
        "encoder_params_if_all_levels_full_table": full_hash,
        "dense_equivalence": eq,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sides", type=int, nargs="+", default=[64, 128, 256, 512, 1024, 2048])
    ap.add_argument("--frozen-run", type=Path, default=FROZEN_RUN)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--n-eq-points", type=int, default=1 << 20)
    args = ap.parse_args()

    run = json.loads(args.frozen_run.read_text())
    a = run["provenance"]["args"]
    keys = ["hash_n_levels", "hash_n_features_per_level", "hash_log2_hashmap_size",
            "hash_base_resolution", "hash_max_resolution", "hash_max_resolution_h",
            "hash_max_resolution_w", "hash_max_resolution_mult", "hash_interpolation"]
    frozen = {k: a.get(k) for k in keys}
    device = torch.device("cuda")
    report = {
        "frozen_run": str(args.frozen_run.relative_to(ROOT)),
        "frozen_settings": frozen,
        "frozen_model_parameters": run.get("model_parameters"),
        "tinycudann_version": run["provenance"]["versions"].get("tinycudann"),
        "sides": [audit_side(s, frozen, device, args.n_eq_points) for s in args.sides],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"frozen: {frozen}")
    for s in report["sides"]:
        eq = s["dense_equivalence"]
        eqs = (f"dense-equiv feat {eq['max_abs_feature_diff']:.2e} grad rel-L2 {eq['rel_grad_diff_l2']:.2e} "
                f"(hash repeat {eq['rel_grad_diff_l2_hash_vs_hash_repeat']:.2e})"
               if eq and "max_abs_feature_diff" in eq else f"dense-equiv: {eq}")
        print(f"LR{s['lr_side']:<5d} res {s['levels'][0]['resolution']}..{s['levels'][-1]['resolution']} "
              f"dense {s['n_dense_levels']}/{len(s['levels'])}  params implied {s['encoder_params_implied']:,} "
              f"actual {s['encoder_params_actual']:,} (full table {s['encoder_params_if_all_levels_full_table']:,})  {eqs}")
    print(args.out)


if __name__ == "__main__":
    main()
