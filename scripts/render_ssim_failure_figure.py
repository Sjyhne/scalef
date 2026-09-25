#!/usr/bin/env python3
"""Representative SSIM failure (Åmli) beside a typical success (Asker) from ``analyze_ssim_failure.py``.

Each site shows its median-ΔSSIM 128x128 block (320 m) as LR base frame, bilinear, ScaleF, and NIB,
once with the fixed display mapping (0-0.4) and once with a dark display range (0-0.1) applied
identically to all four panels, followed by the per-pixel SSIM difference (ScaleF minus bilinear)
and the local-mean error of ScaleF against the reference (Gaussian sigma 1.5, the SSIM window).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.display import highlight_compress  # noqa: E402

SRC = ROOT / "paper" / "results" / "ssim_failure"
OUT = ROOT / "ScaleF_Overleaf" / "generated" / "figures" / "ssim_failure_success"
DARK_MAX = 0.1
LABELS = {"amli": "\u00c5mli (failure)", "asker": "Asker (success)"}

plt.rcParams.update({"font.family": "serif", "font.size": 7, "pdf.fonttype": 42, "ps.fonttype": 42})


def local_mean(img: np.ndarray) -> np.ndarray:
    return gaussian_filter(img.mean(axis=-1), 1.5, truncate=5 / 1.5, mode="reflect")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sites", nargs="+", default=["amli", "asker"])
    args = ap.parse_args()
    cols = ("LR (base)", "Bilinear", "ScaleF", "NIB", "$\\Delta$SSIM vs bilinear", "Local-mean error")
    fig, axes = plt.subplots(2 * len(args.sites), len(cols), figsize=(7.2, 2.55 * len(args.sites)),
                             gridspec_kw={"hspace": 0.08, "wspace": 0.04})
    meta = []
    for i, city in enumerate(args.sites):
        summary = json.loads((SRC / f"{city}.json").read_text())
        z = np.load(SRC / f"{city}_median_block.npz")
        off = float(summary["boa_offset"])
        lr = np.where(z["lr"] > 0, z["lr"] - off, z["lr"])
        panels = (lr, z["bilinear"], z["model"], z["reference"])
        d_ssim = z["ssim_model"] - z["ssim_bilinear"]
        d_mean = local_mean(z["model"]) - local_mean(z["reference"])
        for r, vmax in enumerate((0.4, DARK_MAX)):
            row = 2 * i + r
            for c, img in enumerate(panels):
                axes[row, c].imshow(highlight_compress(img, 0.0, vmax), interpolation="nearest")
            if r == 0:
                im1 = axes[row, 4].imshow(d_ssim, cmap="RdBu", vmin=-0.4, vmax=0.4, interpolation="nearest")
                im2 = axes[row, 5].imshow(d_mean, cmap="PuOr_r", vmin=-0.03, vmax=0.03, interpolation="nearest")
            else:
                axes[row, 4].axis("off")
                axes[row, 5].axis("off")
                for c, (im, lab) in enumerate(((im1, "$\\Delta$SSIM"), (im2, "reflectance")), start=4):
                    cax = axes[row, c].inset_axes([0.08, 0.86, 0.84, 0.08])
                    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
                    cb.ax.tick_params(labelsize=6, length=2, pad=1)
                    cb.set_label(lab, fontsize=6, labelpad=1)
            for ax in axes[row]:
                ax.set_xticks([])
                ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(False)
            axes[row, 0].set_ylabel(f"{LABELS.get(city, city)}\n0\u2013{vmax:g}", fontsize=7)
        b = summary["median_block"]
        site = summary["ssim_check_masked_ssim"]
        axes[2 * i + 1, 4].text(
            1.02, 0.28,
            f"Block: $\\Delta$SSIM {b['d_ssim']:+.3f}, $\\Delta$LPIPS {-b['d_lpips']:+.3f}\n"
            f"Site SSIM: ScaleF {site['model']:.3f}, bilinear {site['bilinear']:.3f}",
            transform=axes[2 * i + 1, 4].transAxes, ha="center", va="center", fontsize=6)
        meta.append({"site": city, "block": [b["by"], b["bx"]], "block_px": 128, "d_ssim": b["d_ssim"],
                     "d_lpips_model_minus_bilinear": -b["d_lpips"], "site_ssim": site})
    for c, t in enumerate(cols):
        axes[0, c].set_title(t, fontsize=7, pad=3)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    OUT.with_suffix(".json").write_text(json.dumps({"display": ["highlight_compress(0, 0.4)", f"highlight_compress(0, {DARK_MAX})"],
                                                    "rows": meta}, indent=1) + "\n")
    print(f"wrote {OUT.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
