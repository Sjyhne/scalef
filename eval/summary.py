"""Multi-sample aggregate reports."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

def create_summary_visualization(all_results, output_dir):
    """Create summary visualization showing metrics across all samples."""
    if not all_results:
        return

    # Extract metrics from nested structure
    sample_indices = [r["sample_idx"] for r in all_results]
    model_psnr = [r["image_metrics"]["model_psnr"] for r in all_results]
    bilinear_psnr = [r["image_metrics"]["bilinear_psnr"] for r in all_results]
    psnr_improvement = [r["image_metrics"]["psnr_improvement"] for r in all_results]
    model_ssim = [r["image_metrics"]["model_ssim"] for r in all_results]
    bilinear_ssim = [r["image_metrics"]["bilinear_ssim"] for r in all_results]
    ssim_improvement = [r["image_metrics"]["ssim_improvement"] for r in all_results]
    model_lpips = [r["image_metrics"]["model_lpips"] for r in all_results]
    bilinear_lpips = [r["image_metrics"]["bilinear_lpips"] for r in all_results]
    lpips_improvement = [r["image_metrics"]["lpips_improvement"] for r in all_results]
    trans_loss_values = [
        (
            r["training_metrics"]["final_trans_loss"]
            if r["training_metrics"]["final_trans_loss"] is not None
            else 0.0
        )
        for r in all_results
    ]

    # Create summary plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # PSNR comparison
    axes[0, 0].bar(sample_indices, model_psnr, alpha=0.7, label="Model", color="blue")
    axes[0, 0].bar(sample_indices, bilinear_psnr, alpha=0.7, label="Bilinear", color="orange")
    axes[0, 0].set_xlabel("Sample Index")
    axes[0, 0].set_ylabel("PSNR (dB)")
    axes[0, 0].set_title("PSNR Comparison Across Samples")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # PSNR improvement
    colors = ["green" if x > 0 else "red" for x in psnr_improvement]
    axes[0, 1].bar(sample_indices, psnr_improvement, color=colors, alpha=0.7)
    axes[0, 1].axhline(y=0, color="black", linestyle="-", alpha=0.5)
    axes[0, 1].set_xlabel("Sample Index")
    axes[0, 1].set_ylabel("PSNR Improvement (dB)")
    axes[0, 1].set_title("PSNR Improvement (Model - Bilinear)")
    axes[0, 1].grid(True, alpha=0.3)

    # Transformation Loss
    axes[0, 2].bar(sample_indices, trans_loss_values, alpha=0.7, color="teal")
    axes[0, 2].set_xlabel("Sample Index")
    axes[0, 2].set_ylabel("Transformation Loss")
    axes[0, 2].set_title("Final Transformation Loss Across Samples")
    axes[0, 2].grid(True, alpha=0.3)

    # SSIM comparison
    axes[1, 0].bar(sample_indices, model_ssim, alpha=0.7, label="Model", color="purple")
    axes[1, 0].bar(sample_indices, bilinear_ssim, alpha=0.7, label="Bilinear", color="orange")
    axes[1, 0].set_xlabel("Sample Index")
    axes[1, 0].set_ylabel("SSIM")
    axes[1, 0].set_title("SSIM Comparison Across Samples")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # LPIPS comparison
    axes[1, 1].bar(sample_indices, model_lpips, alpha=0.7, label="Model", color="brown")
    axes[1, 1].bar(sample_indices, bilinear_lpips, alpha=0.7, label="Bilinear", color="orange")
    axes[1, 1].set_xlabel("Sample Index")
    axes[1, 1].set_ylabel("LPIPS")
    axes[1, 1].set_title("LPIPS Comparison Across Samples")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Overall improvement metrics
    axes[1, 2].bar(sample_indices, psnr_improvement, alpha=0.7, color="green")
    axes[1, 2].axhline(y=0, color="black", linestyle="-", alpha=0.5)
    axes[1, 2].set_xlabel("Sample Index")
    axes[1, 2].set_ylabel("Improvement (dB)")
    axes[1, 2].set_title("PSNR Improvement per Sample")
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "summary_metrics.png", bbox_inches="tight", pad_inches=0.1, dpi=300)
    plt.close()

    # Create box plots for aggregated metrics
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # PSNR box plot
    axes[0].boxplot([model_psnr, bilinear_psnr], labels=["Model", "Bilinear"])
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].set_title("PSNR Distribution Comparison")
    axes[0].grid(True, alpha=0.3)

    # SSIM box plot
    axes[1].boxplot([model_ssim, bilinear_ssim], labels=["Model", "Bilinear"])
    axes[1].set_ylabel("SSIM")
    axes[1].set_title("SSIM Distribution Comparison")
    axes[1].grid(True, alpha=0.3)

    # LPIPS box plot
    axes[2].boxplot([model_lpips, bilinear_lpips], labels=["Model", "Bilinear"])
    axes[2].set_ylabel("LPIPS")
    axes[2].set_title("LPIPS Distribution Comparison")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        output_dir / "metrics_distribution.png", bbox_inches="tight", pad_inches=0.1, dpi=300
    )
    plt.close()

    # Calculate and save aggregated statistics
    summary_stats = {
        "total_samples": len(all_results),
        "psnr": {
            "model_mean": np.mean(model_psnr),
            "model_std": np.std(model_psnr),
            "model_min": np.min(model_psnr),
            "model_max": np.max(model_psnr),
            "bilinear_mean": np.mean(bilinear_psnr),
            "bilinear_std": np.std(bilinear_psnr),
            "bilinear_min": np.min(bilinear_psnr),
            "bilinear_max": np.max(bilinear_psnr),
            "improvement_mean": np.mean(psnr_improvement),
            "improvement_std": np.std(psnr_improvement),
            "improvement_min": np.min(psnr_improvement),
            "improvement_max": np.max(psnr_improvement),
        },
        "ssim": {
            "model_mean": np.mean(model_ssim),
            "model_std": np.std(model_ssim),
            "model_min": np.min(model_ssim),
            "model_max": np.max(model_ssim),
            "bilinear_mean": np.mean(bilinear_ssim),
            "bilinear_std": np.std(bilinear_ssim),
            "bilinear_min": np.min(bilinear_ssim),
            "bilinear_max": np.max(bilinear_ssim),
            "improvement_mean": np.mean(ssim_improvement),
            "improvement_std": np.std(ssim_improvement),
            "improvement_min": np.min(ssim_improvement),
            "improvement_max": np.max(ssim_improvement),
        },
        "lpips": {
            "model_mean": np.mean(model_lpips),
            "model_std": np.std(model_lpips),
            "model_min": np.min(model_lpips),
            "model_max": np.max(model_lpips),
            "bilinear_mean": np.mean(bilinear_lpips),
            "bilinear_std": np.std(bilinear_lpips),
            "bilinear_min": np.min(bilinear_lpips),
            "bilinear_max": np.max(bilinear_lpips),
            "improvement_mean": np.mean(lpips_improvement),
            "improvement_std": np.std(lpips_improvement),
            "improvement_min": np.min(lpips_improvement),
            "improvement_max": np.max(lpips_improvement),
        },
        "transformation_loss": {
            "mean": np.mean(trans_loss_values),
            "std": np.std(trans_loss_values),
            "min": np.min(trans_loss_values),
            "max": np.max(trans_loss_values),
        },
    }

    # Save aggregated statistics to JSON
    with open(output_dir / "summary_statistics.json", "w") as f:
        json.dump(summary_stats, f, indent=2)

    # Save human-readable summary
    summary_text = f"""Multi-Sample Super-Resolution Results Summary
================================================

Total Samples Processed: {len(all_results)}

PSNR Results (dB):
------------------
Model Output:
  Mean: {summary_stats['psnr']['model_mean']:.2f} ± {summary_stats['psnr']['model_std']:.2f}
  Range: {summary_stats['psnr']['model_min']:.2f} - {summary_stats['psnr']['model_max']:.2f}

Bilinear Baseline:
  Mean: {summary_stats['psnr']['bilinear_mean']:.2f} ± {summary_stats['psnr']['bilinear_std']:.2f}
  Range: {summary_stats['psnr']['bilinear_min']:.2f} - {summary_stats['psnr']['bilinear_max']:.2f}

PSNR Improvement (Model - Bilinear):
  Mean: {summary_stats['psnr']['improvement_mean']:.2f} ± {summary_stats['psnr']['improvement_std']:.2f}
  Range: {summary_stats['psnr']['improvement_min']:.2f} - {summary_stats['psnr']['improvement_max']:.2f}

SSIM Results:
-------------
Model Output:
  Mean: {summary_stats['ssim']['model_mean']:.4f} ± {summary_stats['ssim']['model_std']:.4f}
  Range: {summary_stats['ssim']['model_min']:.4f} - {summary_stats['ssim']['model_max']:.4f}

Bilinear Baseline:
  Mean: {summary_stats['ssim']['bilinear_mean']:.4f} ± {summary_stats['ssim']['bilinear_std']:.4f}
  Range: {summary_stats['ssim']['bilinear_min']:.4f} - {summary_stats['ssim']['bilinear_max']:.4f}

SSIM Improvement (Model - Bilinear):
  Mean: {summary_stats['ssim']['improvement_mean']:.4f} ± {summary_stats['ssim']['improvement_std']:.4f}
  Range: {summary_stats['ssim']['improvement_min']:.4f} - {summary_stats['ssim']['improvement_max']:.4f}

LPIPS Results:
--------------
Model Output:
  Mean: {summary_stats['lpips']['model_mean']:.4f} ± {summary_stats['lpips']['model_std']:.4f}
  Range: {summary_stats['lpips']['model_min']:.4f} - {summary_stats['lpips']['model_max']:.4f}

Bilinear Baseline:
  Mean: {summary_stats['lpips']['bilinear_mean']:.4f} ± {summary_stats['lpips']['bilinear_std']:.4f}
  Range: {summary_stats['lpips']['bilinear_min']:.4f} - {summary_stats['lpips']['bilinear_max']:.4f}

LPIPS Improvement (Bilinear - Model):
  Mean: {summary_stats['lpips']['improvement_mean']:.4f} ± {summary_stats['lpips']['improvement_std']:.4f}
  Range: {summary_stats['lpips']['improvement_min']:.4f} - {summary_stats['lpips']['improvement_max']:.4f}

Transformation Loss Results:
----------------------------
Final Transformation Loss:
  Mean: {summary_stats['transformation_loss']['mean']:.6f} ± {summary_stats['transformation_loss']['std']:.6f}
  Range: {summary_stats['transformation_loss']['min']:.6f} - {summary_stats['transformation_loss']['max']:.6f}

Files Generated:
- summary_metrics.png: Bar charts comparing metrics across samples
- metrics_distribution.png: Box plots showing metric distributions
- summary_statistics.json: Detailed numerical statistics
- sample_XXX/: Individual results for each sample
"""

    with open(output_dir / "summary_report.txt", "w") as f:
        f.write(summary_text)

    print(f"\n{'='*60}")
    print("Summary Statistics")
    print(f"{'='*60}")
    print(
        f"PSNR Improvement: {summary_stats['psnr']['improvement_mean']:.2f} ± {summary_stats['psnr']['improvement_std']:.2f} dB"
    )
    print(
        f"SSIM Improvement: {summary_stats['ssim']['improvement_mean']:.4f} ± {summary_stats['ssim']['improvement_std']:.4f}"
    )
    print(
        f"LPIPS Improvement: {summary_stats['lpips']['improvement_mean']:.4f} ± {summary_stats['lpips']['improvement_std']:.4f}"
    )
    print(
        f"Average Transformation Loss: {summary_stats['transformation_loss']['mean']:.6f} ± {summary_stats['transformation_loss']['std']:.6f}"
    )
    print(f"{'='*60}\n")
    print(
        f"📊 Summary visualizations saved to {output_dir}/summary_metrics.png and {output_dir}/metrics_distribution.png"
    )
    print(
        f"📈 Aggregated statistics saved to {output_dir}/summary_statistics.json and {output_dir}/summary_report.txt"
    )
