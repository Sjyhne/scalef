"""GNLL variance visualizations."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from models.lr_alignment import default_lr_align_args

from eval.rgb import (
    _fetch_lr_pre_standardize_hwc_np,
    _fetch_lr_supervision_batch,
    _hwc_rgb_for_imshow,
    _std_map_2d,
)

def visualize_lr_variance(model, train_data, device, output_dir, sample_id):
    """
    Visualize learned σ next to each LR frame (native LR resolution) for GNLL validation.

    Writes ``sample_XXX_variance_map.png`` (LR | σ), ``sample_XXX_variance_analysis.png``
    (LR | σ, HR GT | model LR pred), and ``lr_variance_grid_2x8.png``.
    """
    use_gnll_loss = model.use_gnll
    if not use_gnll_loss:
        print("Warning: visualize_lr_variance called but model does not use GNLL")
        return

    model.eval()
    with torch.no_grad():
        # Get HR coordinates for inference
        hr_coords = train_data.get_hr_coordinates().unsqueeze(0).to(device)
        hr_image = train_data.get_original_hr().unsqueeze(0).to(device)

        # Create output directory for variance visualizations
        variance_dir = output_dir / "variance_visualizations"
        variance_dir.mkdir(exist_ok=True)

        # Get number of LR samples based on dataset type
        if hasattr(train_data, "num_samples"):
            num_samples = train_data.num_samples
        elif hasattr(train_data, "lr_paths"):
            num_samples = len(train_data.lr_paths)
        else:
            print(
                "Warning: Cannot determine number of LR samples. Skipping variance visualization."
            )
            return

        print(f"Creating variance visualizations for {num_samples} LR samples...")

        # Collect data for 2x8 grid visualization
        lr_samples_for_grid = []
        variance_maps_for_grid = []
        global_vmin = None
        global_vmax = None

        lr_align_args = default_lr_align_args(
            lr_degradation=str(getattr(model, "lr_degradation", "area"))
        )

        # Process each LR sample individually
        for i in range(num_samples):
            sample_id_tensor = torch.tensor([i]).to(device)
            lr_batch = _fetch_lr_supervision_batch(train_data, i, device)
            lr_np = _fetch_lr_pre_standardize_hwc_np(train_data, i)

            # GNLL variance at LR resolution (same grid as supervision)
            output_lr, _, variance = model(
                hr_coords,
                sample_id_tensor,
                lr_frames=lr_batch,
                lr_align_args=lr_align_args,
            )

            if isinstance(variance, list):
                try:
                    variance = torch.stack(variance, dim=0)
                except Exception:
                    variance = None
            if variance is None:
                variance = torch.full_like(output_lr, 1e-6)

            output_lr_np = output_lr.squeeze(0).cpu().numpy()
            variance_np = variance.squeeze(0).cpu().numpy()
            hr_np = hr_image.squeeze(0).cpu().numpy()
            hr_np = np.clip(hr_np, 0, 1)
            if variance_np.min() < 0:
                variance_np = np.maximum(variance_np, 0)

            std_np = np.sqrt(variance_np)
            std_map = _std_map_2d(std_np)
            lr_vis = _hwc_rgb_for_imshow(lr_np, stretch=False)
            pred_phys = output_lr_np
            try:
                std_i = train_data.get_lr_std(i)
                mean_i = train_data.get_lr_mean(i)
                if torch.is_tensor(std_i):
                    std_i = std_i.cpu().numpy()
                if torch.is_tensor(mean_i):
                    mean_i = mean_i.cpu().numpy()
                if np.ndim(std_i) == 1:
                    std_i = std_i.reshape(1, 1, -1)
                if np.ndim(mean_i) == 1:
                    mean_i = mean_i.reshape(1, 1, -1)
                pred_phys = output_lr_np * std_i + mean_i
            except (AttributeError, TypeError, IndexError):
                pass
            pred_vis = _hwc_rgb_for_imshow(pred_phys, stretch=False)

            # Symmetric color scale around 1 for sqrt(variance) in standardized units
            max_deviation = max(abs(std_map.max() - 1), abs(std_map.min() - 1), 1e-6)
            vmin = max(0.0, 1.0 - max_deviation)
            vmax = 1.0 + max_deviation
            if global_vmin is None:
                global_vmin, global_vmax = vmin, vmax
            else:
                dev = max(abs(global_vmax - 1), abs(global_vmin - 1), max_deviation)
                global_vmin = 1.0 - dev
                global_vmax = 1.0 + dev

            lr_samples_for_grid.append(lr_vis.copy())
            variance_maps_for_grid.append(std_map.copy())

            # Primary panel: LR | std at native LR resolution
            fig_pair, axes_pair = plt.subplots(1, 2, figsize=(10, 5))
            axes_pair[0].imshow(lr_vis)
            axes_pair[0].set_title(f"LR pre-standardize (frame {i})", fontsize=11, fontweight="bold")
            axes_pair[0].axis("off")
            im_pair = axes_pair[1].imshow(std_map, cmap="Blues", vmin=vmin, vmax=vmax)
            axes_pair[1].set_title(f"Learned σ (frame {i})", fontsize=11, fontweight="bold")
            axes_pair[1].axis("off")
            cbar_pair = plt.colorbar(im_pair, ax=axes_pair[1], fraction=0.046, pad=0.04)
            cbar_pair.set_label("σ (standardized units)", rotation=270, labelpad=12)
            plt.tight_layout(pad=1.0)
            variance_map_path = variance_dir / f"sample_{i:03d}_variance_map.png"
            plt.savefig(variance_map_path, bbox_inches="tight", pad_inches=0.1, dpi=300)
            plt.close(fig_pair)

            # Detailed 2×2: LR | σ, then HR GT | model LR prediction
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))
            axes[0, 0].imshow(lr_vis)
            axes[0, 0].set_title(f"LR pre-standardize (frame {i})", fontsize=12, fontweight="bold")
            axes[0, 0].axis("off")

            im_var = axes[0, 1].imshow(std_map, cmap="Blues", vmin=vmin, vmax=vmax)
            axes[0, 1].set_title(f"Learned σ (frame {i})", fontsize=12, fontweight="bold")
            axes[0, 1].axis("off")
            cbar = plt.colorbar(im_var, ax=axes[0, 1], fraction=0.046, pad=0.04)
            cbar.set_label("σ (standardized units)", rotation=270, labelpad=15)

            axes[1, 0].imshow(hr_np)
            axes[1, 0].set_title("HR ground truth", fontsize=12, fontweight="bold")
            axes[1, 0].axis("off")

            axes[1, 1].imshow(pred_vis)
            axes[1, 1].set_title(f"Model LR prediction (frame {i})", fontsize=12, fontweight="bold")
            axes[1, 1].axis("off")

            plt.tight_layout(pad=2.0)
            variance_path = variance_dir / f"sample_{i:03d}_variance_analysis.png"
            plt.savefig(variance_path, bbox_inches="tight", pad_inches=0.1, dpi=300)
            plt.close()

            fig_lr_only = plt.figure(figsize=(5, 5))
            plt.imshow(lr_vis)
            plt.axis("off")
            plt.tight_layout(pad=0)
            plt.savefig(variance_dir / f"sample_{i:03d}_lr_sample.png", bbox_inches="tight", pad_inches=0, dpi=300)
            plt.close()

            np.save(variance_dir / f"sample_{i:03d}_std.npy", std_np)
            np.save(variance_dir / f"sample_{i:03d}_variance.npy", variance_np)
            np.save(variance_dir / f"sample_{i:03d}_lr_target.npy", lr_np)
            np.save(variance_dir / f"sample_{i:03d}_output_lr.npy", output_lr_np)

        # Create a summary visualization showing all variance maps side by side
        create_variance_summary(train_data, variance_dir, device)

        # Create 2x8 grid: top row = LR samples, bottom row = std maps
        if len(lr_samples_for_grid) >= 8 and len(variance_maps_for_grid) >= 8:
            create_lr_variance_grid(
                lr_samples_for_grid[:8],
                variance_maps_for_grid[:8],
                global_vmin,
                global_vmax,
                variance_dir,
            )

        print(f"Standard deviation visualizations saved to {variance_dir}")


def create_lr_variance_grid(lr_samples, variance_maps, vmin, vmax, variance_dir):
    """
    Create a 2x8 grid visualization: top row = LR samples, bottom row = std maps.

    Args:
        lr_samples: List of LR sample images (numpy arrays) or None
        variance_maps: List of std maps (numpy arrays) - note: variable name kept for compatibility
        vmin: Minimum value for std color scale
        vmax: Maximum value for std color scale
        variance_dir: Directory to save the grid
    """
    from matplotlib.patches import Rectangle

    if len(variance_maps) < 8:
        print(f"Warning: Only {len(variance_maps)} samples available, need 8 for grid")
        return

    # Ensure vmin is at least 0 (std is sqrt(variance) which is always >= 0)
    vmin = max(0, vmin)

    fig, axes = plt.subplots(
        2, 8, figsize=(24, 8)
    )  # Increased height from 6 to 8 for less compact y direction

    # Top row: LR samples (native LR resolution)
    for i in range(8):
        ax = axes[0, i]
        if lr_samples[i] is not None:
            ax.imshow(lr_samples[i])
            ax.set_title(f"LR {i}", fontsize=9)
        else:
            ax.text(0.5, 0.5, f"LR {i}", ha="center", va="center", transform=ax.transAxes)
        # Get image bounds for border
        if lr_samples[i] is not None:
            h, w = lr_samples[i].shape[:2]
            rect = Rectangle(
                (-0.5, -0.5), w, h, fill=False, edgecolor="gray", linewidth=0.5, clip_on=False
            )
            ax.add_patch(rect)
        ax.axis("off")

    # Bottom row: Standard deviation maps
    for i in range(8):
        ax = axes[1, i]
        im = ax.imshow(variance_maps[i], cmap="Blues", vmin=vmin, vmax=vmax)
        ax.set_title(f"σ {i}", fontsize=9)
        # Get image bounds for border
        h, w = variance_maps[i].shape[:2]
        rect = Rectangle(
            (-0.5, -0.5), w, h, fill=False, edgecolor="gray", linewidth=0.5, clip_on=False
        )
        ax.add_patch(rect)
        ax.axis("off")

    # Adjust layout to leave room for colorbar on the right
    # Use tight_layout first to get proper spacing, then adjust for colorbar
    plt.tight_layout(pad=1.0)

    # Get the position of the bottom-right subplot to align colorbar
    # The bottom row is axes[1, 7] (last std map)
    bottom_right_ax = axes[1, 7]
    bbox = bottom_right_ax.get_position()

    # Position colorbar to the right of the last std map
    # [left, bottom, width, height] in figure coordinates
    cbar_width = 0.015
    cbar_left = bbox.x1 + 0.02  # Small gap after the last subplot
    cbar_bottom = bbox.y0  # Align with bottom of bottom row
    cbar_height = bbox.height  # Match height of bottom row subplots

    cbar_ax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Standard Deviation (1 = neutral)", rotation=270, labelpad=20)
    grid_path = variance_dir / "lr_variance_grid_2x8.png"
    plt.savefig(grid_path, bbox_inches="tight", pad_inches=0.1, dpi=300)
    plt.close()
    print(f"Created 2x8 grid visualization: {grid_path}")


def create_variance_summary(train_data, variance_dir, device):
    """
    Create a summary visualization showing all variance maps in a grid.
    """
    # This would require loading all the saved variance maps and creating a grid
    # For now, we'll create a simple summary
    summary_path = variance_dir / "variance_summary.txt"

    # Get number of LR samples based on dataset type
    if hasattr(train_data, "num_samples"):
        num_samples = train_data.num_samples
    elif hasattr(train_data, "lr_paths"):
        num_samples = len(train_data.lr_paths)
    else:
        num_samples = "Unknown"

    with open(summary_path, "w") as f:
        f.write("Standard Deviation Analysis Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Number of LR samples: {num_samples}\n")
        f.write(
            "Each sample_XXX_variance_map.png shows LR observation | learned σ at LR resolution.\n"
        )
        f.write(
            "sample_XXX_variance_analysis.png adds HR GT and model LR prediction for context.\n"
        )
        f.write(f"High σ regions indicate where the model assigns higher observation noise.\n")
        f.write(f"Standard deviation is computed as sqrt(variance) for easier interpretation.\n")
        f.write(f"Check individual sample_XXX_variance_analysis.png files for detailed analysis.\n")

    print(f"Variance summary saved to {summary_path}")
