"""
SuperF SR Demo - Gradio Interface
Satellite Image Super-Resolution using Sentinel-2 data
"""

import hashlib
import json
import logging
import os
import pickle
import subprocess
import sys
import tempfile
import time

# Set up logging (already imported above)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.info("Starting application...")

# Set OpenMP environment variables before importing numpy/torch to avoid libgomp errors
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
# Ensure rasterio/GDAL can stream unsigned Sentinel COGs in Spaces
os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".TIF,.tif,.JP2,.jp2,.png,.PNG")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")
# Don't set GDAL_DATA or PROJ_LIB - let rasterio/pyproj use their bundled data

import gradio as gr
import numpy as np
import torch
from input_projections.coord_utils import make_normalized_grid

# Configure PyTorch threading to avoid conflicts
torch.set_num_threads(1)
logger.info("PyTorch imported")
import io
import math
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import folium
import matplotlib.pyplot as plt
import rasterio
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from folium import plugins
from PIL import Image
from pystac_client import Client
from rasterio.warp import transform as transform_coords
from rasterio.windows import Window
from tqdm import tqdm

logger.info("All imports successful")


# Helper functions for INR
def get_learnable_transforms(num_samples, coordinate_dim, zeros=True, freeze_first=False):
    """Create ParameterList of learnable transform parameters."""
    params = nn.ParameterList(
        [nn.Parameter(torch.zeros(coordinate_dim)) for _ in range(num_samples)]
    )
    if freeze_first:
        params[0].requires_grad = False
    return params


def _prepare_cloudmask_input(data):
    """Normalize and reshape input for omnicloudmask.

    Returns array with shape (3, H, W), float32, values in [0, 1].
    """
    x = np.asarray(data, dtype=np.float32)
    if x.max() > 1.5:
        x = np.clip(x / 255.0, 0.0, 1.0)

    if x.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape: {x.shape}")
    if x.shape[-1] == 3:
        x = np.transpose(x, (2, 0, 1))
    if x.shape[0] != 3:
        raise ValueError(f"Expected shape (3, H, W) or (H, W, 3), got {x.shape}")
    if x.shape[1] < 32 or x.shape[2] < 32:
        raise ValueError(f"Image must be >= 32x32 pixels, got {x.shape[1]}x{x.shape[2]}")
    return x


def _run_cloudmask_prediction(x, device_hint="cuda"):
    """Run omnicloudmask prediction with API fallbacks."""
    from omnicloudmask import predict_from_array

    for param_name in ["inference_device", "device", None]:
        try:
            if param_name:
                pred_mask = predict_from_array(x, **{param_name: device_hint})
            else:
                pred_mask = predict_from_array(x)
            return pred_mask, f"{device_hint.upper()} ({param_name or 'default'})"
        except TypeError:
            continue
    raise RuntimeError("All omnicloudmask API variants failed")


def run_omnicloudmask_gpu(x):
    """Execute omnicloudmask on GPU. Expects pre-prepared input (3, H, W)."""
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA-capable device available")

    pred_mask, device_used = _run_cloudmask_prediction(x, "cuda")
    cloud_percentage = (np.sum(pred_mask > 0) / pred_mask.size) * 100.0
    return cloud_percentage, device_used


def create_output_directory(base_dir="output_cog"):
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    temp_path = tempfile.mkdtemp(prefix="session_", dir=str(base_dir))
    return Path(temp_path)


class MLP(nn.Module):
    """Simple MLP decoder."""

    def __init__(self, input_dim=2, hidden_dim=256, depth=4, output_dim=3):
        super().__init__()
        self.output_dim = output_dim
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class FourierFeatures(nn.Module):
    def __init__(self, input_dim, mapping_size=256, scale=10.0):
        super().__init__()
        self.input_dim = input_dim
        self.mapping_size = mapping_size
        self.scale = scale
        B = torch.randn(input_dim, mapping_size // 2) * scale
        self.register_buffer("B", B)

    def forward(self, x):
        x_proj = 2 * np.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class INR(nn.Module):
    def __init__(self, input_projection, decoder, num_samples, coordinate_dim=2, use_gnll=False):
        super(INR, self).__init__()

        self.input_projection = input_projection
        self.decoder = decoder

        self.coordinate_dim = coordinate_dim
        self.num_samples = num_samples
        self.use_gnll = use_gnll

        # Always use direct affine parameters with base frame frozen.
        identity = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]], dtype=torch.float32)
        self.affine_params = nn.ParameterList(
            [
                nn.Parameter(identity.clone(), requires_grad=(i != 0))
                for i in range(num_samples)
            ]
        )

        # Always use color transforms
        ct = nn.ModuleList([nn.Linear(1, 1) for _ in range(3)])
        self.color_transforms = nn.ModuleList([ct for _ in range(num_samples)])

        self.color_transforms[0].requires_grad = False

        # Initialize all biases to 0
        for color_transform in self.color_transforms:
            for ct in color_transform:
                ct.bias.data.zero_()

        # Initialize all weights to 1
        for color_transform in self.color_transforms:
            for ct in color_transform:
                ct.weight.data.fill_(1)

        if self.use_gnll:
            # MLP variance predictor
            self.variance_predictor = nn.Sequential(
                nn.Linear(3 + 3, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 3)
            )

    def get_affine_transform(self, sample_id):
        return self.get_direct_affine(sample_id)

    def get_direct_affine(self, sample_id):
        B = sample_id.shape[0]
        params = [self.affine_params[idx.item()] for idx in sample_id]
        affine = torch.cat(params, dim=0)  # [B, 6]
        A = affine.view(B, 2, 3)

        assert A.shape == (B, 2, 3), f"A.shape: {A.shape}"

        return A

    def apply_affine(self, coords, A):
        B, H, W, C = coords.shape

        coords = coords.reshape(B, -1, C)  # [B, H*W, C]

        homogenous_coords = torch.cat(
            [coords, torch.ones(B, coords.shape[1], 1, device=coords.device)], dim=2
        )  # B, HW, 3 - Homogeneous coordinates
        transformed_coords = torch.matmul(homogenous_coords, A.mT)  # B, HW, 2

        return transformed_coords.reshape(B, H, W, C)

    def apply_color_transform(self, x, sample_idx):
        """Apply per-channel color scaling."""
        result = x.clone()

        for i, idx in enumerate(sample_idx):
            if idx != 0:  # Skip reference sample
                for channel in range(3):
                    transformed = self.color_transforms[idx][channel](
                        x[i, :, :, channel].unsqueeze(-1)
                    )
                    result[i, :, :, channel] = transformed.squeeze(-1)

        return result

    def forward(self, coords, sample_idx=None, scale_factor=None, training=True, lr_frames=None):
        B, H, W, C = coords.shape

        # Initialize shift lists
        dx_list = None
        dy_list = None

        if training:
            A = self.get_affine_transform(sample_idx)  # [B, 2, 3]
            dx_list = A[:, 0, 2]
            dy_list = A[:, 1, 2]
            coords = self.apply_affine(coords, A)

        if self.input_projection is not None:
            coords = self.input_projection(coords)

        # Reshape coordinates from [B, H, W, C] to [B*H*W, C] for decoder
        B, H, W, C = coords.shape
        coords_flat = coords.reshape(B * H * W, C)
        output_flat = self.decoder(coords_flat)
        # Reshape output back to [B, H, W, output_dim]
        output = output_flat.reshape(B, H, W, -1)

        logvars = None
        if self.use_gnll:
            # Check if output has enough channels for logvars
            if output.shape[-1] >= 3 + self.num_samples * 3:
                rgb = output[:, :, :, :3]

                # We need to partition the logvars into num_samples of 3 channels each
                logvars_list = []

                for i in range(self.num_samples):
                    logvars_list.append(output[:, :, :, 3 + i * 3 : 6 + i * 3])

                logvars = torch.stack(logvars_list, dim=0)  # [num_samples, B, H, W, 3]

                output = rgb
            else:
                # If decoder doesn't output logvars, create dummy logvars
                B, H, W, _ = output.shape
                logvars = torch.zeros(self.num_samples, B, H, W, 3, device=output.device)

        # Always apply color transform
        output = self.apply_color_transform(output, sample_idx)

        shifts = [dx_list, dy_list] if dx_list is not None else None

        if training:  # pool the supersampled output to the LR resolution
            if scale_factor.unique().shape[0] == 1:
                scale_factor = scale_factor.unique().item()
            else:
                raise ValueError(
                    "Not implemented functionality that supports multiple scale factors in the same batch"
                )

            output = F.interpolate(
                output.permute(0, 3, 1, 2), scale_factor=scale_factor, mode="area"
            ).permute(0, 2, 3, 1)

        if self.use_gnll and lr_frames is not None:
            variances = []
            # logvars has shape [num_samples, B, H, W, 3] where first dim is LR frame index
            # We need to select logvars for each batch element
            if logvars is not None and logvars.shape[1] > 0:
                batch_size = logvars.shape[1]
                selected_logvars = []
                for b in range(batch_size):
                    # Map sample_idx to LR frame index using modulo
                    frame_idx = (sample_idx[b] % self.num_samples).item()
                    # Clamp to valid range [0, num_samples-1]
                    frame_idx = min(max(0, frame_idx), self.num_samples - 1)
                    selected_logvars.append(logvars[frame_idx, b])  # [H, W, 3]
                if selected_logvars:
                    variances = torch.stack(selected_logvars, dim=0)  # [B, H, W, 3]
                    variances = F.interpolate(
                        variances.permute(0, 3, 1, 2), scale_factor=scale_factor
                    ).permute(0, 2, 3, 1)
                    variances = torch.exp(variances)
                else:
                    # Fallback: create dummy variances
                    B, H, W = output.shape[:3]
                    variances = torch.ones(B, H, W, 3, device=output.device) * 0.1
            else:
                # Fallback: create dummy variances
                B, H, W = output.shape[:3]
                variances = torch.ones(B, H, W, 3, device=output.device) * 0.1

            return output, shifts, variances
        else:
            return output, shifts


CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def compute_collection_cache_key(parameters):
    serialized = json.dumps(parameters, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def get_cache_paths(cache_key):
    cache_dir = CACHE_DIR / cache_key
    return cache_dir, cache_dir / "TCI", cache_dir / "metadata.json"


def load_cache_metadata(metadata_path):
    if metadata_path.exists():
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.warning("Failed to load cache metadata %s: %s", metadata_path, exc)
    return {}


def save_cache_metadata(metadata_path, data):
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def get_cloud_cover(rgn_array):
    """Get cloud cover using omnicloudmask, preferring GPU but falling back to CPU."""
    x = _prepare_cloudmask_input(rgn_array)

    # Try GPU path first
    try:
        cloud_percentage, device_used = run_omnicloudmask_gpu(x)
        logger.info("Cloud cover from omnicloudmask: %.2f%% (%s)", cloud_percentage, device_used)
        return cloud_percentage
    except Exception as gpu_err:
        logger.warning("GPU omnicloudmask failed (%s); falling back to subprocess.", gpu_err)

    with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".pkl") as tmp_data:
        pickle.dump(x, tmp_data)
        tmp_data_path = tmp_data.name

    try:
        script_path = os.path.join(os.path.dirname(__file__), "run_omnicloudmask.py")
        minimal_env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
        }
        for key in ["HOME", "USER", "LANG", "LC_ALL"]:
            if key in os.environ:
                minimal_env[key] = os.environ[key]

        result = subprocess.run(
            [sys.executable, script_path, tmp_data_path],
            capture_output=True,
            text=True,
            timeout=30,
            env=minimal_env,
        )
        if result.returncode != 0:
            logger.error("omnicloudmask subprocess failed: %s", result.stderr)
            raise RuntimeError(f"omnicloudmask subprocess failed: {result.stderr}")
        output = result.stdout.strip()
        if not output:
            raise RuntimeError("omnicloudmask subprocess returned empty output")

        if "|" in output:
            cloud_percentage_str, device_used = output.split("|", 1)
            cloud_percentage = float(cloud_percentage_str)
            logger.info(
                "Cloud cover from omnicloudmask: %.2f%% (using %s)", cloud_percentage, device_used
            )
        else:
            cloud_percentage = float(output)
            logger.info("Cloud cover from omnicloudmask: %.2f%%", cloud_percentage)

        return cloud_percentage
    except Exception as cpu_err:
        logger.error("omnicloudmask failed: %s", cpu_err, exc_info=True)
        raise RuntimeError(f"omnicloudmask cloud detection failed: {cpu_err}") from cpu_err
    finally:
        try:
            os.unlink(tmp_data_path)
        except Exception:
            pass


def get_search_bounds(center_lat, center_lon, size_degrees):
    half_size = size_degrees / 2
    return [
        center_lon - half_size,
        center_lat - half_size,
        center_lon + half_size,
        center_lat + half_size,
    ]


def search_sentinel2_cogs(bbox, start_date, end_date):
    STAC_ENDPOINT = "https://earth-search.aws.element84.com/v1"
    COLLECTION = "sentinel-2-l2a"
    catalog = Client.open(STAC_ENDPOINT)
    search = catalog.search(
        collections=[COLLECTION], bbox=bbox, datetime=f"{start_date}/{end_date}"
    )
    items = list(search.items())
    return items


def is_mostly_white_image(data, value_threshold=250, ratio_threshold=0.97):
    """Check if the image is mostly white.

    Returns a tuple of (is_white, white_ratio).
    """
    arr = np.asarray(data)

    if arr.size == 0:
        return False, 0.0

    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))

    arr = arr.astype(np.float32)

    max_value = float(arr.max()) if arr.size else 0.0
    if max_value <= 1.0:
        arr = arr * 255.0

    if arr.ndim == 3:
        white_mask = np.all(arr >= value_threshold, axis=-1)
    else:
        white_mask = arr >= value_threshold

    white_ratio = float(np.mean(white_mask))
    return white_ratio >= ratio_threshold, white_ratio


def save_as_png(data, output_path):
    data = np.transpose(data, (1, 2, 0))
    image = Image.fromarray(data)
    image.save(output_path, "PNG")


def read_cog_window(cog_url, lat, lon, window_size=100):
    with rasterio.open(cog_url) as src:
        src_crs = src.crs
        dst_crs = "EPSG:4326"
        x, y = transform_coords(dst_crs, src_crs, [lon], [lat])
        row, col = rasterio.transform.rowcol(src.transform, x[0], y[0])
        half_size = window_size // 2
        row_start = max(0, row - half_size)
        row_end = min(src.height, row + half_size)
        col_start = max(0, col - half_size)
        col_end = min(src.width, col + half_size)
        window = Window(col_start, row_start, col_end - col_start, row_end - row_start)
        data = src.read(window=window)
        transform = rasterio.windows.transform(window, src.transform)
        return data, transform, src_crs


def process_scene(scene, lat, lon, size_pixels, output_dir, snow_max, cloud_max):
    scene_date = scene.properties.get("datetime", "").split("T")[0]
    scene_id = scene.id

    scl_asset = scene.assets.get("scl")
    if scl_asset is None:
        return {"period": scene_id, "status": "error", "error": "No SCL asset found"}

    scl_url = scl_asset.href
    scl_data, _, _ = read_cog_window(scl_url, lat, lon, size_pixels)
    snw_pct = float(np.sum(scl_data == 11)) / float(scl_data.size) * 100.0

    if snw_pct > snow_max:
        return {
            "period": scene_id,
            "status": "skipped_snow",
            "reason": f"snow {snw_pct:.1f}% > {snow_max:.1f}%",
        }

    if "visual" not in scene.assets:
        return {"period": scene_id, "status": "error", "error": "No visual asset found"}

    visual_url = scene.assets["visual"].href
    assets = scene.assets

    def _href(keys):
        for key in keys:
            if key in assets:
                return assets[key].href
        return None

    r_url = _href(["B04", "red"])
    g_url = _href(["B03", "green"])
    nir_url = _href(["B08", "nir"])

    if not all([r_url, g_url, nir_url]):
        return {"period": scene_id, "status": "error", "error": "Missing required bands"}

    r, _, _ = read_cog_window(r_url, lat, lon, size_pixels)
    g, _, _ = read_cog_window(g_url, lat, lon, size_pixels)
    nir, _, _ = read_cog_window(nir_url, lat, lon, size_pixels)

    r = r[0] if r.ndim == 3 and r.shape[0] == 1 else r
    g = g[0] if g.ndim == 3 and g.shape[0] == 1 else g
    nir = nir[0] if nir.ndim == 3 and nir.shape[0] == 1 else nir
    r = np.clip(r.astype(np.float32) / 10000.0, 0.0, 1.0)
    g = np.clip(g.astype(np.float32) / 10000.0, 0.0, 1.0)
    nir = np.clip(nir.astype(np.float32) / 10000.0, 0.0, 1.0)

    # Stack as R-G-NIR for omnicloudmask: shape should be (H, W, 3)
    # Order: Red, Green, NIR
    rgn_array = np.stack([r, g, nir], axis=-1)

    cloud_cover = get_cloud_cover(rgn_array)
    if cloud_cover > cloud_max:
        return {
            "period": scene_id,
            "status": "skipped_cloud",
            "reason": f"cloud {cloud_cover:.1f}% > {cloud_max:.1f}% (OmniCloudMask)",
        }

    visual_data, _, _ = read_cog_window(visual_url, lat, lon, size_pixels)

    WHITE_RATIO_THRESHOLD = 0.97
    is_white, white_ratio = is_mostly_white_image(
        visual_data, ratio_threshold=WHITE_RATIO_THRESHOLD
    )
    if is_white:
        logger.info(
            "Skipping scene %s due to white frame (white_ratio=%.3f)",
            scene_id,
            white_ratio,
        )
        return {
            "period": scene_id,
            "status": "skipped_white",
            "reason": f"white {white_ratio * 100:.1f}% >= {WHITE_RATIO_THRESHOLD * 100:.1f}%",
        }

    tci_dir = output_dir / "TCI"
    tci_dir.mkdir(exist_ok=True)
    tci_output_path = tci_dir / f"{scene_date}_{scene_id}.png"
    save_as_png(visual_data, tci_output_path)

    return {"period": scene_id, "output_path": str(tci_output_path), "status": "processed"}


def standardize_image(img_tensor):
    mean = img_tensor.mean(dim=(0, 1), keepdim=True)
    std = img_tensor.std(dim=(0, 1), keepdim=True)
    std = torch.clamp(std, min=1e-8)
    return (img_tensor - mean) / std, mean, std


def load_lr_images(folder_path):
    import glob

    extensions = ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG"]
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(str(Path(folder_path) / ext)))

    if not image_paths:
        return None, None, None

    # First pass: find the minimum dimensions across all images
    min_h, min_w = float("inf"), float("inf")
    for img_path in sorted(image_paths):
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        min_h = min(min_h, h)
        min_w = min(min_w, w)

    lr_images = []
    means = []
    stds = []

    # Second pass: load images and crop to minimum size
    for img_path in sorted(image_paths):
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Crop to minimum size (center crop)
        h, w = img.shape[:2]
        start_h = (h - min_h) // 2
        start_w = (w - min_w) // 2
        img = img[start_h : start_h + min_h, start_w : start_w + min_w]

        lr_tensor = torch.from_numpy(img).float() / 255.0
        lr_std, mean, std = standardize_image(lr_tensor)
        lr_images.append(lr_std)
        means.append(mean)
        stds.append(std)

    return lr_images, means, stds


def create_hr_coords(lr_shape, scale_factor):
    lr_h, lr_w = lr_shape[:2]
    hr_h, hr_w = lr_h * scale_factor, lr_w * scale_factor
    return make_normalized_grid(hr_h, hr_w, vmin=-1.0, vmax=1.0, pixel_center=True)


def train_step(model, optimizer, coords, lr_target, sample_id, scale_factor):
    optimizer.zero_grad()

    # Pass lr_frames when using GNLL for uncertainty estimation
    if model.use_gnll:
        result = model(
            coords,
            sample_idx=sample_id,
            scale_factor=scale_factor,
            training=True,
            lr_frames=lr_target,
        )
        # When use_gnll=True and lr_frames is provided, returns (output, shifts, variances)
        if isinstance(result, tuple) and len(result) == 3:
            output, _, pred_variance = result
            recon_criterion = nn.GaussianNLLLoss()
            loss = recon_criterion(output, lr_target, pred_variance)
        else:
            # Fallback to MSE if variance not available
            output = result[0] if isinstance(result, tuple) else result
            loss = F.mse_loss(output, lr_target)
    else:
        result = model(coords, sample_idx=sample_id, scale_factor=scale_factor, training=True)
        # Handle different return signatures from INR forward
        if isinstance(result, tuple):
            output = result[0]  # First element is always the output
        else:
            output = result
        loss = F.mse_loss(output, lr_target)

    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate(model, coords, sample_id, mean, std, device):
    with torch.no_grad():
        coords = coords.unsqueeze(0).to(device)
        sample_id = torch.tensor([sample_id]).to(device)
        scale_factor = torch.tensor([1.0], device=device)

        # Call model - handle different return values based on use_gnll
        result = model(
            coords, sample_idx=sample_id, scale_factor=scale_factor, training=False, lr_frames=None
        )

        # Handle different return signatures
        variance = None
        if model.use_gnll:
            # When use_gnll=True but lr_frames=None, model returns (output, shifts) - 2 values
            # When use_gnll=True and lr_frames is not None, model returns (output, shifts, variances) - 3 values
            if isinstance(result, tuple) and len(result) == 3:
                output, _, variance = result
                variance = variance.squeeze().cpu().numpy() if variance is not None else None
            else:
                output, _ = result
        else:
            # When use_gnll=False, model returns (output, shifts) - 2 values
            if isinstance(result, tuple):
                output, _ = result
            else:
                output = result

        # Unstandardize
        output = output * std.to(device) + mean.to(device)
        output = torch.clamp(output, 0, 1)

        output_np = output.squeeze().cpu().numpy()
        return output_np, variance


def train_inr(
    lr_images,
    means,
    stds,
    hr_coords,
    scale_factor,
    iterations,
    fourier_scale,
    use_gnll=False,
    device=None,
    progress_callback=None,
):
    """Train INR model on specified device."""
    start_time = time.time()

    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            device_info = f"CUDA ({gpu_name}, {gpu_memory:.2f} GB)"
            logger.info("Using GPU: %s (%.2f GB)", gpu_name, gpu_memory)
        else:
            device = torch.device("cpu")
            device_info = "CPU"
            logger.info("Using CPU")
    elif device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        device_info = f"CUDA ({gpu_name}, {gpu_memory:.2f} GB)"
        logger.info("Using GPU: %s (%.2f GB)", gpu_name, gpu_memory)
    else:
        device_info = "CPU (GPU unavailable)"
        logger.info("Training on CPU")

    num_samples = len(lr_images)
    if num_samples == 0:
        raise ValueError("No low-resolution images available for training")

    hr_coords_batch = hr_coords.unsqueeze(0).to(device)

    input_projection = FourierFeatures(2, mapping_size=256, scale=fourier_scale)
    decoder = MLP(input_dim=256, hidden_dim=256, depth=4, output_dim=3)
    model = INR(input_projection, decoder, num_samples, use_gnll=use_gnll).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations, eta_min=1e-5)

    scale_factor_tensor = torch.tensor([1.0 / scale_factor]).to(device)

    logger.info("Starting INR training: %d iterations across %d samples", iterations, num_samples)
    iteration = 0
    last_loss = None
    while iteration < iterations:
        for idx in range(num_samples):
            if iteration >= iterations:
                break
            lr_target = lr_images[idx].unsqueeze(0).to(device)
            sample_id = torch.tensor([idx]).to(device)
            loss = train_step(
                model, optimizer, hr_coords_batch, lr_target, sample_id, scale_factor_tensor
            )
            scheduler.step()
            last_loss = loss
            iteration += 1

            if iteration % 50 == 0:
                logger.info("Iteration %d/%d - loss %.4f", iteration, iterations, loss)
                if progress_callback:
                    # Progress from 0.5 to 0.9 during training
                    train_progress = 0.5 + 0.4 * (iteration / iterations)
                    progress_callback(
                        train_progress,
                        desc=f"Training: {iteration}/{iterations} (loss: {loss:.4f})",
                    )

    logger.info(
        "Training complete. Final loss: %.4f", last_loss if last_loss is not None else float("nan")
    )

    sr_output, variance = evaluate(model, hr_coords, 0, means[0], stds[0], device)

    lr_unstandardized = lr_images[0] * stds[0] + means[0]
    lr_unstandardized = np.clip(lr_unstandardized.numpy(), 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(lr_unstandardized)
    axes[0].set_title("Original Sentinel-2 (10m/pixel)")
    axes[0].axis("off")
    axes[1].imshow(sr_output)
    axes[1].set_title(f"Super-Resolution ({scale_factor}x)")
    axes[1].axis("off")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    comparison_img = Image.open(buf)
    plt.close()

    sr_img = Image.fromarray((np.clip(sr_output, 0, 1) * 255).astype(np.uint8))

    lr_frames_for_gallery = []
    for i, lr_img in enumerate(lr_images):
        lr_unstd = lr_img * stds[i] + means[i]
        lr_unstd = np.clip(lr_unstd.numpy(), 0, 1)
        lr_pil = Image.fromarray((lr_unstd * 255).astype(np.uint8))
        lr_frames_for_gallery.append(lr_pil)

    variance_images = []
    if use_gnll and model.use_gnll:
        logger.info("Generating uncertainty map for final output...")
        sr_output_var, variance = evaluate(model, hr_coords, 0, means[0], stds[0], device)

        if variance is not None:
            lr_sample = lr_images[0] * stds[0] + means[0]
            lr_sample_np = (
                lr_sample.cpu().numpy() if torch.is_tensor(lr_sample) else np.array(lr_sample)
            )
            lr_sample_np = np.clip(lr_sample_np, 0, 1)

            hr_h, hr_w = sr_output_var.shape[:2]
            lr_resized = cv2.resize(lr_sample_np, (hr_w, hr_h), interpolation=cv2.INTER_LINEAR)
            lr_resized = np.clip(lr_resized, 0, 1).astype(np.float32)

            variance = np.maximum(variance, 0)
            std_map = np.sqrt(variance).mean(axis=-1) if variance.ndim == 3 else np.sqrt(variance)
            vmin, vmax = np.percentile(std_map, [5, 95])

            fig_var, axes_var = plt.subplots(2, 2, figsize=(12, 12))

            axes_var[0, 0].imshow(lr_resized)
            axes_var[0, 0].set_title("LR Sample (upsampled)", fontsize=12, fontweight="bold")
            axes_var[0, 0].axis("off")

            axes_var[0, 1].imshow(sr_output_var)
            axes_var[0, 1].set_title("SR Output", fontsize=12, fontweight="bold")
            axes_var[0, 1].axis("off")

            im_var = axes_var[1, 0].imshow(std_map, cmap="viridis", vmin=vmin, vmax=vmax)
            axes_var[1, 0].set_title("Standard Deviation Map", fontsize=12, fontweight="bold")
            axes_var[1, 0].axis("off")
            plt.colorbar(
                im_var, ax=axes_var[1, 0], fraction=0.046, pad=0.04, label="Standard Deviation"
            )

            std_normalized = (std_map - std_map.min()) / (std_map.max() - std_map.min() + 1e-8)
            overlay = np.zeros_like(sr_output_var)
            overlay[:, :, 0] = std_normalized
            alpha = 0.5 * std_normalized[..., np.newaxis]
            blended = sr_output_var * (1 - alpha) + overlay * alpha

            axes_var[1, 1].imshow(blended)
            axes_var[1, 1].set_title(
                "SR with Std Overlay (Red=High Std)", fontsize=12, fontweight="bold"
            )
            axes_var[1, 1].axis("off")

            plt.tight_layout()

            buf_var = io.BytesIO()
            plt.savefig(buf_var, format="png", dpi=150, bbox_inches="tight")
            buf_var.seek(0)
            variance_img = Image.open(buf_var)
            plt.close(fig_var)

            variance_images.append(variance_img)
            logger.info(
                f"Generated uncertainty map: std range [{std_map.min():.6f}, {std_map.max():.6f}], mean {std_map.mean():.6f}"
            )

    training_time = time.time() - start_time
    return (
        sr_img,
        comparison_img,
        lr_frames_for_gallery,
        device_info,
        training_time,
        last_loss,
        variance_images,
    )


def train_inr_on_gpu_wrapper(
    lr_images,
    means,
    stds,
    hr_coords,
    scale_factor,
    iterations,
    fourier_scale,
    use_gnll=False,
    progress_callback=None,
):
    """Try GPU training, fall back to CPU if GPU is unavailable."""
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA device available")
        return train_inr(
            lr_images,
            means,
            stds,
            hr_coords,
            scale_factor,
            iterations,
            fourier_scale,
            use_gnll,
            device=torch.device("cuda:0"),
            progress_callback=progress_callback,
        )
    except Exception as e:
        logger.warning("GPU training failed (%s), falling back to CPU", e)
        return train_inr(
            lr_images,
            means,
            stds,
            hr_coords,
            scale_factor,
            iterations,
            fourier_scale,
            use_gnll,
            device=torch.device("cpu"),
            progress_callback=progress_callback,
        )


def search_address(address):
    """Search for an address using Nominatim (OpenStreetMap) geocoding"""
    import time

    import requests

    if not address or address.strip() == "":
        return None, None, "Please enter an address to search."

    try:
        # Use Nominatim API (free, no API key required)
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": address, "format": "json", "limit": 1}
        headers = {"User-Agent": "SuperF-Gradio-App/1.0"}  # Nominatim requires a user agent

        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()

        results = response.json()

        if len(results) > 0:
            result = results[0]
            lat = float(result["lat"])
            lon = float(result["lon"])
            display_name = result.get("display_name", address)

            # Add a small delay to be respectful to Nominatim's usage policy
            time.sleep(1)

            return lat, lon, f"Found: {display_name}"
        else:
            return None, None, f"No results found for '{address}'. Try a different search term."

    except Exception as e:
        return None, None, f"Search error: {str(e)}"


def create_map(lat, lon):
    """Create an interactive Folium map with click-to-auto-fill coordinates"""
    import random
    import string

    # Generate unique ID to avoid conflicts when map updates
    map_id = "".join(random.choices(string.ascii_lowercase, k=8))

    # Create base map centered on current location
    m = folium.Map(location=[lat, lon], zoom_start=10, tiles="OpenStreetMap")

    # Add red marker at selected location
    folium.Marker(
        [lat, lon],
        popup=f"<b>Selected Location</b><br>Lat: {lat:.6f}<br>Lon: {lon:.6f}",
        tooltip=f"Selected: {lat:.6f}, {lon:.6f}",
        icon=folium.Icon(color="red", icon="info-sign"),
    ).add_to(m)

    # Add custom JavaScript to handle map clicks
    click_handler_js = f"""
    <script>
    (function() {{
        var currentMarker = null;
        var mapReady = false;

        function initMap() {{
            var mapDivs = document.querySelectorAll('.folium-map');

            mapDivs.forEach(function(mapDiv) {{
                var mapId = mapDiv.id;

                if (window[mapId] && !mapReady) {{
                    mapReady = true;

                    // Store reference to existing marker
                    window[mapId].eachLayer(function(layer) {{
                        if (layer instanceof L.Marker) {{
                            currentMarker = layer;
                        }}
                    }});

                    // Add click event to the map
                    window[mapId].on('click', function(e) {{
                        var lat = e.latlng.lat.toFixed(6);
                        var lon = e.latlng.lng.toFixed(6);

                        // Move the marker to the new position
                        if (currentMarker) {{
                            currentMarker.setLatLng(e.latlng);
                            currentMarker.setPopupContent('<b>Selected Location</b><br>Lat: ' + lat + '<br>Lon: ' + lon);
                        }}

                        // Send coordinates to parent window
                        window.parent.postMessage({{
                            type: 'map-click',
                            lat: lat,
                            lon: lon
                        }}, '*');

                        window.top.postMessage({{
                            type: 'map-click',
                            lat: lat,
                            lon: lon
                        }}, '*');
                    }});
                }}
            }});
        }}

        // Try multiple times to catch the map when it's ready
        setTimeout(initMap, 100);
        setTimeout(initMap, 500);
        setTimeout(initMap, 1000);

        if (document.readyState === 'loading') {{
            document.addEventListener('DOMContentLoaded', function() {{
                setTimeout(initMap, 100);
            }});
        }}
    }})();
    </script>
    """

    m.get_root().html.add_child(folium.Element(click_handler_js))

    # Return the HTML representation
    return m._repr_html_()


def process_super_resolution(
    lat,
    lon,
    start_date,
    end_date,
    size_pixels=100,
    snow_max=0,
    cloud_max=20,
    scale_factor=5,
    iterations=2000,
    max_images=8,
    fourier_scale=10,
    use_gnll=False,
    progress=gr.Progress(),
):
    """Main processing function for Gradio interface"""

    output_dir = None
    try:
        # Initialize device_info (will be set later)
        device_info = "Unknown"

        # Normalize numeric inputs
        size_pixels = int(size_pixels)
        max_images = int(max_images)
        iterations = int(iterations)
        scale_factor = int(scale_factor)
        snow_max = float(snow_max)
        cloud_max = float(cloud_max)
        fourier_scale = float(fourier_scale)

        # Setup
        progress(0, desc="Setting up...")
        size_degrees = size_pixels * 0.0001
        output_dir = create_output_directory("results")
        search_bounds = get_search_bounds(lat, lon, size_degrees)

        cache_parameters = {
            "lat": round(float(lat), 6),
            "lon": round(float(lon), 6),
            "start_date": str(start_date),
            "end_date": str(end_date),
            "size_pixels": size_pixels,
            "snow_max": snow_max,
            "cloud_max": cloud_max,
            "max_images": max_images,
        }
        cache_key = compute_collection_cache_key(cache_parameters)
        cache_dir, cache_tci_dir, cache_metadata_path = get_cache_paths(cache_key)
        cache_metadata = {}
        cache_used = False
        cache_info = "Miss"

        # Search for images
        progress(0.1, desc="Searching for satellite images...")
        available_items = search_sentinel2_cogs(search_bounds, start_date, end_date)

        if len(available_items) == 0:
            return (
                None,
                None,
                [],
                [],
                "No satellite images found for the specified location and date range.",
            )

        results = []
        processed = []

        if cache_tci_dir.exists() and any(cache_tci_dir.iterdir()):
            progress(0.2, desc="Loading cached imagery...")
            cache_metadata = load_cache_metadata(cache_metadata_path)
            cached_results = cache_metadata.get("results", [])
            if isinstance(cached_results, list):
                results = cached_results
            processed = [r for r in results if r.get("status") == "processed"]

            if processed:
                try:
                    shutil.copytree(cache_tci_dir, output_dir / "TCI")
                    cache_used = True
                    cache_info = "Hit"
                    logger.info(
                        "Cache hit for key %s (processed images: %d)",
                        cache_key,
                        len(processed),
                    )
                except Exception as exc:
                    logger.warning("Failed to copy cached imagery for key %s: %s", cache_key, exc)
                    results = []
                    processed = []
            else:
                logger.info(
                    "Cache entry for key %s contains no processed images; refreshing.", cache_key
                )
                results = []
                processed = []

        if not cache_used:
            progress(0.2, desc=f"Processing scenes (max {max_images} cloud-free images)...")
            processed_count = 0
            results = []

            for i, item in enumerate(available_items):
                if processed_count >= max_images:
                    break

                result = process_scene(item, lat, lon, size_pixels, output_dir, snow_max, cloud_max)
                results.append(result)

                if result["status"] == "processed":
                    processed_count += 1

                progress(
                    0.2 + 0.2 * (i + 1) / len(available_items),
                    desc=f"Processing scene {i+1}/{len(available_items)} ({processed_count}/{max_images} cloud-free)",
                )

            processed = [r for r in results if r["status"] == "processed"]
            if len(processed) == 0:
                return (
                    None,
                    None,
                    [],
                    [],
                    f"No clear images found. {len(results)} images were filtered due to clouds/snow.",
                )

            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                if cache_tci_dir.exists():
                    shutil.rmtree(cache_tci_dir)
                shutil.copytree(output_dir / "TCI", cache_tci_dir)
                cache_metadata = {
                    "parameters": cache_parameters,
                    "created_at": datetime.utcnow().isoformat() + "Z",
                    "results": results,
                    "available_items": [item.id for item in available_items],
                }
                save_cache_metadata(cache_metadata_path, cache_metadata)
                cache_info = "Stored"
                logger.info(
                    "Stored %d processed images in cache key %s.", len(processed), cache_key
                )
            except Exception as exc:
                logger.warning("Failed to store cache for key %s: %s", cache_key, exc)

        if cache_used:
            tci_dir = output_dir / "TCI"
            if not results:
                image_paths = sorted(tci_dir.glob("*.png"))
                results = [
                    {
                        "period": path.stem,
                        "output_path": str(path),
                        "status": "processed",
                    }
                    for path in image_paths
                ]
            processed = [r for r in results if r.get("status") == "processed"]
            progress(0.3, desc="Cached images ready")

        if len(processed) == 0:
            return (
                None,
                None,
                [],
                [],
                f"No clear images found. {len(results)} images were filtered due to clouds/snow.",
            )

        # Load LR images
        progress(0.4, desc="Loading images...")
        lr_images, means, stds = load_lr_images(output_dir / "TCI")
        if not lr_images:
            return None, None, [], [], "Unable to load low-resolution images after filtering."

        hr_coords = create_hr_coords(lr_images[0].shape, scale_factor)

        progress(0.5, desc="Starting training...")
        (
            sr_img,
            comparison_img,
            lr_frames_for_gallery,
            device_info,
            training_time,
            final_loss,
            variance_images,
        ) = train_inr_on_gpu_wrapper(
            lr_images,
            means,
            stds,
            hr_coords,
            scale_factor,
            iterations,
            fourier_scale,
            use_gnll,
            progress_callback=progress,
        )

        progress(0.92, desc="Preparing results...")
        progress(1.0, desc="Complete!")

        training_time_display = f"{training_time:.1f}"
        final_loss_display = f"{final_loss:.4f}" if final_loss is not None else "n/a"

        # Check if we stopped early
        stopped_early = len(processed) >= max_images and len(results) < len(available_items)
        early_stop_msg = (
            f"\n        - Stopped early: Yes (reached max of {max_images} images)"
            if stopped_early
            else ""
        )

        gnll_status = "Enabled (Uncertainty Estimation)" if use_gnll else "Disabled (MSE Loss)"
        summary = f"""
        Processing Summary:
        - Location: ({lat}, {lon})
        - Date Range: {start_date} to {end_date}
        - Total images available: {len(available_items)}
        - Images examined: {len(results)}
        - Clear images used: {len(processed)}/{max_images}{early_stop_msg}
        - Scale factor: {scale_factor}x
        - Training iterations: {iterations}
        - GNLL: {gnll_status}
        - Cache: {cache_info} (key: {cache_key[:10]})
        - Training time: {training_time_display}s
        - Final loss: {final_loss_display}
        - Device: {device_info}
        """

        # Return variance_images if available, otherwise empty list
        if not variance_images:
            variance_images = []

        return sr_img, comparison_img, lr_frames_for_gallery, variance_images, summary

    except Exception as e:
        return None, None, [], [], f"Error: {str(e)}"
    finally:
        if output_dir and output_dir.exists():
            shutil.rmtree(output_dir, ignore_errors=True)


# Gradio Interface
with gr.Blocks(
    title="SuperF - Satellite Image Super-Resolution",
    head="""
    <script>
    // Listen for map clicks and update Latitude/Longitude fields
    window.addEventListener('message', function(event) {
        if (event.data && event.data.type === 'map-click') {
            setTimeout(function() {
                var latValue = event.data.lat;
                var lonValue = event.data.lon;

                // Find and update the Latitude and Longitude fields
                var labels = document.querySelectorAll('label');
                var latInput = null;
                var lonInput = null;

                for (var i = 0; i < labels.length; i++) {
                    var labelText = labels[i].textContent.trim();
                    if (labelText === 'Latitude') {
                        latInput = labels[i].parentElement.querySelector('input[type="number"]');
                    } else if (labelText === 'Longitude') {
                        lonInput = labels[i].parentElement.querySelector('input[type="number"]');
                    }
                }

                // Update latitude
                if (latInput) {
                    var latSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
                    latSetter.call(latInput, latValue);
                    latInput.dispatchEvent(new Event('input', { bubbles: true }));
                }

                // Update longitude
                if (lonInput) {
                    var lonSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
                    lonSetter.call(lonInput, lonValue);
                    lonInput.dispatchEvent(new Event('input', { bubbles: true }));
                }
            }, 100);
        }
    });
    </script>
    """,
) as demo:
    gr.Markdown(
        """
    # SuperF: Neural Implicit Fields for Multi-Image Super-Resolution

    ## Super-Resolving Any Place on Earth

    **TLDR:** This app super-resolves Sentinel-2 optical satellite images (original resolution 10m) using the SuperF approach, e.g. by factor 5.

    More details about the project: [https://sjyhne.github.io/superf/](https://sjyhne.github.io/superf/)

    **What happens under the hood?** After the user enters a geographic coordinate (latitude, longitude), this application downloads multiple cloud free Sentinel-2 images of the same location. These 10 meter images that have subtle spatial misalignments, which is useful to compute a super-resolved image. Using these images, the SuperF approach optimizes an implicit neural representation (INR) of the shared underlying high-resolution image. SuperF achieves this by i) jointly optimizing the INR with the affine alignment between the individual frames, and by ii) sharing a coordinate-based neural network to represent the high-resolution signal underlying all low-resolution Sentinel-2 images.

    **Note:** Processing may take 5-15 minutes depending on the settings, e.g. number of images, time window, iterations, image size, scale factor.
    """
    )

    gr.Markdown("### Location Selection")

    address_search = gr.Textbox(
        label="Search Address or Landmark (press Enter to search)",
        placeholder="e.g., Eiffel Tower, Paris or Central Park, New York",
    )

    search_status = gr.Textbox(label="Search Result", visible=False, lines=1)

    with gr.Row():
        lat = gr.Number(label="Latitude", value=58.148775, precision=6)
        lon = gr.Number(label="Longitude", value=7.989809, precision=6)
        start_date = gr.Textbox(label="Start Date (YYYY-MM-DD)", value="2024-05-01")
        end_date = gr.Textbox(label="End Date (YYYY-MM-DD)", value="2024-08-01")

    gr.Markdown("💡 **Tip:** Press Enter after editing the coordinates to update the map.")

    with gr.Row():
        gr.Column(scale=1, min_width=0)
        with gr.Column(scale=3, min_width=500):
            location_map = gr.HTML(label="Interactive Map - Click to select location")
        gr.Column(scale=1, min_width=0)

    gr.Markdown("### Processing Settings")

    with gr.Row():
        size_pixels = gr.Number(
            label="Image Size (pixels)", value=100, minimum=50, maximum=200, info="Range: 50-200"
        )
        scale_factor = gr.Number(
            label="Scale Factor", value=4, minimum=2, maximum=10, info="Range: 2-10x"
        )
        max_images = gr.Number(
            label="Max Images", value=8, minimum=1, maximum=20, info="Range: 1-20"
        )
        iterations = gr.Number(
            label="Training Iterations",
            value=2000,
            minimum=500,
            maximum=5000,
            info="Range: 500-5000",
        )

    with gr.Accordion("⚙️ Advanced Options", open=False):
        with gr.Row():
            cloud_max = gr.Number(
                label="Max Cloud Coverage (%)", value=0, minimum=0, maximum=50, info="Range: 0-50%"
            )
            snow_max = gr.Number(
                label="Max Snow Coverage (%)", value=0, minimum=0, maximum=50, info="Range: 0-50%"
            )
            fourier_scale = gr.Number(
                label="Fourier Scale", value=5, minimum=1, maximum=20, info="Fourier feature scale"
            )
        use_gnll = gr.Checkbox(
            label="Use GNLL (Gaussian Negative Log Likelihood) for Uncertainty Estimation",
            value=False,
            info="Enable uncertainty estimation during training. Uses GaussianNLLLoss instead of MSE loss.",
        )

    process_btn = gr.Button("🚀 Process Super-Resolution", variant="primary", size="lg")

    gr.Markdown("---")
    gr.Markdown("### Results")

    with gr.Row():
        with gr.Column():
            sr_output = gr.Image(label="Super-Resolution Output", type="pil")
        with gr.Column():
            comparison = gr.Image(label="Before/After Comparison", type="pil")

    with gr.Accordion("View Input LR Frames", open=False):
        lr_frames_gallery = gr.Gallery(
            label="Low-Resolution Input Frames",
            show_label=False,
            columns=4,
            rows=2,
            height="auto",
            object_fit="contain",
        )

    with gr.Accordion("View Uncertainty Maps (GNLL)", open=False):
        variance_gallery = gr.Gallery(
            label="Uncertainty Visualizations",
            show_label=False,
            columns=2,
            rows=2,
            height="auto",
            object_fit="contain",
            visible=True,
        )

    status = gr.Textbox(label="Status", lines=8)

    # Process button
    process_btn.click(
        fn=process_super_resolution,
        inputs=[
            lat,
            lon,
            start_date,
            end_date,
            size_pixels,
            snow_max,
            cloud_max,
            scale_factor,
            iterations,
            max_images,
            fourier_scale,
            use_gnll,
        ],
        outputs=[sr_output, comparison, lr_frames_gallery, variance_gallery, status],
    )

    # Address search handlers
    def handle_search(address):
        """Handle address search and update coordinates"""
        search_lat, search_lon, message = search_address(address)

        if search_lat is not None and search_lon is not None:
            # Update coordinates and map
            return (
                search_lat,
                search_lon,
                message,
                gr.update(visible=True),
                create_map(search_lat, search_lon),
            )
        else:
            # Keep existing coordinates, just show error message
            return gr.update(), gr.update(), message, gr.update(visible=True), gr.update()

    # Address search on Enter key
    address_search.submit(
        fn=handle_search,
        inputs=[address_search],
        outputs=[lat, lon, search_status, search_status, location_map],
    )

    # Initialize map on load
    demo.load(fn=create_map, inputs=[lat, lon], outputs=location_map)

    # Update map when Enter is pressed in lat or lon fields
    lat.submit(fn=create_map, inputs=[lat, lon], outputs=location_map)

    lon.submit(fn=create_map, inputs=[lat, lon], outputs=location_map)

demo.queue(default_concurrency_limit=1, max_size=2, status_update_rate=1)

if __name__ == "__main__":
    try:
        logger.info("Launching Gradio application...")
        demo.launch()
    except Exception as e:
        logger.error("Failed to launch application: %s", e, exc_info=True)
        raise
