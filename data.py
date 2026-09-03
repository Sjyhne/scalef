import torch
import numpy as np


def _pixel_center_axis(size: int, vmin: float = 0.0, vmax: float = 1.0) -> np.ndarray:
    """Pixel-center coordinates: (i + 0.5) / N scaled to [vmin, vmax)."""
    span = float(vmax) - float(vmin)
    return (np.arange(int(size), dtype=np.float64) + 0.5) * (span / int(size)) + float(vmin)


def _make_coord_grid(
    height: int,
    width: int,
    vmin: float = 0.0,
    vmax: float = 1.0,
    device=None,
) -> torch.Tensor:
    """Pixel-center coordinate grid of shape [H, W, 2] with x,y independently in [vmin, vmax)."""
    xs = _pixel_center_axis(width, vmin, vmax)
    ys = _pixel_center_axis(height, vmin, vmax)
    coords = np.stack(np.meshgrid(xs, ys), -1)
    return torch.as_tensor(coords, dtype=torch.float32, device=device)


def _make_unit_square_coord_grid(side: int, vmin: float = 0.0, vmax: float = 1.0, device=None) -> torch.Tensor:
    """Cached-friendly HR/LR coordinate grid with pixel-center samples per axis."""
    return _make_coord_grid(side, side, vmin, vmax, device)


def _build_input_coord_cache(
    lr_side: int,
    scale_factors: list[float],
    *,
    vmin: float = 0.0,
    vmax: float = 1.0,
    device=None,
    lr_height: int | None = None,
    lr_width: int | None = None,
) -> dict[float, torch.Tensor]:
    height = int(lr_side if lr_height is None else lr_height)
    width = int(lr_side if lr_width is None else lr_width)
    return {
        float(sf): _make_coord_grid(int(round(height * sf)), int(round(width * sf)), vmin, vmax, device)
        for sf in scale_factors
    }


def get_and_standardize_image(image):
    """Per-channel zero mean, unit std. Handles 2D, 3D (HWC/CHW), 4D. Returns (standardized, mean, std)."""
    if image.dim() == 2:
        img = image.unsqueeze(-1)
        mean = img.mean(dim=(0, 1), keepdim=True)
        std = img.std(dim=(0, 1), keepdim=True)
        std = torch.clamp(std, min=1e-8)
        standardized = (img - mean) / std
        return standardized.squeeze(-1), mean.squeeze(0), std.squeeze(0)

    if image.dim() == 3:
        if image.shape[0] in (1, 3, 4) and image.shape[0] != image.shape[-1]:
            mean = image.mean(dim=(1, 2), keepdim=True)
            std = image.std(dim=(1, 2), keepdim=True)
        else:
            mean = image.mean(dim=(0, 1), keepdim=True)
            std = image.std(dim=(0, 1), keepdim=True)

        std = torch.clamp(std, min=1e-8)
        return (image - mean) / std, mean, std

    if image.dim() == 4:
        if image.shape[-1] in (1, 3, 4):
            mean = image.mean(dim=(1, 2), keepdim=True)
            std = image.std(dim=(1, 2), keepdim=True)
        else:
            mean = image.mean(dim=(2, 3), keepdim=True)
            std = image.std(dim=(2, 3), keepdim=True)

        std = torch.clamp(std, min=1e-8)
        return (image - mean) / std, mean, std

    mean = image.mean()
    std = torch.clamp(image.std(), min=1e-8)
    return (image - mean) / std, mean, std


def resolve_dataset_device(args, training_device: torch.device | None = None) -> torch.device:
    """Resolve where cached LR/coords live (default: same GPU as training)."""
    spec = getattr(args, "dataset_device", "auto")
    if isinstance(spec, torch.device):
        return spec
    text = str(spec).strip().lower()
    if text in {"auto", ""}:
        if training_device is not None:
            return training_device
        train_spec = getattr(args, "device", "cpu")
        if str(train_spec).lower() == "cpu" or not torch.cuda.is_available():
            return torch.device("cpu")
        try:
            idx = int(train_spec)
        except ValueError:
            idx = 0
        if idx < 0 or idx >= torch.cuda.device_count():
            idx = 0
        return torch.device(f"cuda:{idx}")
    if text == "cpu":
        return torch.device("cpu")
    return torch.device(spec)


def get_dataset(args, name="s2", keep_in_memory=True, training_device=None):
    del keep_in_memory
    from s2_dataset import S2NIBRevisitDataset, is_s2_dataset_request

    if is_s2_dataset_request(args, name):
        return S2NIBRevisitDataset(args, name=name, training_device=training_device)
    raise ValueError(
        f"No loader for dataset {name!r}. Use --dataset s2 with --s2-dir "
        "(e.g. data/s2_revisits/bergen) after scripts/fetch_s2_revisits.py."
    )
