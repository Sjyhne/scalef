import torch


def _centered_axis(size: int, vmin: float, vmax: float, device=None) -> torch.Tensor:
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    span = float(vmax) - float(vmin)
    return (torch.arange(size, dtype=torch.float32, device=device) + 0.5) * (span / size) + float(vmin)


def _corner_axis(size: int, vmin: float, vmax: float, device=None) -> torch.Tensor:
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    span = float(vmax) - float(vmin)
    return torch.arange(size, dtype=torch.float32, device=device) * (span / size) + float(vmin)


def make_normalized_grid(
    height: int,
    width: int | None = None,
    *,
    vmin: float = 0.0,
    vmax: float = 1.0,
    pixel_center: bool = True,
    device=None,
) -> torch.Tensor:
    """Return an HxWx2 grid with x in channel 0 and y in channel 1."""
    if width is None:
        width = height

    axis_fn = _centered_axis if pixel_center else _corner_axis
    ys = axis_fn(height, vmin, vmax, device=device)
    xs = axis_fn(width, vmin, vmax, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((xx, yy), dim=-1)
