import torch
import torch.nn as nn


class BasicLosses(nn.Module):
    """L1, L2, RMSE with optional masking."""

    @staticmethod
    def mae_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if mask is None:
            return torch.abs(pred - target).mean()
        return (torch.abs(pred - target) * mask).mean()

    @staticmethod
    def mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if mask is None:
            return torch.square(pred - target).mean()
        return (torch.square(pred - target) * mask).mean()

    @staticmethod
    def huber_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        delta: float = 0.03,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r = pred - target
        abs_r = torch.abs(r)
        d = float(delta)
        quad = 0.5 * r**2
        lin = d * (abs_r - 0.5 * d)
        loss = torch.where(abs_r <= d, quad, lin)
        if mask is None:
            return loss.mean()
        return (loss * mask).mean()

    @staticmethod
    def charbonnier_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        epsilon: float = 1e-3,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        e2 = float(epsilon) ** 2
        r2 = torch.square(pred - target) + e2
        loss = torch.sqrt(r2)
        if mask is None:
            return loss.mean()
        return (loss * mask).mean()

    @staticmethod
    def rmse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.square(pred - target).mean())
