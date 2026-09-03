import torch
import torch.nn as nn

_RECON_LOSS_CHOICES = ("mse", "mae", "charbonnier", "huber")


class BasicLosses(nn.Module):
    """L1, L2, RMSE, Charbonnier, and Huber with optional masking."""

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
    def rmse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.square(pred - target).mean())

    @staticmethod
    def charbonnier_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        eps: float = 1e-3,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        diff = pred - target
        loss = torch.sqrt(diff * diff + float(eps) ** 2)
        if mask is None:
            return loss.mean()
        return (loss * mask).mean()

    @staticmethod
    def huber_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        delta: float = 0.05,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        r = pred - target
        a = r.abs()
        d = float(delta)
        quadratic = torch.clamp(a, max=d)
        linear = a - quadratic
        loss = 0.5 * quadratic * quadratic + d * linear
        if mask is None:
            return loss.mean()
        return (loss * mask).mean()


def resolve_recon_criterion(
    recon_loss: str,
    *,
    charbonnier_eps: float = 1e-3,
    huber_delta: float = 0.05,
):
    """Return a (pred, target[, mask]) -> scalar loss callable for training."""
    name = str(recon_loss).lower().strip()
    if name == "mse":
        return BasicLosses.mse_loss
    if name == "mae":
        return BasicLosses.mae_loss
    if name == "charbonnier":
        eps = float(charbonnier_eps)

        def _charbonnier(pred, target, mask=None):
            return BasicLosses.charbonnier_loss(pred, target, eps=eps, mask=mask)

        return _charbonnier
    if name == "huber":
        delta = float(huber_delta)

        def _huber(pred, target, mask=None):
            return BasicLosses.huber_loss(pred, target, delta=delta, mask=mask)

        return _huber
    raise ValueError(
        f"Unknown recon_loss {recon_loss!r}. Use one of: {', '.join(_RECON_LOSS_CHOICES)}."
    )


def laplace_nll_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    scale: torch.Tensor,
    *,
    eps: float = 1e-6,
    full: bool = False,
) -> torch.Tensor:
    """Heteroscedastic Laplace NLL: |pred - target| / b + log(b).

    ``scale`` is the positive Laplace scale b (decoder log-scale is exp()'d upstream).
    """
    b = scale.clamp(min=float(eps))
    loss = (pred - target).abs() / b + torch.log(b)
    if full:
        return loss
    return loss.mean()
