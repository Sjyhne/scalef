import torch
from torch.optim import Optimizer


def zeropower_via_newtonschulz5(g: torch.Tensor, steps: int):
    """Official Muon Newton-Schulz quintic iteration."""
    assert g.ndim >= 2
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.bfloat16()
    if g.size(-2) > g.size(-1):
        x = x.mT

    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        a_mat = x @ x.mT
        b_mat = b * a_mat + c * a_mat @ a_mat
        x = a * x + b_mat @ x

    if g.size(-2) > g.size(-1):
        x = x.mT
    return x


def muon_update(grad: torch.Tensor, momentum: torch.Tensor, beta: float = 0.95, ns_steps: int = 5, nesterov: bool = True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    original_shape = update.shape
    if update.ndim == 4:
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update = update * max(1, update.size(-2) / update.size(-1)) ** 0.5
    return update.reshape(original_shape)


class Muon(Optimizer):
    """
    Strict Muon optimizer (no fallback optimizers).
    """

    def __init__(
        self,
        params,
        lr: float = 2e-3,
        weight_decay: float = 0.0,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        eps: float = 1e-8,
    ):
        if lr <= 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if not 0 <= momentum < 1:
            raise ValueError(f"Invalid momentum: {momentum}")
        if ns_steps < 1:
            raise ValueError(f"Invalid ns_steps: {ns_steps}")
        if eps <= 0:
            raise ValueError(f"Invalid eps: {eps}")

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            eps=eps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients")
                if p.ndim < 2:
                    raise RuntimeError(
                        f"Muon requires parameter ndim >= 2, got shape={tuple(p.shape)}"
                    )

                state = self.state[p]
                if wd != 0:
                    p.mul_(1.0 - lr * wd)

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                momentum_buffer = state["momentum_buffer"]
                update = muon_update(
                    grad,
                    momentum_buffer,
                    beta=momentum,
                    ns_steps=ns_steps,
                    nesterov=nesterov,
                )
                p.add_(update.to(p.dtype), alpha=-lr)

        return loss


def build_optimizer(params, args):
    params = list(params)
    optimizer_name = args.optimizer.lower()
    if optimizer_name == "adamw":
        return torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    if optimizer_name == "muon":
        invalid_shapes = [tuple(p.shape) for p in params if p.requires_grad and p.ndim < 2]
        if invalid_shapes:
            raise RuntimeError(
                "Strict Muon selected, but found parameters with ndim < 2. "
                f"Unsupported shapes: {invalid_shapes[:8]}"
            )
        return Muon(
            params,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            momentum=args.muon_momentum,
            nesterov=not args.no_muon_nesterov,
            ns_steps=args.muon_ns_steps,
            eps=args.muon_eps,
        )
    raise ValueError(f"Unsupported optimizer: {args.optimizer}")
