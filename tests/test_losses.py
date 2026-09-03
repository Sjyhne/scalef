import torch

from losses import BasicLosses, laplace_nll_loss, resolve_recon_criterion


def test_charbonnier_matches_sqrt_form_at_zero_residual():
    pred = torch.tensor([0.0, 0.1, -0.2])
    target = torch.zeros_like(pred)
    loss = BasicLosses.charbonnier_loss(pred, target, eps=1e-3)
    expected = torch.sqrt(pred * pred + 1e-6).mean()
    assert torch.allclose(loss, expected)


def test_huber_is_quadratic_near_zero_and_linear_far():
    pred = torch.tensor([0.0, 0.01, 0.2])
    target = torch.zeros_like(pred)
    delta = 0.05
    loss = BasicLosses.huber_loss(pred, target, delta=delta)
    r = (pred - target).abs()
    quad = torch.clamp(r, max=delta)
    lin = r - quad
    expected = (0.5 * quad * quad + delta * lin).mean()
    assert torch.allclose(loss, expected)


def test_resolve_recon_criterion_unknown_raises():
    try:
        resolve_recon_criterion("l2")
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "l2" in str(exc)


def test_resolve_recon_criterion_huber_uses_delta():
    crit = resolve_recon_criterion("huber", huber_delta=0.1)
    pred = torch.tensor([0.0, 0.3])
    target = torch.zeros_like(pred)
    loss = crit(pred, target)
    expected = BasicLosses.huber_loss(pred, target, delta=0.1)
    assert torch.allclose(loss, expected)


def test_laplace_nll_matches_manual_formula():
    pred = torch.tensor([0.0, 0.2, -0.1])
    target = torch.zeros_like(pred)
    scale = torch.tensor([0.1, 0.2, 0.5])
    loss = laplace_nll_loss(pred, target, scale)
    expected = ((pred - target).abs() / scale + torch.log(scale)).mean()
    assert torch.allclose(loss, expected)
