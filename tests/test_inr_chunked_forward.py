"""Row-chunked encode/decode must match a single unchunked pass."""

from __future__ import annotations

import torch

from models.inr import INRBase


class _Decoder(torch.nn.Module):
    def __init__(self, in_dim: int = 2, out_dim: int = 3):
        super().__init__()
        self.net = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.net(x)


def _module() -> INRBase:
    torch.manual_seed(0)
    module = INRBase(None, _Decoder(), num_samples=4)
    with torch.no_grad():
        for sid in range(1, 4):
            module.affine_params[sid].copy_(
                torch.tensor([[1.01, 0.02, -0.01, 0.03, 0.99, 0.02]])
            )
    return module


def test_chunked_matches_unchunked_forward():
    module = _module()
    coords = torch.rand(2, 9, 7, 2)
    sid = torch.tensor([1, 3])

    module.max_forward_rows = 0
    full, _ = module._encode_and_decode(coords, sid)

    n_rows = coords.shape[0] * coords.shape[1] * coords.shape[2]
    module.max_forward_rows = 16
    assert n_rows > module.max_forward_rows
    chunked, _ = module._encode_and_decode(coords, sid)

    assert chunked.shape == full.shape
    torch.testing.assert_close(chunked, full, rtol=1e-6, atol=1e-6)


def test_chunked_matches_unchunked_gradients():
    module = _module()
    sid = torch.tensor([1, 3])
    grads = []
    for max_rows in (0, 16):
        module.max_forward_rows = max_rows
        module.zero_grad(set_to_none=True)
        coords = torch.rand(2, 9, 7, 2, generator=torch.Generator().manual_seed(3))
        coords.requires_grad_(True)
        out, _ = module._encode_and_decode(coords, sid)
        out.square().sum().backward()
        grads.append((coords.grad.clone(), module.affine_params[1].grad.clone()))

    torch.testing.assert_close(grads[0][0], grads[1][0], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(grads[0][1], grads[1][1], rtol=1e-5, atol=1e-6)


def test_default_row_cap_is_below_uint32_limit():
    # rows * hidden width must stay under 2**32 for the fused MLP.
    assert INRBase.MAX_FORWARD_ROWS * 256 < 2**32
