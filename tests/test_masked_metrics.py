import lpips
import pytest
import torch
from torchmetrics.functional.image import structural_similarity_index_measure as ssim

from eval.masked_metrics import masked_lpips, masked_ssim


@pytest.fixture(scope="module")
def lpips_fn():
    return lpips.LPIPS(net="vgg", verbose=False).eval()


def _pair(seed: int = 0, h: int = 96, w: int = 128):
    g = torch.Generator().manual_seed(seed)
    gt = torch.rand(1, 3, h, w, generator=g)
    pred = (gt + 0.2 * torch.rand(1, 3, h, w, generator=g)).clamp(0, 1)
    return pred, gt


def test_all_valid_mask_matches_unmasked(lpips_fn):
    pred, gt = _pair()
    mask = torch.ones(gt.shape[-2:], dtype=torch.bool)
    with torch.no_grad():
        ref_lpips = float(lpips_fn(pred * 2 - 1, gt * 2 - 1))
        assert masked_lpips(lpips_fn, pred, gt, mask) == pytest.approx(ref_lpips, rel=1e-5)
    assert masked_ssim(pred, gt, mask) == pytest.approx(float(ssim(pred, gt, data_range=1.0)), rel=1e-6)


def test_invalid_pixels_do_not_affect_scores(lpips_fn):
    pred, gt = _pair(1)
    mask = torch.ones(gt.shape[-2:], dtype=torch.bool)
    mask[:, 90:] = False
    gt = gt * mask
    other = pred.clone()
    other[..., 90:] = torch.rand_like(other[..., 90:])
    with torch.no_grad():
        assert masked_lpips(lpips_fn, pred, gt, mask) == pytest.approx(masked_lpips(lpips_fn, other, gt, mask))
    assert masked_ssim(pred, gt, mask) == pytest.approx(masked_ssim(other, gt, mask))
