"""Spatial LR pixel-block holdout for validation / early stopping.

Holds out blocks independently per frame so every revisit (including the
anchor) still trains, while each held block is an unseen observation. Whole-
frame holdout is intentionally not used: removing frame 0 frees the
radiometric/geometric gauge.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import torch
import torch.nn.functional as F


def build_holdout_mask(
    h: int,
    w: int,
    block: int,
    frac: float,
    device: torch.device | str | None = None,
    seed: int = 0,
) -> torch.Tensor:
    """Boolean ``[1,H,W,1]`` mask; ``True`` where a pixel is used for training.

    Pixels are held out in blocks rather than singly, because the hashgrid's
    finest level is comparable to the LR pitch and isolated pixels would be
    interpolated from their trained neighbours.
    """
    block = max(1, int(block))
    frac = float(frac)
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    bh = (h + block - 1) // block
    bw = (w + block - 1) // block
    keep_block = torch.rand((bh, bw), generator=g) >= frac
    mask = keep_block.repeat_interleave(block, 0).repeat_interleave(block, 1)[:h, :w]
    out = mask.view(1, h, w, 1)
    if device is not None:
        out = out.to(device)
    return out


def build_frame_masks(
    num_frames: int,
    h: int,
    w: int,
    *,
    block: int = 8,
    frac: float = 0.1,
    device: torch.device | str | None = None,
) -> list[torch.Tensor]:
    """One train-mask per frame; seed = frame index for independent holdouts."""
    return [
        build_holdout_mask(h, w, block, frac, device=device, seed=f)
        for f in range(int(num_frames))
    ]


def masked_mean(elem: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of an elementwise loss over selected pixels only.

    ``mask`` is broadcastable to ``elem`` (typically ``[B,H,W,1]`` bool/float).
    Divides by the held-in count (and channel count when ``elem`` has a trailing
    channel dim), unlike zeroing then ``.mean()`` over the full tensor.
    """
    m = mask.to(dtype=elem.dtype)
    while m.ndim < elem.ndim:
        m = m.unsqueeze(-1)
    # Average over selected spatial locations; also average channels.
    spatial = (elem * m).sum() / m.sum().clamp(min=1.0)
    if elem.ndim >= 1 and elem.shape[-1] > 1 and m.shape[-1] == 1:
        spatial = spatial / float(elem.shape[-1])
    return spatial


def gather_train_masks(
    masks: Sequence[torch.Tensor],
    sample_id: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Stack per-frame train masks for a batch of ``sample_id`` → ``[B,H,W,1]``."""
    ids = sample_id.reshape(-1).long().tolist()
    return torch.cat([masks[int(i)].to(device=device, non_blocking=True) for i in ids], dim=0)


def val_frame_ids(num_frames: int, max_frames: int = 4) -> list[int]:
    """Stable subset of frames for a quieter validation signal."""
    n = int(num_frames)
    if n <= 0:
        return []
    if n <= max_frames:
        return list(range(n))
    step = max(1, n // max_frames)
    ids = list(range(0, n, step))[:max_frames]
    return ids


def resolve_holdout_block(lr_height: int, lr_width: int, holdout_block: int) -> int:
    """Return LR holdout block side. ``0`` = auto-scale vs LR512@8 density."""
    requested = int(holdout_block)
    if requested > 0:
        return max(1, requested)
    side = max(int(lr_height), int(lr_width), 1)
    # LR512 with 8×8 → 64×64 block grid. Keep similar grid count on larger AOIs.
    return max(8, int(round(8 * side / 512.0)))


def resolve_holdout_patch_batch(holdout_block: int, patch_batch: int | None = None) -> int:
    """Mini-batch of holdout patches per forward. ``0``/None = auto from block size."""
    if patch_batch is not None and int(patch_batch) > 0:
        return max(1, int(patch_batch))
    block = max(1, int(holdout_block))
    # Target ~64 × 8² LR pixels per chunk, but never go below 64 launches-worth
    # of patches so large-block val still saturates the GPU.
    target_lr_px = 64 * 8 * 8
    return max(64, target_lr_px // (block * block))


EARLY_STOP_METRICS = ("holdout_mse", "lpips", "psnr", "mae")


def metric_higher_is_better(metric: str) -> bool:
    return str(metric).lower().strip() == "psnr"


def resolve_early_stop_score(
    metric: str,
    *,
    holdout_mse: float | None,
    hr_metrics: dict | None,
) -> tuple[str, float] | None:
    """Return (resolved_metric, score) for checkpointing / early stop."""
    name = str(metric).lower().strip()
    if name == "holdout_mse":
        if holdout_mse is None:
            return None
        return name, float(holdout_mse)
    if name == "lpips":
        if not hr_metrics or hr_metrics.get("model_lpips") is None:
            return None
        return name, float(hr_metrics["model_lpips"])
    if name == "psnr":
        if not hr_metrics:
            return None
        score = hr_metrics.get("test_psnr")
        if score is None:
            score = hr_metrics.get("model_psnr")
        if score is None:
            return None
        return name, float(score)
    if name == "mae":
        if not hr_metrics or hr_metrics.get("model_mae") is None:
            return None
        return name, float(hr_metrics["model_mae"])
    raise ValueError(
        f"Unknown early_stop_metric {metric!r}; use holdout_mse, lpips, psnr, or mae."
    )


@dataclass
class EarlyStopState:
    """Tracks a validation score and the best checkpoint."""

    train_masks: list[torch.Tensor]
    val_ids: list[int]
    patience: int
    min_iters: int
    spatial_masks: list[torch.Tensor] | None = None
    clear_masks: list[torch.Tensor] | None = None
    metric: str = "lpips"
    min_delta: float = 0.0
    max_regression: float = 0.0
    ema_alpha: float = 0.0
    holdout_block: int = 8
    best_score: float = field(default_factory=lambda: float("nan"))
    best_iter: int = 0
    best_state: dict | None = None
    checks_without_improve: int = 0
    stopped: bool = False
    stopped_iter: int | None = None
    score_history_iters: list[int] = field(default_factory=list)
    score_history: list[float] = field(default_factory=list)
    score_ema_history: list[float] = field(default_factory=list)
    holdout_mse_history: list[float] = field(default_factory=list)
    score_ema: float = field(default_factory=lambda: float("nan"))
    restored: bool = False

    def __post_init__(self) -> None:
        self.metric = str(self.metric).lower().strip()
        if self.metric not in EARLY_STOP_METRICS:
            raise ValueError(f"Unknown early_stop_metric {self.metric!r}")
        self.ema_alpha = float(self.ema_alpha)
        if not (0.0 <= self.ema_alpha <= 1.0):
            raise ValueError(f"ema_alpha must be in [0, 1], got {self.ema_alpha}")
        if metric_higher_is_better(self.metric):
            if not math.isfinite(self.best_score):
                self.best_score = float("-inf")
        elif not math.isfinite(self.best_score):
            self.best_score = float("inf")

    @property
    def enabled_stop(self) -> bool:
        return int(self.patience) > 0

    @property
    def best_val(self) -> float:
        """Backward-compatible alias for the tracked stop score."""
        return self.best_score

    def _smooth(self, score: float) -> float:
        """EMA of the stop metric; alpha=0 → raw score (no smoothing)."""
        raw = float(score)
        a = float(self.ema_alpha)
        if a <= 0.0:
            self.score_ema = raw
            return raw
        if not math.isfinite(self.score_ema):
            self.score_ema = raw
        else:
            self.score_ema = a * raw + (1.0 - a) * float(self.score_ema)
        return float(self.score_ema)

    def _is_improvement(self, score: float) -> bool:
        delta = float(self.min_delta)
        if metric_higher_is_better(self.metric):
            return score > (self.best_score + delta)
        return score < (self.best_score - delta)

    def observe(
        self,
        iteration: int,
        score: float,
        model: torch.nn.Module,
        *,
        holdout_mse: float | None = None,
    ) -> bool:
        """Record stop metric; return True if training should stop.

        Checks before ``min_iters`` are logged but do not update the best
        checkpoint or consume patience (warmup). When ``ema_alpha>0``,
        patience / regression / best-score use the EMA of ``score``.
        """
        decision = self._smooth(score)
        self.score_history_iters.append(int(iteration))
        self.score_history.append(float(score))
        self.score_ema_history.append(float(decision))
        if holdout_mse is not None:
            self.holdout_mse_history.append(float(holdout_mse))
        if iteration < int(self.min_iters):
            return False
        improved = self._is_improvement(decision)
        if improved:
            self.best_score = float(decision)
            self.best_iter = int(iteration)
            self.best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            self.checks_without_improve = 0
        else:
            self.checks_without_improve += 1
            reg = float(self.max_regression)
            if reg > 0.0 and math.isfinite(self.best_score):
                if metric_higher_is_better(self.metric):
                    regressed = decision < (self.best_score - reg)
                else:
                    regressed = decision > (self.best_score + reg)
                if regressed:
                    self.checks_without_improve = max(
                        self.checks_without_improve, int(self.patience)
                    )

        if (
            self.enabled_stop
            and iteration >= int(self.min_iters)
            and self.checks_without_improve >= int(self.patience)
        ):
            self.stopped = True
            self.stopped_iter = int(iteration)
            return True
        return False

    def restore_best(self, model: torch.nn.Module) -> bool:
        if self.best_state is None:
            return False
        model.load_state_dict(self.best_state)
        self.restored = True
        return True

    def summary(self) -> dict:
        best = self.best_score
        if metric_higher_is_better(self.metric) and best == float("-inf"):
            best = None
        elif not metric_higher_is_better(self.metric) and best == float("inf"):
            best = None
        return {
            "metric": self.metric,
            "holdout_block": int(self.holdout_block),
            "best_score": best,
            "max_regression": float(self.max_regression),
            "ema_alpha": float(self.ema_alpha),
            "best_val_loss": best if self.metric == "holdout_mse" else None,
            "best_val_iter": self.best_iter,
            "stopped": self.stopped,
            "stopped_iter": self.stopped_iter,
            "restored_best": self.restored,
            "checks_without_improve": self.checks_without_improve,
            "score_history": {
                "iterations": list(self.score_history_iters),
                "values": list(self.score_history),
                "ema": list(self.score_ema_history),
            },
            "val_history": {
                "iterations": list(self.score_history_iters),
                "val_loss": list(self.holdout_mse_history or self.score_history),
            },
        }


def default_early_stop_regression(metric: str) -> float:
    """Regression tolerance before forcing stop (lower-is-better metrics only)."""
    name = str(metric).lower().strip()
    if name == "lpips":
        return 0.003
    if name == "mae":
        return 0.002
    if name == "holdout_mse":
        # Absolute on typical Charbonnier holdout (~0.3–0.4). Zero previously
        # left fused-k to die only on patience, which is noisy under tile sampling.
        return 0.01
    if name == "psnr":
        return 0.05
    return 0.0


def default_early_stop_ema(metric: str) -> float:
    """Default EMA alpha for the stop score (0 = raw / no smoothing)."""
    name = str(metric).lower().strip()
    if name == "holdout_mse":
        return 0.4
    return 0.0


def init_early_stop_state(
    *,
    num_frames: int,
    lr_height: int,
    lr_width: int,
    spatial_holdout: float,
    holdout_block: int,
    patience: int,
    min_iters: int,
    min_delta: float = 0.0,
    metric: str = "lpips",
    max_regression: float | None = None,
    ema_alpha: float | None = None,
    device: torch.device | str | None = None,
) -> EarlyStopState | None:
    """Build masks + early-stop tracker, or ``None`` if holdout is disabled."""
    frac = float(spatial_holdout)
    if frac <= 0.0:
        return None
    block = resolve_holdout_block(lr_height, lr_width, holdout_block)
    masks = build_frame_masks(
        num_frames,
        int(lr_height),
        int(lr_width),
        block=block,
        frac=frac,
        device=device,
    )
    held = 100.0 * float(torch.stack([(~m).float().mean() for m in masks]).mean())
    metric = str(metric).lower().strip()
    if max_regression is None:
        max_regression = default_early_stop_regression(metric)
    if ema_alpha is None:
        ema_alpha = default_early_stop_ema(metric)
    auto = int(holdout_block) <= 0
    print(
        f"spatial holdout: {held:.1f}% of each frame's LR pixels held out in "
        f"{block}x{block} blocks"
        f"{' (auto)' if auto else ''} "
        f"(frac={frac:.3f}, stop_metric={metric}, "
        f"patience={int(patience)}, min_iters={int(min_iters)}, "
        f"min_delta={float(min_delta):g}, max_regression={float(max_regression):g}, "
        f"ema_alpha={float(ema_alpha):g})"
    )
    return EarlyStopState(
        train_masks=masks,
        spatial_masks=list(masks),
        val_ids=val_frame_ids(num_frames),
        patience=int(patience),
        min_iters=int(min_iters),
        min_delta=float(min_delta),
        metric=metric,
        max_regression=float(max_regression),
        ema_alpha=float(ema_alpha),
        holdout_block=block,
    )


def apply_lr_clear_masks(state: EarlyStopState, clear: torch.Tensor) -> EarlyStopState:
    """AND per-frame SCL/cloud clear maps into train masks; keep spatial holdout for val."""
    from eval.s2_cloud_mask import apply_clear_to_holdout_masks, frame_clear_fractions

    spatial = list(state.spatial_masks or state.train_masks)
    anded = apply_clear_to_holdout_masks(spatial, clear)
    clear_list = [
        clear[i].to(dtype=torch.bool).view(1, *clear.shape[1:], 1)
        if clear[i].ndim == 2
        else clear[i].to(dtype=torch.bool)
        for i in range(int(clear.shape[0]))
    ]
    fracs = frame_clear_fractions(clear)
    mean_clear = float(sum(fracs) / max(len(fracs), 1))
    print(
        f"LR cloud mask: mean clear {mean_clear * 100:.1f}% of pixels "
        f"(min {min(fracs) * 100:.1f}%, {sum(1 for f in fracs if f < 0.05)} frame(s) <5% clear)",
        flush=True,
    )
    state.spatial_masks = spatial
    state.train_masks = anded
    state.clear_masks = clear_list
    return state


def holdout_block_origins(
    train_mask: torch.Tensor,
    block: int,
) -> list[tuple[int, int, int, int]]:
    """LR block windows ``(row, col, h, w)`` that contain any held-out pixel.

    ``train_mask`` is ``True`` on train pixels, shape ``[1,H,W,1]`` or ``[H,W]``.
    """
    m = train_mask.reshape(train_mask.shape[-3], train_mask.shape[-2]) if train_mask.ndim >= 3 else train_mask
    if m.ndim != 2:
        m = m.view(int(m.shape[-2]), int(m.shape[-1]))
    h, w = int(m.shape[0]), int(m.shape[1])
    block = max(1, int(block))
    origins: list[tuple[int, int, int, int]] = []
    for row in range(0, h, block):
        for col in range(0, w, block):
            bh = min(block, h - row)
            bw = min(block, w - col)
            patch = m[row : row + bh, col : col + bw]
            if not bool(patch.all().item()):
                origins.append((row, col, bh, bw))
    return origins


def holdout_psf_pad_lr(df: int, *, sigma_m: float = 4.2, truncate: float = 4.0, native_gsd_m: float = 10.0) -> int:
    """LR pixels of context needed so a cropped PSF matches the full-frame degrade."""
    import math

    df = max(1, int(df))
    gsd_hr = float(native_gsd_m) / float(df)
    radius_hr = max(1, int(math.ceil(float(truncate) * float(sigma_m) / gsd_hr)))
    return max(1, (radius_hr + df - 1) // df)


def _padded_lr_window(
    row: int, col: int, bh: int, bw: int, lr_h: int, lr_w: int, pad: int
) -> tuple[int, int, int, int, int, int]:
    r0 = max(0, row - pad)
    c0 = max(0, col - pad)
    r1 = min(lr_h, row + bh + pad)
    c1 = min(lr_w, col + bw + pad)
    return r0, c0, r1, c1, row - r0, col - c0


def _group_padded_origins(
    origins: list[tuple[int, int, int, int]],
    lr_h: int,
    lr_w: int,
    pad: int,
) -> dict[tuple[int, ...], list[tuple[int, int, int, int]]]:
    grouped: dict[tuple[int, ...], list[tuple[int, int, int, int]]] = {}
    for origin in origins:
        row, col, bh, bw = origin
        r0, c0, r1, c1, ir, ic = _padded_lr_window(row, col, bh, bw, lr_h, lr_w, pad)
        key = (r1 - r0, c1 - c0, ir, ic, bh, bw)
        grouped.setdefault(key, []).append(origin)
    return grouped


def _stack_lr_hr_patches(
    hr_coords_hw2: torch.Tensor,
    lr_target_hwc: torch.Tensor,
    train_mask_hw: torch.Tensor,
    origins: list[tuple[int, int, int, int]],
    df: int,
    *,
    lr_h: int,
    lr_w: int,
    pad: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Padded HR/LR stacks plus inner offset so PSF context is included."""
    coords = []
    targets = []
    hold_masks = []
    inner_r = inner_c = bh = bw = 0
    for row, col, bh, bw in origins:
        r0, c0, r1, c1, inner_r, inner_c = _padded_lr_window(row, col, bh, bw, lr_h, lr_w, pad)
        coords.append(hr_coords_hw2[r0 * df : r1 * df, c0 * df : c1 * df])
        targets.append(lr_target_hwc[r0:r1, c0:c1])
        hold_masks.append(~train_mask_hw[row : row + bh, col : col + bw])
    return (
        torch.stack(coords, dim=0),
        torch.stack(targets, dim=0),
        torch.stack(hold_masks, dim=0).unsqueeze(-1),
        inner_r,
        inner_c,
    )


@torch.no_grad()
def compute_holdout_val_loss(
    model: torch.nn.Module,
    dataset,
    state: EarlyStopState,
    args,
    device: torch.device,
    step_kwargs: dict | None = None,
) -> float:
    """Reconstruction loss on held-out LR blocks only (no full-frame HR render)."""
    was_training = model.training
    model.eval()
    hr_coords = dataset.get_hr_coordinates().to(device)
    recon_loss = str(getattr(args, "recon_loss", "mse") or "mse")
    charbonnier_eps = float(getattr(args, "charbonnier_eps", 1e-3) or 1e-3)
    huber_delta = float(getattr(args, "huber_delta", 0.05) or 0.05)
    fwd = dict(step_kwargs or {})
    lr_h = int(getattr(dataset, "lr_height", 0) or 0)
    if lr_h <= 0:
        lr_h = int(dataset.get_lr_sample_hwc(0).shape[0])
    lr_w = int(getattr(dataset, "lr_width", 0) or 0)
    if lr_w <= 0:
        lr_w = int(dataset.get_lr_sample_hwc(0).shape[1])
    block = int(getattr(state, "holdout_block", 0) or 0)
    if block <= 0:
        block = resolve_holdout_block(lr_h, lr_w, int(getattr(args, "holdout_block", 0) or 0))
    raw_batch = getattr(args, "holdout_patch_batch", None)
    patch_batch = resolve_holdout_patch_batch(
        block, None if raw_batch is None else int(raw_batch)
    )
    df = int(hr_coords.shape[0]) // int(lr_h)
    if df < 1 or hr_coords.shape[0] != df * lr_h:
        raise ValueError(
            f"HR {tuple(hr_coords.shape[:2])} is not an integer multiple of LR height {lr_h}."
        )

    vals: list[torch.Tensor] = []
    raw_pad = getattr(args, "holdout_psf_pad_lr", None)
    if raw_pad is None or int(raw_pad) < 0:
        pad = holdout_psf_pad_lr(df)
    else:
        pad = int(raw_pad)
    for f in state.val_ids:
        target = dataset.get_lr_sample_hwc(int(f)).to(device)
        spatial = (state.spatial_masks or state.train_masks)[int(f)].to(device)
        spatial_hw = spatial.reshape(spatial.shape[-3], spatial.shape[-2])
        if state.clear_masks is not None:
            clear_f = state.clear_masks[int(f)].to(device)
            if clear_f.ndim >= 3:
                clear_f = clear_f.reshape(clear_f.shape[-3], clear_f.shape[-2])
            val_hw = (~spatial_hw) & clear_f.bool()
        else:
            val_hw = ~spatial_hw
        origins = holdout_block_origins(spatial, block)
        if not origins:
            continue
        for key, group in _group_padded_origins(origins, lr_h, lr_w, pad).items():
            _ph, _pw, inner_r, inner_c, bh, bw = key
            for i0 in range(0, len(group), patch_batch):
                chunk = group[i0 : i0 + patch_batch]
                coords_b, target_b, hold_b, inner_r, inner_c = _stack_lr_hr_patches(
                    hr_coords,
                    target,
                    ~val_hw,
                    chunk,
                    df,
                    lr_h=lr_h,
                    lr_w=lr_w,
                    pad=pad,
                )
                n = coords_b.shape[0]
                sid = torch.full((n,), int(f), device=device, dtype=torch.long)
                out = model(coords_b, sid, lr_frames=target_b, **fwd)
                pred = out[0] if isinstance(out, (tuple, list)) else out
                pred = pred[..., :3]
                pred = pred[:, inner_r : inner_r + bh, inner_c : inner_c + bw]
                target_inner = target_b[:, inner_r : inner_r + bh, inner_c : inner_c + bw]
                elem = elementwise_recon(
                    recon_loss,
                    pred,
                    target_inner,
                    charbonnier_eps=charbonnier_eps,
                    huber_delta=huber_delta,
                )
                vals.append(masked_mean(elem, hold_b))
    if was_training:
        model.train()
    if not vals:
        return float("nan")
    return float(torch.stack(vals).mean().item())


def elementwise_recon(
    recon_loss: str,
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    charbonnier_eps: float = 1e-3,
    huber_delta: float = 0.05,
) -> torch.Tensor:
    """Per-pixel reconstruction residual for masking."""
    name = str(recon_loss).lower().strip()
    if name == "mse":
        return (pred - target) ** 2
    if name == "mae":
        return (pred - target).abs()
    if name == "charbonnier":
        return torch.sqrt((pred - target) ** 2 + float(charbonnier_eps) ** 2)
    if name == "huber":
        a = (pred - target).abs()
        d = float(huber_delta)
        q = a.clamp(max=d)
        return 0.5 * q * q + d * (a - q)
    raise ValueError(f"Unknown recon_loss {recon_loss!r}")
