"""Training schedules: PSF curriculum and sigma ramp."""

from __future__ import annotations

from copy import copy
from types import SimpleNamespace
from typing import Sequence

DEFAULT_SCHEDULE_HORIZON_ITERS = 3000
DEFAULT_SCHEDULE_BOUNDARIES = (0.27, 0.53)


def parse_schedule_boundaries(raw: str | Sequence[float] | None) -> tuple[float, ...]:
    if raw is None:
        return DEFAULT_SCHEDULE_BOUNDARIES
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        return tuple(float(p) for p in parts) if parts else DEFAULT_SCHEDULE_BOUNDARIES
    return tuple(float(x) for x in raw)


def schedule_enabled(name: str | None) -> bool:
    return str(name or "none").lower().strip() not in {"", "none", "off", "disabled"}


def resolve_schedule_horizon_iters(args) -> int:
    raw = int(getattr(args, "schedule_horizon_iters", 0) or 0)
    return raw if raw > 0 else DEFAULT_SCHEDULE_HORIZON_ITERS


def resolve_schedule_boundaries(args) -> tuple[float, ...]:
    return parse_schedule_boundaries(getattr(args, "schedule_boundaries", None))


def iteration_progress(iteration: int, horizon_iters: int) -> float:
    horizon = max(1, int(horizon_iters))
    return min(1.0, max(0.0, float(iteration) / float(horizon)))


def _phase_index(progress: float, boundaries: Sequence[float]) -> int:
    bounds = tuple(sorted(boundaries))
    for idx, bound in enumerate(bounds):
        if progress < bound:
            return idx
    return len(bounds)


def _lerp(a: float, b: float, t: float) -> float:
    t = min(1.0, max(0.0, float(t)))
    return a + (b - a) * t


def resolve_psf_curriculum_mode(args, iteration: int) -> str | None:
    """Step curriculum: area -> s2_psf -> target degradation."""
    mode = str(getattr(args, "psf_curriculum", "none") or "none").lower().strip()
    if not schedule_enabled(mode):
        return None
    target = str(getattr(args, "lr_degradation", "s2_psf_m") or "s2_psf_m")
    if mode == "step":
        progress = iteration_progress(iteration, resolve_schedule_horizon_iters(args))
        phase = _phase_index(progress, resolve_schedule_boundaries(args))
        phases = ("area", "s2_psf", target)
        return phases[min(phase, len(phases) - 1)]
    raise ValueError(f"Unknown psf_curriculum {mode!r}; use none or step.")


def resolve_psf_sigma_scale(args, iteration: int) -> float | None:
    """Ramp s2_psf_m sigmas from min_scale -> 1.0."""
    mode = str(getattr(args, "psf_sigma_schedule", "none") or "none").lower().strip()
    if not schedule_enabled(mode):
        return None
    progress = iteration_progress(iteration, resolve_schedule_horizon_iters(args))
    min_scale = float(getattr(args, "psf_sigma_min_scale", 0.0) or 0.0)
    bounds = resolve_schedule_boundaries(args)
    if mode == "linear":
        return _lerp(min_scale, 1.0, progress)
    if mode == "step":
        phase = _phase_index(progress, bounds)
        scales = (min_scale, (min_scale + 1.0) * 0.5, 1.0)
        return scales[min(phase, len(scales) - 1)]
    raise ValueError(f"Unknown psf_sigma_schedule {mode!r}; use none, linear, or step.")


def effective_lr_align_args(args, iteration: int) -> SimpleNamespace:
    """Copy args with per-iteration PSF curriculum / sigma scaling applied."""
    align = copy(args)
    curriculum = resolve_psf_curriculum_mode(args, iteration)
    if curriculum is not None:
        align.lr_degradation = curriculum
    sigma_scale = resolve_psf_sigma_scale(args, iteration)
    if sigma_scale is not None:
        align.lr_degradation = str(getattr(args, "lr_degradation", "s2_psf_m") or "s2_psf_m")
        align.psf_sigma_scale = float(sigma_scale)
    elif not hasattr(align, "psf_sigma_scale"):
        align.psf_sigma_scale = 1.0
    return align
