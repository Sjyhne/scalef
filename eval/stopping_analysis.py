"""Analyze holdout-val curves vs HR metrics for early-stopping validation."""

from __future__ import annotations

from typing import Any


def _nearest_index(iters: list[int], target: int) -> int:
    if not iters:
        raise ValueError("empty iteration list")
    return min(range(len(iters)), key=lambda i: abs(int(iters[i]) - int(target)))


def simulate_patience_stop(
    val_iters: list[int],
    val_losses: list[float],
    *,
    patience: int,
    min_iters: int,
    min_delta: float = 0.0,
    ema_alpha: float = 0.0,
    max_regression: float = 0.0,
) -> dict[str, Any]:
    """Replay patience-based early stop on a logged val curve.

    Optional EMA smoothing and regression trip-wire match ``EarlyStopState``.
    """
    best_val = float("inf")
    best_iter = 0
    checks_without = 0
    stop_iter: int | None = None
    ema = float("nan")
    a = float(ema_alpha)

    for it, val in zip(val_iters, val_losses):
        it = int(it)
        raw = float(val)
        if a <= 0.0:
            decision = raw
            ema = raw
        elif ema != ema:  # NaN
            ema = raw
            decision = raw
        else:
            ema = a * raw + (1.0 - a) * ema
            decision = ema

        if it < int(min_iters):
            continue

        if decision < best_val - float(min_delta):
            best_val = decision
            best_iter = it
            checks_without = 0
        else:
            checks_without += 1
            reg = float(max_regression)
            if reg > 0.0 and decision > (best_val + reg):
                checks_without = max(checks_without, int(patience))

        if (
            int(patience) > 0
            and it >= int(min_iters)
            and checks_without >= int(patience)
        ):
            stop_iter = it
            break

    return {
        "best_val_iter": best_iter,
        "best_val_loss": best_val if best_val != float("inf") else None,
        "simulated_stop_iter": stop_iter,
        "patience": int(patience),
        "min_iters": int(min_iters),
        "min_delta": float(min_delta),
        "ema_alpha": float(ema_alpha),
        "max_regression": float(max_regression),
    }


def analyze_stopping_trajectory(
    metrics: dict[str, Any],
    *,
    patience: int = 5,
    min_iters: int = 1000,
    min_delta: float = 0.0,
    ema_alpha: float = 0.0,
    max_regression: float = 0.0,
    fixed_iters: tuple[int, ...] = (4000, 5000),
) -> dict[str, Any]:
    """Compare holdout stop vs oracle LPIPS vs fixed iteration endpoints."""
    hist = (metrics.get("training") or {}).get("history") or {}
    iters = [int(x) for x in hist.get("iterations") or []]
    val_loss = [float(x) for x in hist.get("val_loss") or []]
    lpips = [float(x) for x in hist.get("model_lpips") or []]
    psnr = [float(x) for x in hist.get("psnr") or []]

    if not iters or not val_loss:
        raise ValueError("metrics missing aligned val_loss history (run with spatial_holdout + HR eval)")

    if len(val_loss) != len(iters):
        # Fall back to early_stop val_history if training history is misaligned.
        es = metrics.get("early_stop") or {}
        vh = es.get("val_history") or {}
        iters = [int(x) for x in vh.get("iterations") or []]
        val_loss = [float(x) for x in vh.get("val_loss") or []]

    if len(lpips) != len(iters):
        raise ValueError(
            f"val/iters ({len(iters)}) and LPIPS ({len(lpips)}) length mismatch — "
            "run without --skip_eval (use --force_hr_eval with holdout_mse)"
        )

    sim = simulate_patience_stop(
        iters,
        val_loss,
        patience=patience,
        min_iters=min_iters,
        min_delta=min_delta,
        ema_alpha=ema_alpha,
        max_regression=max_regression,
    )
    best_lpips_i = min(range(len(lpips)), key=lambda i: lpips[i])
    oracle = {
        "iter": iters[best_lpips_i],
        "lpips": lpips[best_lpips_i],
        "psnr": psnr[best_lpips_i] if psnr else None,
    }

    def _pick(label: str, target_iter: int | None) -> dict[str, Any] | None:
        if target_iter is None:
            return None
        j = _nearest_index(iters, int(target_iter))
        return {
            "label": label,
            "target_iter": int(target_iter),
            "actual_iter": iters[j],
            "lpips": lpips[j],
            "psnr": psnr[j] if psnr else None,
            "val_loss": val_loss[j],
        }

    picks = {
        "oracle_best_lpips": _pick("oracle", oracle["iter"]),
        "simulated_early_stop": _pick("early_stop", sim.get("simulated_stop_iter")),
        "holdout_val_minimum": _pick("val_min", sim["best_val_iter"]),
        "final_logged": _pick("final", iters[-1]),
    }
    for t in fixed_iters:
        picks[f"fixed_{t}"] = _pick(f"fixed_{t}", t)

    def _regret_lpips(pick: dict | None) -> float | None:
        if pick is None:
            return None
        return float(pick["lpips"]) - float(oracle["lpips"])

    out = {
        **sim,
        "oracle_best_lpips": oracle,
        "completed_iters": int(metrics.get("completed_iters") or iters[-1]),
        "checkpoints": picks,
        "regret_lpips_vs_oracle": {
            k: _regret_lpips(v) for k, v in picks.items() if v is not None
        },
        "trajectory": {
            "iterations": iters,
            "val_loss": val_loss,
            "model_lpips": lpips,
            "psnr": psnr,
        },
    }
    return out
