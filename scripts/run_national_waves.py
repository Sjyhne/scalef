#!/usr/bin/env python3
"""Run a national wave plan durably, then build its cross-granule mosaic."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_national_production import atomic_write_json  # noqa: E402

WAVE_STAGES = "plan,fetch,tile,identity,cross_identity,production,delivery"
SUCCESS = {"success", "resumed", "adopted"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_object(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def validate_plan(plan: dict) -> None:
    expected_digest = plan.get("plan_sha256")
    canonical = {key: value for key, value in plan.items() if key != "plan_sha256"}
    actual_digest = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if expected_digest != actual_digest:
        raise ValueError("wave plan checksum is missing or invalid")
    inventory = plan.get("inventory")
    completed = plan.get("completed_granules")
    excluded = plan.get("excluded_granules")
    scheduled = plan.get("scheduled_granules")
    final = plan.get("final_mosaic_granules")
    waves = plan.get("waves")
    if not all(
        isinstance(value, list)
        for value in (inventory, completed, excluded, scheduled, final, waves)
    ):
        raise ValueError("wave plan has incomplete inventory fields")
    flattened = [
        mgrs
        for wave in waves
        for mgrs in (wave.get("granules", []) if isinstance(wave, dict) else [])
    ]
    if flattened != scheduled or len(flattened) != len(set(flattened)):
        raise ValueError("wave plan scheduled inventory does not exactly match its waves")
    groups = [set(completed), set(excluded), set(scheduled)]
    if any(groups[left] & groups[right] for left in range(3) for right in range(left + 1, 3)):
        raise ValueError("wave plan completed, excluded, and scheduled groups overlap")
    if set(inventory) != set(completed) | set(excluded) | set(scheduled):
        raise ValueError("wave plan runnable accounting is incomplete")
    if set(final) != set(inventory) - set(excluded):
        raise ValueError("wave plan final mosaic inventory is incomplete")
    if not final:
        raise ValueError("wave plan has no final mosaic granules")
    policy = plan.get("wave_size_policy") or {}
    minimum, maximum = int(policy.get("minimum", 6)), int(policy.get("maximum", 9))
    for index, wave in enumerate(waves):
        if wave.get("index") != index or wave.get("n_granules") != len(wave.get("granules", [])):
            raise ValueError(f"wave {index} metadata is inconsistent")
        size = len(wave["granules"])
        if len(scheduled) >= minimum and not minimum <= size <= maximum:
            raise ValueError(f"wave {index} size {size} is outside {minimum}-{maximum}")


def national_command(
    *,
    config: Path,
    granules: list[str],
    state_path: Path,
    stages: str,
    gpus: int,
    gpu_offset: int,
    stage_retries: int,
    stage_retry_backoff: float,
    cleanup_delivery_only: bool = False,
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_national_production.py"),
        "--config",
        str(config),
        "--granules",
        ",".join(granules),
        "--state",
        str(state_path),
        "--stages",
        stages,
        "--gpus",
        str(gpus),
        "--gpu-offset",
        str(gpu_offset),
        "--retries",
        str(stage_retries),
        "--retry-backoff",
        str(stage_retry_backoff),
        "--resume",
    ]
    if cleanup_delivery_only:
        command.append("--cleanup-delivery-only")
    return command


def validate_wave_state(path: Path, granules: list[str]) -> None:
    state = _load_object(path)
    if state.get("status") != "success":
        raise ValueError(f"{path}: wave status is not success")
    selected = state.get("selected_granules") or []
    if len(selected) != len(granules) or set(selected) != set(granules):
        raise ValueError(f"{path}: selected granules differ from wave plan")
    for mgrs in granules:
        delivery = (state.get("stages") or {}).get(f"{mgrs}:delivery", {})
        if delivery.get("status") not in SUCCESS:
            raise ValueError(f"{path}: {mgrs} delivery was not validated successfully")


def validate_cross_state(path: Path, granules: list[str]) -> None:
    state = _load_object(path)
    if state.get("status") != "success":
        raise ValueError(f"{path}: final cross status is not success")
    selected = state.get("selected_granules") or []
    if len(selected) != len(granules) or set(selected) != set(granules):
        raise ValueError(f"{path}: final cross granules differ from wave plan")
    if (state.get("stages") or {}).get("national:cross", {}).get("status") not in SUCCESS:
        raise ValueError(f"{path}: final cross mosaic was not validated successfully")


def _run_logged(
    command: list[str],
    log_path: Path,
    *,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> subprocess.CompletedProcess:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{utc_now()}] $ {shlex.join(command)}\n")
        log.flush()
        os.fsync(log.fileno())
        result = runner(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            start_new_session=True,
        )
        log.write(f"[{utc_now()}] returncode={result.returncode}\n")
        log.flush()
        os.fsync(log.fileno())
        return result


def execute_job(
    *,
    key: str,
    command: list[str],
    child_state: Path,
    log_path: Path,
    validate: Callable[[Path, list[str]], None],
    granules: list[str],
    state: dict,
    state_path: Path,
    retries: int,
    retry_backoff: float,
    dry_run: bool,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    sleeper: Callable[[float], None] = time.sleep,
) -> bool:
    previous = (state.get("jobs") or {}).get(key, {})
    if previous.get("status") in {"success", "resumed"}:
        try:
            validate(child_state, granules)
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        else:
            previous["status"] = "resumed"
            previous["updated_utc"] = utc_now()
            state["jobs"][key] = previous
            atomic_write_json(state_path, state)
            return True
    base = {
        "command": command,
        "command_shell": shlex.join(command),
        "child_state": str(child_state),
        "log": str(log_path),
        "granules": granules,
        "attempts": list(previous.get("attempts") or []),
    }
    if dry_run:
        state["jobs"][key] = {**base, "status": "planned", "updated_utc": utc_now()}
        atomic_write_json(state_path, state)
        print(f"PLAN {key}: {base['command_shell']}")
        return True
    for number in range(1, retries + 2):
        state["jobs"][key] = {
            **base,
            "status": "running",
            "active_attempt": number,
            "updated_utc": utc_now(),
        }
        atomic_write_json(state_path, state)
        started = time.monotonic()
        result = _run_logged(command, log_path, runner=runner)
        attempt = {
            "number": number,
            "returncode": result.returncode,
            "finished_utc": utc_now(),
            "duration_s": round(time.monotonic() - started, 3),
        }
        try:
            if result.returncode != 0:
                raise RuntimeError(f"national driver returned {result.returncode}")
            validate(child_state, granules)
        except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
            attempt.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            base["attempts"].append(attempt)
            state["jobs"][key] = {
                **base,
                "status": "failed",
                "error": attempt["error"],
                "updated_utc": utc_now(),
            }
            atomic_write_json(state_path, state)
            if number <= retries:
                sleeper(retry_backoff * (2 ** (number - 1)))
                continue
            return False
        attempt["status"] = "success"
        base["attempts"].append(attempt)
        state["jobs"][key] = {**base, "status": "success", "updated_utc": utc_now()}
        atomic_write_json(state_path, state)
        print(f"DONE {key}", flush=True)
        return True
    return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--gpu-offset", type=int, default=0)
    parser.add_argument("--retries", type=int, default=2, help="Whole-wave retries")
    parser.add_argument("--retry-backoff", type=float, default=30.0)
    parser.add_argument("--stage-retries", type=int, default=2)
    parser.add_argument("--stage-retry-backoff", type=float, default=5.0)
    parser.add_argument("--cleanup-delivery-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if min(args.retries, args.retry_backoff, args.stage_retries, args.stage_retry_backoff) < 0:
        raise SystemExit("retry counts and backoffs must be non-negative")
    if args.gpus < 1 or args.gpu_offset < 0:
        raise SystemExit("--gpus must be positive and --gpu-offset non-negative")
    plan_path = args.plan.resolve()
    try:
        plan = _load_object(plan_path)
        validate_plan(plan)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc
    config = Path(plan["country_config"])
    if not config.is_absolute():
        config = ROOT / config
    state_dir = (args.state_dir or plan_path.parent / "wave_states").resolve()
    log_dir = (args.log_dir or plan_path.parent / "wave_logs").resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / "wave_runner.json"
    lock_path = state_dir / "wave_runner.lock"

    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit(f"another wave runner holds {lock_path}") from exc
        state = (
            _load_object(state_path)
            if state_path.is_file()
            else {
                "schema_version": 1,
                "plan": str(plan_path),
                "plan_sha256": plan.get("plan_sha256"),
                "created_utc": utc_now(),
                "jobs": {},
            }
        )
        if state.get("plan_sha256") != plan.get("plan_sha256"):
            raise SystemExit(f"{state_path}: wave plan changed; use a new --state-dir")
        state.update({"status": "running", "last_started_utc": utc_now(), "dry_run": args.dry_run})
        atomic_write_json(state_path, state)

        for wave in plan["waves"]:
            index = wave["index"]
            granules = wave["granules"]
            child_state = state_dir / f"wave_{index:03d}.json"
            command = national_command(
                config=config,
                granules=granules,
                state_path=child_state,
                stages=WAVE_STAGES,
                gpus=args.gpus,
                gpu_offset=args.gpu_offset,
                stage_retries=args.stage_retries,
                stage_retry_backoff=args.stage_retry_backoff,
                cleanup_delivery_only=args.cleanup_delivery_only,
            )
            if not execute_job(
                key=f"wave:{index:03d}",
                command=command,
                child_state=child_state,
                log_path=log_dir / f"wave_{index:03d}.log",
                validate=validate_wave_state,
                granules=granules,
                state=state,
                state_path=state_path,
                retries=args.retries,
                retry_backoff=args.retry_backoff,
                dry_run=args.dry_run,
            ):
                state.update({"status": "failed", "finished_utc": utc_now()})
                atomic_write_json(state_path, state)
                return 1

        final_granules = plan["final_mosaic_granules"]
        final_state = state_dir / "final_cross.json"
        cross_command = national_command(
            config=config,
            granules=final_granules,
            state_path=final_state,
            stages="cross",
            gpus=args.gpus,
            gpu_offset=args.gpu_offset,
            stage_retries=args.stage_retries,
            stage_retry_backoff=args.stage_retry_backoff,
        )
        if not execute_job(
            key="final:cross",
            command=cross_command,
            child_state=final_state,
            log_path=log_dir / "final_cross.log",
            validate=validate_cross_state,
            granules=final_granules,
            state=state,
            state_path=state_path,
            retries=args.retries,
            retry_backoff=args.retry_backoff,
            dry_run=args.dry_run,
        ):
            state.update({"status": "failed", "finished_utc": utc_now()})
            atomic_write_json(state_path, state)
            return 1
        state.update(
            {
                "status": "planned" if args.dry_run else "success",
                "finished_utc": utc_now(),
            }
        )
        atomic_write_json(state_path, state)
        print(f"Wave runner state -> {state_path}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
