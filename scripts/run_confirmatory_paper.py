#!/usr/bin/env python3
"""Canonical, declarative runner for new confirmatory paper experiments.

This runner deliberately uses ``confirmatory_v1`` sample IDs, run names, and
JSON outputs. It never discovers or imports metrics from legacy ``paper_*``
runs.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent.parent
NAMESPACE = "confirmatory_v1"
DEFAULT_EVAL_MANIFEST = ROOT / "eval" / "confirmatory_eval_manifest.v1.json"
DEFAULT_RUN_MANIFEST = (
    ROOT / "single_samples" / "sweep_results" / NAMESPACE / "run_manifest.json"
)

PAPER7 = ("asker", "bergen", "rana", "tromso", "amli", "vennesla", "trondheim")
CITY_SETS = {
    "paper7": PAPER7,
    "b0_17": (
        "algard",
        "alta",
        "amli",
        "asker",
        "bergen",
        "flekkefjord",
        "karasjok",
        "kautokeino",
        "melhus",
        "naerbo",
        "nittedal",
        "rafsbotn",
        "rana",
        "stavanger",
        "tromso",
        "trondheim",
        "vennesla",
    ),
    "asker": ("asker",),
}

FAMILY_DEFAULT_CITY_SET = {
    "fixed_k": "paper7",
    "encoding_size": "paper7",
    "gsd_ladder": "paper7",
    "misr_controls": "paper7",
    "mtf_sigma": "paper7",
    "b0_17": "b0_17",
}
FAMILIES = tuple(FAMILY_DEFAULT_CITY_SET)

BASE_FLAGS: tuple[tuple[str, str], ...] = (
    ("--df", "4"),
    ("--scale_factor", "4"),
    ("--num_samples", "16"),
    ("--input_projection", "hashgrid_tcnn"),
    ("--lr_degradation", "s2_psf_m"),
    ("--recon_loss", "charbonnier"),
    ("--charbonnier_eps", "0.01"),
    ("--lr_tile", "128"),
    ("--lr_tiles_per_step", "4"),
    ("--lr_tile_mix", "within"),
    ("--early_stop_metric", "holdout_mse"),
    ("--early_stop_patience", "8"),
    ("--early_stop_min_iters", "1000"),
    ("--eval_every", "200"),
    ("--spatial_holdout", "0.1"),
    ("--holdout_block", "0"),
    ("--hr_render_tile", "2048"),
    ("--lr_stats_pixels", "all"),
    ("--lr_tile_halo", "-1"),
)

# Controls that still cannot be represented faithfully. Freeze-align and
# freeze-radiometry *are* wired (``--freeze_affines`` / ``--freeze_radiometry``).
# Single-frame / repeated-frame packages are a data gap, not a flag gap.
BLOCKED_CONTROLS = (
    {
        "id": "misr_nonreference_base_frame",
        "reason": "--no_base_frame is parsed but is not consumed by optimize.py/model construction",
    },
    {
        "id": "misr_indirect_transform_parameterization",
        "reason": "--no_direct_param_T is parsed but is not consumed by optimize.py/model construction",
    },
)

DEFAULT_SIGMAS_M = {
    "--s2-psf-sigma-b02-m": 2.8,
    "--s2-psf-sigma-b03-m": 3.25,
    "--s2-psf-sigma-b04-m": 4.2,
    "--s2-psf-sigma-b08-m": 3.5,
}


@dataclass(frozen=True)
class Config:
    family: str
    config: str
    flags: tuple[tuple[str, str], ...]
    lr_size: int | None = None
    data_dir_template: str | None = None


@dataclass(frozen=True)
class Job:
    job_id: str
    run_name: str
    family: str
    config: str
    city: str
    seed: int
    gpu: int
    s2_dir: str
    available_frames: int | None
    requested_frames: int
    command: list[str]
    command_shell: str
    expected_metrics_path: str


def family_configs(family: str) -> tuple[Config, ...]:
    if family == "fixed_k":
        configs = [
            Config(family, f"lr128_k{k}", (("--lr_tile", "128"), ("--lr_tiles_per_step", str(k))))
            for k in (1, 2, 4, 8)
        ]
        configs.append(
            Config(family, "lr128_full", (("--lr_tile", "0"), ("--lr_tiles_per_step", "1")))
        )
        return tuple(configs)
    if family == "gsd_ladder":
        # Train-time 5 m. Query-time 5 m / 1 m reuses the ship df=4 field.
        # 1 m train-time is Asker-only and launched separately.
        return (
            Config(
                family,
                "df2_5m",
                (
                    ("--df", "2"),
                    ("--scale_factor", "2"),
                    ("--hr_render_tile", "1024"),
                ),
            ),
            Config(family, "df4_query", (("--query_gsd_m", "5,1"),)),
        )
    if family == "encoding_size":
        configs = []
        for size in (64, 128, 256, 512):
            full = (("--lr_tile", "0"), ("--lr_tiles_per_step", "1"))
            configs.append(
                Config(family, f"lr{size}_hash", (*full, ("--input_projection", "hashgrid_tcnn")), size)
            )
            for scale in (2, 5, 10):
                configs.append(
                    Config(
                        family,
                        f"lr{size}_fourier_s{scale}",
                        (*full, ("--input_projection", "fourier"), ("--fourier_scale", str(scale))),
                        size,
                    )
                )
        return tuple(configs)
    if family == "misr_controls":
        return (
            Config(family, "joint", (), 512),
            Config(family, "frozen_affines", (("--freeze_affines", ""),), 512),
            Config(family, "frozen_radiometry", (("--freeze_radiometry", ""),), 512),
            Config(
                family,
                "single_base",
                (),
                data_dir_template="{city}_control_single_lr512",
            ),
            Config(
                family,
                "repeated_base",
                (),
                data_dir_template="{city}_control_repeated_lr512",
            ),
        )
    if family == "mtf_sigma":
        configs = []
        for label, multiplier in (("x0p75", 0.75), ("x1p00", 1.0), ("x1p25", 1.25)):
            flags = tuple(
                (flag, f"{sigma * multiplier:g}") for flag, sigma in DEFAULT_SIGMAS_M.items()
            )
            configs.append(Config(family, label, flags))
        return tuple(configs)
    if family == "b0_17":
        # The confirmatory 17-site table compares one spatial unit. Refuse to
        # mix the native ~256 px packages with the seven existing LR512 crops.
        return (Config(family, "lr512_b0", (), 512),)
    raise ValueError(f"unknown family {family!r}")


def _merge_flags(*groups: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    merged: dict[str, str] = {}
    for group in groups:
        for flag, value in group:
            merged[flag] = value
    return list(merged.items())


def _append_flags(argv: list[str], flags: Iterable[tuple[str, str]]) -> None:
    """Append CLI flags. Empty values mean store_true (flag with no argument)."""
    for flag, value in flags:
        if value == "":
            argv.append(flag)
        else:
            argv.extend((flag, value))


def _configs_for_family(family: str, selected: tuple[str, ...] | None) -> tuple[Config, ...]:
    configs = family_configs(family)
    if not selected:
        return configs
    wanted = set(selected)
    filtered = tuple(config for config in configs if config.config in wanted)
    if not filtered:
        known = ", ".join(config.config for config in configs)
        raise ValueError(f"no configs in {family!r} match {sorted(wanted)}; known: {known}")
    return filtered


def load_eval_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "scalef.confirmatory_eval.v1":
        raise ValueError(f"{path} is not a scalef.confirmatory_eval.v1 manifest")
    if not isinstance(manifest.get("cities"), dict) or not manifest["cities"]:
        raise ValueError(f"{path} has no frozen cities")
    return manifest


def _cities_for_family(
    family: str, *, city_set: str | None, cities: tuple[str, ...] | None
) -> tuple[str, ...]:
    if cities:
        return cities
    selected = city_set or FAMILY_DEFAULT_CITY_SET[family]
    return CITY_SETS[selected]


def _s2_dir(root: Path, city: str, config: Config, eval_manifest: dict[str, Any]) -> Path:
    if config.data_dir_template is not None:
        dirname = config.data_dir_template.format(city=city)
    elif config.lr_size is not None:
        dirname = f"{city}_lr{config.lr_size}"
    else:
        entry = eval_manifest["cities"].get(city)
        if entry is None:
            raise ValueError(f"city {city!r} is absent from the frozen eval manifest")
        dirname = entry["s2_dir_name"]
    return root / "data" / "s2_revisits" / dirname


def _validate_data(path: Path, city: str, config: Config, requested_frames: int) -> int:
    if not path.is_dir():
        raise FileNotFoundError(
            f"missing requested data for {config.family}/{config.config}/{city}: {path}"
        )
    meta = path / "meta.json"
    if not meta.is_file():
        raise FileNotFoundError(
            f"missing metadata for {config.family}/{config.config}/{city}: {meta}"
        )
    payload = json.loads(meta.read_text(encoding="utf-8"))
    frames = payload.get("frames")
    if not frames:
        raise ValueError(f"{meta} has no frames; refusing to change confirmatory scope")
    if requested_frames > 0 and len(frames) < requested_frames:
        raise ValueError(
            f"{meta} has {len(frames)} frames but config requires {requested_frames}; "
            "refusing to silently reduce the confirmatory sample count"
        )
    return len(frames)


def generate_jobs(
    *,
    families: tuple[str, ...],
    seeds: tuple[int, ...],
    eval_manifest: dict[str, Any],
    eval_manifest_path: Path = DEFAULT_EVAL_MANIFEST,
    root: Path = ROOT,
    city_set: str | None = None,
    cities: tuple[str, ...] | None = None,
    configs: tuple[str, ...] | None = None,
    iters: int = 5000,
    gpus: int = 1,
    gpu_offset: int = 0,
    validate_data: bool = True,
    namespace: str = NAMESPACE,
    gpu_list: tuple[int, ...] | None = None,
) -> list[Job]:
    if not families:
        raise ValueError("at least one family is required")
    if not seeds:
        raise ValueError("at least one seed is required")
    if gpus < 1:
        raise ValueError("gpus must be >= 1")

    specs: list[tuple[Config, str, int, Path, int | None, int]] = []
    for family in families:
        for config in _configs_for_family(family, configs):
            for city in _cities_for_family(family, city_set=city_set, cities=cities):
                entry = eval_manifest["cities"].get(city)
                if entry is None:
                    raise ValueError(f"city {city!r} is absent from the frozen eval manifest")
                requested_frames = int(entry.get("requested_frames", 16))
                if config.family == "misr_controls" and config.config == "single_base":
                    requested_frames = 1
                path = _s2_dir(root, city, config, eval_manifest)
                available_frames = None
                if validate_data:
                    available_frames = _validate_data(path, city, config, requested_frames)
                for seed in seeds:
                    specs.append(
                        (config, city, seed, path, available_frames, requested_frames)
                    )

    jobs: list[Job] = []
    for index, (
        config,
        city,
        seed,
        s2_dir,
        available_frames,
        requested_frames,
    ) in enumerate(specs):
        ns = str(namespace)
        run_name = f"{ns}__{config.family}__{config.config}__{city}__seed{seed}"
        job_id = run_name
        gpu = gpu_list[index % len(gpu_list)] if gpu_list else gpu_offset + (index % gpus)
        metrics = (
            root
            / "single_samples"
            / city
            / ns
            / run_name
            / "metrics.json"
        )
        argv = [
            sys.executable,
            str(root / "optimize.py"),
            "--dataset",
            city,
            "--sample_id",
            ns,
            "--s2-dir",
            str(s2_dir),
            "--run_name",
            run_name,
            "--seed",
            str(seed),
            "--iters",
            str(iters),
            "--device",
            str(gpu),
            "--force_hr_eval",
            "--spatial_alignment_path",
            str(eval_manifest_path),
            "--no_qgis_export",
        ]
        _append_flags(
            argv,
            _merge_flags(
                BASE_FLAGS,
                config.flags,
                (("--num_samples", str(requested_frames)),),
            ),
        )
        jobs.append(
            Job(
                job_id=job_id,
                run_name=run_name,
                family=config.family,
                config=config.config,
                city=city,
                seed=seed,
                gpu=gpu,
                s2_dir=str(s2_dir),
                available_frames=available_frames,
                requested_frames=requested_frames,
                command=argv,
                command_shell=shlex.join(argv),
                expected_metrics_path=str(metrics),
            )
        )

    ids = [job.job_id for job in jobs]
    if len(ids) != len(set(ids)):
        duplicates = sorted({job_id for job_id in ids if ids.count(job_id) > 1})
        raise ValueError(f"duplicate confirmatory job IDs: {duplicates}")
    return jobs


def _write_manifest(
    path: Path,
    jobs: list[Job],
    *,
    eval_manifest_path: Path,
    eval_manifest: dict[str, Any],
    dry_run: bool,
    skip_existing: bool,
    statuses: dict[str, dict[str, Any]] | None = None,
    namespace: str = NAMESPACE,
) -> None:
    payload = {
        "schema": "scalef.confirmatory_runs.v1",
        "namespace": namespace,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "skip_existing": skip_existing,
        "eval_manifest": str(eval_manifest_path),
        "eval_source_sha256": eval_manifest["source_sha256"],
        "blocked_controls": list(BLOCKED_CONTROLS),
        "jobs": [
            {**asdict(job), **((statuses or {}).get(job.job_id) or {"status": "planned"})}
            for job in jobs
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_jobs(jobs: list[Job], *, skip_existing: bool) -> dict[str, dict[str, Any]]:
    work: dict[int, list[Job]] = {}
    for job in jobs:
        work.setdefault(job.gpu, []).append(job)
    statuses: dict[str, dict[str, Any]] = {}
    lock = threading.Lock()

    def worker(gpu_jobs: list[Job]) -> None:
        for job in gpu_jobs:
            metrics = Path(job.expected_metrics_path)
            if skip_existing and metrics.is_file():
                result = {"status": "skipped_existing"}
            else:
                try:
                    subprocess.run(job.command, cwd=ROOT, check=True)
                    if not metrics.is_file():
                        raise FileNotFoundError(
                            f"command succeeded but expected metrics are missing: {metrics}"
                        )
                    result = {"status": "completed"}
                except Exception as exc:  # noqa: BLE001
                    result = {"status": "failed", "error": str(exc)}
            with lock:
                statuses[job.job_id] = result

    workers = [
        threading.Thread(target=worker, args=(gpu_jobs,), daemon=True)
        for gpu_jobs in work.values()
    ]
    for thread in workers:
        thread.start()
    for thread in workers:
        thread.join()
    return statuses


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="Optional subset of family config names (e.g. joint frozen_affines).",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[6])
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--city-set", choices=tuple(CITY_SETS))
    selection.add_argument("--cities", nargs="+")
    parser.add_argument("--iters", type=int, default=5000)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--gpu-offset", type=int, default=0)
    parser.add_argument("--gpu-list", nargs="+", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume",
        "--skip-existing",
        dest="skip_existing",
        action="store_true",
        help="Skip only jobs whose confirmatory_v1 metrics.json already exists.",
    )
    parser.add_argument("--eval-manifest", type=Path, default=DEFAULT_EVAL_MANIFEST)
    parser.add_argument("--manifest-out", type=Path, default=DEFAULT_RUN_MANIFEST)
    parser.add_argument(
        "--namespace",
        type=str,
        default=NAMESPACE,
        help=(
            "Output/run-name prefix. Use a new name (e.g. confirmatory_v1_cloudmask) "
            "so a loss-masking A/B does not overwrite the frozen confirmatory_v1 B0."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    eval_path = args.eval_manifest.resolve()
    eval_manifest = load_eval_manifest(eval_path)
    jobs = generate_jobs(
        families=tuple(args.families),
        seeds=tuple(args.seeds),
        eval_manifest=eval_manifest,
        eval_manifest_path=eval_path,
        city_set=args.city_set,
        cities=tuple(args.cities) if args.cities else None,
        configs=tuple(args.configs) if args.configs else None,
        iters=args.iters,
        gpus=args.gpus,
        gpu_offset=args.gpu_offset,
        validate_data=True,
        namespace=str(args.namespace),
        gpu_list=tuple(args.gpu_list) if args.gpu_list else None,
    )
    out = args.manifest_out.resolve()
    _write_manifest(
        out,
        jobs,
        eval_manifest_path=eval_path,
        eval_manifest=eval_manifest,
        dry_run=args.dry_run,
        skip_existing=args.skip_existing,
        namespace=str(args.namespace),
    )
    print(f"Wrote {out} with {len(jobs)} new-ID jobs")
    if args.dry_run:
        for job in jobs:
            print(job.command_shell)
        return

    statuses = run_jobs(jobs, skip_existing=args.skip_existing)
    _write_manifest(
        out,
        jobs,
        eval_manifest_path=eval_path,
        eval_manifest=eval_manifest,
        dry_run=False,
        skip_existing=args.skip_existing,
        statuses=statuses,
        namespace=str(args.namespace),
    )
    failed = [job_id for job_id, status in statuses.items() if status["status"] == "failed"]
    if failed:
        raise SystemExit(f"{len(failed)} confirmatory jobs failed; see {out}")


if __name__ == "__main__":
    main()
