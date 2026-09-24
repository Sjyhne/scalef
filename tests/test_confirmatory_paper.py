import json
from pathlib import Path

import pytest

from scripts.freeze_confirmatory_eval import build_frozen_manifest, validate_frozen_manifest
from scripts.run_confirmatory_paper import (
    CITY_SETS,
    NAMESPACE,
    family_configs,
    generate_jobs,
)


def _eval_manifest() -> dict:
    return {
        "schema": "scalef.confirmatory_eval.v1",
        "source_sha256": "test",
        "cities": {
            city: {"s2_dir_name": f"{city}_lr512"}
            for city in CITY_SETS["b0_17"]
        },
    }


def test_family_config_matrix_is_canonical():
    assert [c.config for c in family_configs("fixed_k")] == [
        "lr128_k1",
        "lr128_k2",
        "lr128_k4",
        "lr128_k8",
        "lr128_full",
    ]
    assert [c.config for c in family_configs("gsd_ladder")] == ["df2_5m", "df4_query"]
    encoding = family_configs("encoding_size")
    assert len(encoding) == 16
    assert {c.lr_size for c in encoding} == {64, 128, 256, 512}
    assert {c.config for c in encoding if "fourier" in c.config} == {
        f"lr{size}_fourier_s{scale}"
        for size in (64, 128, 256, 512)
        for scale in (2, 5, 10)
    }


def test_job_generation_uses_new_namespace_and_exact_paths(tmp_path: Path):
    jobs = generate_jobs(
        families=("fixed_k",),
        seeds=(6, 9),
        eval_manifest=_eval_manifest(),
        root=tmp_path,
        cities=("asker",),
        gpus=2,
        gpu_offset=3,
        validate_data=False,
    )
    assert len(jobs) == 10
    assert {job.gpu for job in jobs} == {3, 4}
    for job in jobs:
        assert job.job_id == job.run_name
        assert job.run_name.startswith(f"{NAMESPACE}__fixed_k__")
        assert f"__{job.city}__seed{job.seed}" in job.run_name
        assert f"/{NAMESPACE}/{job.run_name}/metrics.json" in job.expected_metrics_path
        assert "--sample_id" in job.command
        assert job.command[job.command.index("--sample_id") + 1] == NAMESPACE
        assert job.command_shell


def test_all_default_jobs_have_unique_ids(tmp_path: Path):
    jobs = generate_jobs(
        families=("fixed_k", "encoding_size", "misr_controls", "mtf_sigma", "b0_17"),
        seeds=(6, 7),
        eval_manifest=_eval_manifest(),
        root=tmp_path,
        gpus=4,
        validate_data=False,
    )
    ids = [job.job_id for job in jobs]
    assert len(ids) == len(set(ids))
    assert len(jobs) == 2 * (5 * 7 + 16 * 7 + 5 * 7 + 3 * 7 + 17)


def test_missing_data_fails_instead_of_dropping_job(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="missing requested data"):
        generate_jobs(
            families=("b0_17",),
            seeds=(6,),
            eval_manifest=_eval_manifest(),
            root=tmp_path,
            cities=("asker",),
            validate_data=True,
        )


def test_misr_freeze_flags_are_boolean_and_filterable(tmp_path: Path):
    jobs = generate_jobs(
        families=("misr_controls",),
        seeds=(6,),
        eval_manifest=_eval_manifest(),
        root=tmp_path,
        cities=("asker",),
        configs=("joint", "frozen_affines", "frozen_radiometry"),
        validate_data=False,
    )
    assert {job.config for job in jobs} == {"joint", "frozen_affines", "frozen_radiometry"}
    by_config = {job.config: job.command for job in jobs}
    aff_cmd = by_config["frozen_affines"]
    rad_cmd = by_config["frozen_radiometry"]
    aff_idx = aff_cmd.index("--freeze_affines")
    rad_idx = rad_cmd.index("--freeze_radiometry")
    assert aff_idx == len(aff_cmd) - 1 or aff_cmd[aff_idx + 1].startswith("--")
    assert rad_idx == len(rad_cmd) - 1 or rad_cmd[rad_idx + 1].startswith("--")
    assert "--freeze_affines" not in by_config["joint"]
    assert "--freeze_radiometry" not in by_config["joint"]
    assert "--freeze_radiometry" not in by_config["frozen_affines"]
    with pytest.raises(ValueError, match="no configs"):
        generate_jobs(
            families=("misr_controls",),
            seeds=(6,),
            eval_manifest=_eval_manifest(),
            root=tmp_path,
            cities=("asker",),
            configs=("not_a_config",),
            validate_data=False,
        )


def test_short_frame_stack_fails_instead_of_reducing_sample_count(tmp_path: Path):
    stack = tmp_path / "data" / "s2_revisits" / "asker_lr512"
    stack.mkdir(parents=True)
    (stack / "meta.json").write_text(json.dumps({"frames": [{"path": "one.tif"}]}))
    with pytest.raises(ValueError, match="config requires 16"):
        generate_jobs(
            families=("b0_17",),
            seeds=(6,),
            eval_manifest=_eval_manifest(),
            root=tmp_path,
            cities=("asker",),
            validate_data=True,
        )


def test_frozen_eval_manifest_is_minimal_and_deterministic(tmp_path: Path):
    source = tmp_path / "spatial_alignment.json"
    city = {
        "df": 4,
        "s2_dir_name": "asker_lr512",
        "lr_size": [512, 512],
        "hr_size": [2048, 2048],
        "base_frame_index": 0,
        "base_frame_date": "2024-05-14",
        "nib_acquisition_date": "2024-05-15",
        "hr_shift_hr_px": {"dy": -0.8, "dx": -0.3},
        "exploratory_metric": 123,
    }
    source.write_text(
        json.dumps({"version": 1, "method": "alignment", "cities": {"asker": city}})
    )
    first = build_frozen_manifest(source)
    second = build_frozen_manifest(source)
    assert first == second
    assert first["cities"]["asker"]["hr_shift_hr_px"] == {"dy": -0.8, "dx": -0.3}
    assert "exploratory_metric" not in first["cities"]["asker"]

    frozen = tmp_path / "confirmatory_eval.json"
    frozen.write_text(json.dumps(first))
    assert validate_frozen_manifest(frozen, source) == first
    source.write_text(source.read_text().replace('"alignment"', '"revised_alignment"'))
    with pytest.raises(ValueError, match="stale or edited"):
        validate_frozen_manifest(frozen, source)
