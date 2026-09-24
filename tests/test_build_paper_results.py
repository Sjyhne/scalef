import json
from pathlib import Path

import pytest

from scripts.build_paper_results import (
    ValidationError,
    build_manifest,
    latex_escape,
    render_generic,
    validate_artifact,
    validate_collection,
)


def _write_summary(path: Path, **overrides) -> Path:
    payload = {
        "created_utc": "2026-09-08T00:00:00+00:00",
        "rows": [
            {
                "run": "hash_k&4_50%",
                "lpips": 0.25,
                "psnr": 32.0,
                "ssim": 0.9,
                "training_time_s": 12.5,
            }
        ],
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload) + "\n")
    return path


def test_validation_rejects_missing_required_metric_without_allow_partial(tmp_path):
    path = _write_summary(
        tmp_path / "fixed_k_confirmatory.json",
        rows=[{"run": "k4", "lpips": 0.25}],
    )
    artifact = validate_artifact(path)

    assert artifact.partial
    assert any("training_time_s" in issue for issue in artifact.issues)
    with pytest.raises(ValidationError, match="--allow-partial"):
        validate_collection([artifact], allow_partial=False)
    validate_collection([artifact], allow_partial=True)


def test_manifest_records_source_and_canonical_sha256(tmp_path):
    root = tmp_path
    source = _write_summary(root / "accepted.json")
    artifact = validate_artifact(source)
    results = root / "paper" / "results"
    results.mkdir(parents=True)
    canonical = results / source.name
    canonical.write_bytes(source.read_bytes())

    manifest = build_manifest([artifact], [], results, root, partial=False)
    entry = manifest["files"][0]

    assert entry["source"] == "accepted.json"
    assert entry["canonical"] == "paper/results/accepted.json"
    assert entry["source_sha256"] == entry["canonical_sha256"]
    assert len(entry["source_sha256"]) == 64
    assert entry["status"] == "complete"


def test_latex_escape_and_generated_table_cells(tmp_path):
    assert latex_escape(r"A&B_50%#1") == r"A\&B\_50\%\#1"
    artifact = validate_artifact(_write_summary(tmp_path / "fixed_k_confirmatory.json"))

    table = render_generic(artifact, "fixed-k")

    assert "hash\\_k\\&4\\_50\\%" in table
    assert "AUTO-GENERATED" in table
    assert "PARTIAL RESULT" not in table
