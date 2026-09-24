import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import run_production


def test_production_summary_propagates_process_memory(tmp_path, monkeypatch):
    monkeypatch.setattr(run_production, "ROOT", tmp_path)
    metrics = (
        tmp_path
        / "single_samples"
        / "32VNM"
        / "sample"
        / "prod_test_tile"
        / "metrics.json"
    )
    metrics.parent.mkdir(parents=True)
    metrics.write_text(
        json.dumps(
            {
                "training_time_seconds": 12.0,
                "peak_memory_gb": 0.8,
                "gpu_memory": {
                    "process_peak_used_gb": 5.1,
                    "torch_peak_reserved_gb": 1.0,
                    "process_memory_source": "pynvml",
                },
            }
        )
    )
    tile = {"parent": "32VNM", "tile_id": "tile", "s2_dir": "stack"}

    row = run_production._summarize(
        tile,
        metrics,
        skipped=False,
        run_prefix="prod_test",
    )

    assert row["process_peak_gpu_memory_gb"] == 5.1
    assert row["torch_peak_reserved_gpu_memory_gb"] == 1.0
    assert row["gpu_memory_source"] == "pynvml"


def test_production_aggregate_reports_process_memory():
    aggregate = run_production._aggregate(
        [
            {
                "training_time_s": 10.0,
                "process_peak_gpu_memory_gb": 5.0,
                "sr_pred_tif": "a.tif",
                "skipped": False,
            },
            {
                "training_time_s": 20.0,
                "process_peak_gpu_memory_gb": 7.0,
                "sr_pred_tif": "b.tif",
                "skipped": True,
            },
        ]
    )
    assert aggregate["mean_process_peak_gpu_memory_gb"] == 6.0
    assert aggregate["max_process_peak_gpu_memory_gb"] == 7.0
    assert aggregate["n_with_process_peak_gpu_memory"] == 2


def test_existing_run_requires_matching_identity_date_and_geotiff(tmp_path, monkeypatch):
    monkeypatch.setattr(run_production, "ROOT", tmp_path)
    tile = {
        "parent": "32VNM",
        "tile_id": "tile",
        "s2_dir": "stack",
        "force_base_date": "2025-07-12",
    }
    metrics = run_production._metrics_path("32VNM", "tile", "prod_test")
    metrics.parent.mkdir(parents=True)
    metrics.write_text(json.dumps({"args": {"force_base_date": "2025-07-18"}}))
    geotiff = run_production._sr_geotiff_path("32VNM", "tile", "prod_test")
    geotiff.parent.mkdir(parents=True)
    geotiff.write_bytes(b"tif")

    assert not run_production._existing_run_matches(
        tile,
        run_prefix="prod_test",
        export_geotiff=True,
        force_base_date=None,
    )

    metrics.write_text(json.dumps({"args": {"force_base_date": "2025-07-12"}}))
    assert run_production._existing_run_matches(
        tile,
        run_prefix="prod_test",
        export_geotiff=True,
        force_base_date=None,
    )

    geotiff.unlink()
    assert not run_production._existing_run_matches(
        tile,
        run_prefix="prod_test",
        export_geotiff=True,
        force_base_date=None,
    )
