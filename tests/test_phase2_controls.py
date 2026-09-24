import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_phase2_controls import analyze_run, build_report
from scripts.export_superf_srdata import SOURCE_COMMIT, export_arrays
from scripts.materialize_s2_controls import materialize_control
from scripts.run_phase2_controls import build_commands


def _source_stack(root: Path) -> Path:
    root.mkdir()
    frames = []
    for i, day in enumerate(("20240101", "20240109", "20240120"), start=1):
        name = f"{i:03d}_{day}.tif"
        mask = f"{i:03d}_{day}_aoi_cloud.tif"
        (root / name).write_bytes(f"frame-{i}".encode())
        (root / mask).write_bytes(f"mask-{i}".encode())
        frames.append(
            {
                "index": i,
                "path": name,
                "cloud_mask": mask,
                "datetime": f"{day[:4]}-{day[4:6]}-{day[6:]}T10:00:00+00:00",
            }
        )
    (root / "meta.json").write_text(
        json.dumps(
            {
                "center_date": "2024-01-10",
                "frames": frames,
                "transform": [10, 0, 100, 0, -10, 200],
                "crs": "EPSG:32632",
                "aoi_window": {"col_off": 0, "row_off": 0, "width": 2, "height": 2},
            }
        )
    )
    return root


def test_materialize_repeated_base_preserves_metadata_and_links(tmp_path):
    source = _source_stack(tmp_path / "city")
    output = tmp_path / "repeat"
    meta = materialize_control(source, output, repeats=3)
    assert meta["control"]["base_frame_original_index"] == 1
    assert meta["control"]["base_frame_original_path"] == "002_20240109.tif"
    assert len(meta["frames"]) == 3
    assert meta["transform"] == [10, 0, 100, 0, -10, 200]
    assert all((output / frame["path"]).is_symlink() for frame in meta["frames"])
    assert [(output / frame["path"]).read_bytes() for frame in meta["frames"]] == [
        b"frame-2"
    ] * 3


def _write_run(path: Path, shift: float, psnr: float) -> None:
    path.mkdir()
    affine = {
        "lr_width": 100,
        "lr_height": 80,
        "frames": [
            {"frozen": True, "matrix_2x3": [[1, 0, 0], [0, 1, 0]]},
            {"frozen": False, "matrix_2x3": [[1, 0, shift / 100], [0, 1, 0.5 / 80]]},
        ],
    }
    (path / "affines.json").write_text(json.dumps(affine))
    (path / "metrics.json").write_text(
        json.dumps(
            {
                "psnr": {"model": psnr, "improvement": psnr - 30},
                "ssim": {"model": 0.9, "improvement": 0.01},
                "lpips": {"model": 0.2, "improvement": 0.02},
                "training": {"final_trans_loss": 0.001},
            }
        )
    )


def test_analysis_reports_phase_shift_proxies_and_correlations(tmp_path):
    run1, run2 = tmp_path / "run1", tmp_path / "run2"
    _write_run(run1, 0.25, 31.0)
    _write_run(run2, 0.75, 32.0)
    summary = analyze_run(run1)
    assert summary["frame_count"] == 2
    assert summary["frozen_frame_count"] == 1
    assert summary["phase_coverage"]["occupied_bins"] == 2
    report = build_report([run1, run2])
    correlation = report["correlations"]["shift_std__vs__psnr"]
    assert correlation["n"] == 2
    assert np.isclose(correlation["pearson_r"], 1.0)


def test_export_arrays_matches_srdata_layout_and_provenance(tmp_path):
    lr = np.stack(
        [np.zeros((3, 4, 3), dtype=np.float32), np.ones((3, 4, 3), dtype=np.float32)]
    )
    hr = np.full((6, 8, 3), 0.5, dtype=np.float32)
    payload = export_arrays(lr, hr, tmp_path, frame_sources=["a.tif", "b.tif"])
    assert payload["source_commit"] == SOURCE_COMMIT
    assert Image.open(tmp_path / "sample_00.png").size == (4, 3)
    assert Image.open(tmp_path / "hr_ground_truth.png").size == (8, 6)
    log = json.loads((tmp_path / "transform_log.json").read_text())
    assert list(log) == ["sample_00", "sample_01"]
    assert log["sample_01"]["source_file"] == "b.tif"


def test_control_driver_generates_supported_runs_and_frozen_blocker(tmp_path):
    manifest = build_commands(
        city="demo",
        source_dir=tmp_path / "full",
        single_dir=tmp_path / "single",
        repeated_dir=tmp_path / "repeat",
        repeats=4,
        seeds=[3],
        device=0,
        iters=10,
    )
    assert len(manifest["commands"]) == 3
    assert manifest["frozen_alignment_control"]["status"] == "blocked"
    repeated = next(x for x in manifest["commands"] if x["variant"].startswith("repeated"))
    assert repeated["command"][repeated["command"].index("--num_samples") + 1] == "4"
