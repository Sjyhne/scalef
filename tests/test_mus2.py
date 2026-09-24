import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import rasterio
from rasterio.transform import from_origin

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.mus2 import align_official, balanced_score, evaluate_arrays  # noqa: E402
from s2_dataset import S2NIBRevisitDataset  # noqa: E402
from scripts.prepare_mus2 import build_manifest, prepare_manifest  # noqa: E402
from scripts.run_mus2 import _crop_to_prediction_grid, make_jobs  # noqa: E402


def _write_one_band(path: Path, image: np.ndarray, pixel_size: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=image.shape[0],
        width=image.shape[1],
        count=1,
        dtype=image.dtype,
        crs="EPSG:3857",
        transform=from_origin(0, image.shape[0] * pixel_size, pixel_size, pixel_size),
    ) as dst:
        dst.write(image, 1)


def test_official_alignment_recovers_integer_translation():
    rng = np.random.default_rng(4)
    reference = rng.uniform(0.05, 0.95, size=(24, 25))
    prediction = np.zeros_like(reference)
    prediction[1:, 2:] = reference[:-1, :-2]
    valid = np.ones_like(reference, dtype=bool)

    _, _, _, shift, cpsnr = align_official(prediction, reference, valid, max_shift=3)

    assert shift == (-1, -2)
    assert cpsnr > 250 or np.isinf(cpsnr)


def test_evaluation_honors_excluded_mask_and_balanced_score():
    reference = np.linspace(0, 255, 24 * 24, dtype=np.float64).reshape(24, 24)
    prediction = reference * 0.8 + 10.0
    excluded = np.zeros((24, 24), dtype=np.uint8)
    excluded[:6] = 255

    result = evaluate_arrays(
        prediction,
        reference,
        excluded_mask=excluded,
        max_shift=3,
    )

    assert result["shift_yx"] == [0, 0]
    assert 0.0 < result["valid_fraction"] < 1.0
    assert result["cPSNR"] > 100 or np.isinf(result["cPSNR"])
    assert balanced_score(
        {"cPSNR": 30.0, "cSSIM": 0.9},
        {"cPSNR": 20.0, "cSSIM": 0.8},
    ) < 1.0


def test_prepare_tiny_mus2_fixture_loads_in_scalef(tmp_path):
    source = tmp_path / "mus2"
    scene_id = "0_31UFT_11JUN031102-2AS_R4C1"
    scene = source / scene_id
    lr0 = np.arange(64, dtype=np.uint16).reshape(8, 8) + 1000
    lr1 = np.flipud(lr0).copy()
    hr = np.repeat(np.repeat(lr0, 3, axis=0), 3, axis=1)
    hr = np.clip(hr / 8, 1, 255).astype(np.uint8)
    _write_one_band(scene / "b2" / "lrs" / "S2A_20200101_b2.jp2", lr0, 10.0)
    _write_one_band(scene / "b2" / "lrs" / "S2B_20200111_b2.jp2", lr1, 10.0)
    _write_one_band(scene / "hr_resized" / "mul_band_1.tiff", hr, 10.0 / 3.0)
    mask = np.zeros_like(hr, dtype=np.uint8)
    _write_one_band(
        source / "masks" / "final_masks" / "final_masks_b2" / f"mask_{scene_id}.png",
        mask,
        10.0 / 3.0,
    )

    output = tmp_path / "prepared"
    manifest = build_manifest(
        source,
        output,
        bands=["b2"],
        mask_root=source,
        mask_mode="final",
    )
    assert len(manifest["scenes"]) == 1
    assert manifest["scenes"][0]["bands"]["b2"]["mask"] is not None
    prepare_manifest(manifest)

    prepared = output / scene_id / "b2"
    meta = json.loads((prepared / "meta.json").read_text())
    assert len(meta["frames"]) == 2
    assert meta["mus2_band"] == "b2"
    with rasterio.open(prepared / meta["frames"][0]["path"]) as src:
        assert (src.count, src.height, src.width) == (3, 8, 8)
    with rasterio.open(prepared / "hr_worldview2.tif") as src:
        assert (src.count, src.height, src.width) == (3, 24, 24)

    args = SimpleNamespace(
        s2_dir=str(prepared),
        hr_path=str(prepared / "hr_worldview2.tif"),
        dataset_device="cpu",
        device="cpu",
        num_samples=0,
        df=3,
        scale_factor=3,
        lr_size=0,
        no_hr_harmonize=True,
        no_hr_spatial_align=True,
        hr_gsd_m=10.0 / 3.0,
    )
    dataset = S2NIBRevisitDataset(args, name="s2")
    assert dataset.lr_rgb.shape == (2, 8, 8, 3)
    assert dataset.original_hr.shape == (24, 24, 3)

    jobs = make_jobs(
        manifest,
        run_prefix="test",
        iters=10,
        device="cpu",
        predictions_root=tmp_path / "predictions",
        extra=[],
    )
    assert len(jobs) == 1
    assert "--no_hr_harmonize" in jobs[0]["command"]
    assert jobs[0]["prediction"].endswith(f"{scene_id}/b2.tif")


def test_evaluation_crops_reference_and_mask_to_prediction_grid(tmp_path):
    reference = np.arange(12 * 15, dtype=np.uint8).reshape(12, 15)
    mask = np.zeros_like(reference)
    mask[4, 5] = 255
    reference_path = tmp_path / "reference.tif"
    prediction_path = tmp_path / "prediction.tif"
    _write_one_band(reference_path, reference, 3.0)

    expected = reference[3:9, 4:11]
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        prediction_path,
        "w",
        driver="GTiff",
        height=expected.shape[0],
        width=expected.shape[1],
        count=1,
        dtype=expected.dtype,
        crs="EPSG:3857",
        transform=from_origin(12, 27, 3.0, 3.0),
    ) as dst:
        dst.write(expected, 1)

    prediction, cropped_reference, cropped_mask, crop = _crop_to_prediction_grid(
        prediction_path,
        reference_path,
        reference,
        mask,
    )

    np.testing.assert_array_equal(prediction, expected)
    np.testing.assert_array_equal(cropped_reference, expected)
    np.testing.assert_array_equal(cropped_mask, mask[3:9, 4:11])
    np.testing.assert_array_equal(reference[crop], expected)
