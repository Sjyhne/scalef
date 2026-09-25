import json
from pathlib import Path

import pytest

from s2_dataset import apply_frame_screen


def _frames():
    return [
        {"path": "001_a.tif", "cloud_mask": "001_a_aoi_cloud.tif"},
        {"path": "002_b.tif", "cloud_mask": "002_b_aoi_cloud.tif"},
        {"path": "003_c.tif", "cloud_mask": None},
    ]


def _screen(tmp_path: Path, entry: dict) -> Path:
    p = tmp_path / "screen.json"
    p.write_text(json.dumps({"rule": {"x": 1}, "stacks": {"site_lr512": entry}}))
    return p


def test_drops_excluded_and_swaps_masks(tmp_path):
    stack = tmp_path / "site_lr512"
    stack.mkdir()
    for stem in ("001_a", "003_c"):
        (stack / f"{stem}_lr512_ocm.tif").write_bytes(b"")
    frames = _frames()
    kept, info = apply_frame_screen(
        frames, stack, _screen(tmp_path, {"exclude": ["002_b.tif"], "cloud_mask_suffix": "_lr512_ocm"}))
    assert [f["path"] for f in kept] == ["001_a.tif", "003_c.tif"]
    assert [f["cloud_mask"] for f in kept] == ["001_a_lr512_ocm.tif", "003_c_lr512_ocm.tif"]
    assert frames[0]["cloud_mask"] == "001_a_aoi_cloud.tif"
    assert info["excluded"] == ["002_b.tif"] and info["n_kept"] == 2


def test_missing_stack_or_mask_is_an_error(tmp_path):
    stack = tmp_path / "other_lr512"
    stack.mkdir()
    with pytest.raises(ValueError):
        apply_frame_screen(_frames(), stack, _screen(tmp_path, {"exclude": []}))
    stack = tmp_path / "site_lr512"
    stack.mkdir()
    with pytest.raises(FileNotFoundError):
        apply_frame_screen(_frames(), stack, _screen(tmp_path, {"exclude": [], "cloud_mask_suffix": "_lr512_ocm"}))
