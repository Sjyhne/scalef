import json
from argparse import Namespace
from datetime import date
from pathlib import Path

from scripts.plan_cross_mgrs_identity import (
    build_audit,
    build_cells,
    find_cross_edges,
    load_granules,
    revise_assignments,
    run,
)


def _write_inputs(
    root: Path,
    mgrs: str,
    *,
    x_origin: float,
    assigned: str,
    date_clouds: dict[str, float | None],
) -> dict:
    directory = root / mgrs
    directory.mkdir()
    tile_id = f"{mgrs}_t10_y00_x00"
    plan = {
        "parent": mgrs,
        "center": "2025-07-15",
        "assignment": {tile_id: assigned},
        "independent": {tile_id: assigned},
        "cells": [
            {
                "tile_id": tile_id,
                "iy": 0,
                "ix": 0,
                "icm_date": assigned,
                "independent_date": assigned,
                "date_clouds": date_clouds,
            }
        ],
    }
    manifest = {
        "parent": mgrs,
        "side": 10,
        "tiles": [
            {
                "tile_id": tile_id,
                "iy": 0,
                "ix": 0,
                "row_off": 0,
                "col_off": 0,
                "side": 10,
            }
        ],
    }
    meta = {
        "mgrs_tile": mgrs,
        "crs": "EPSG:32632",
        "transform": [1, 0, x_origin, 0, -1, 100],
        "frames": [],
    }
    paths = {}
    for name, payload in (
        ("identity_plan", plan),
        ("granule_manifest", manifest),
        ("s2_meta", meta),
    ):
        path = directory / f"{name}.json"
        path.write_text(json.dumps(payload))
        paths[name] = str(path)
    return {"mgrs": mgrs, **paths}


def _plan(tmp_path, left_clouds, right_clouds):
    specs = [
        _write_inputs(
            tmp_path,
            "32VAA",
            x_origin=500000,
            assigned="2025-07-12",
            date_clouds=left_clouds,
        ),
        _write_inputs(
            tmp_path,
            "32VAB",
            x_origin=500010,
            assigned="2025-07-18",
            date_clouds=right_clouds,
        ),
    ]
    granules = load_granules(specs)
    cells = build_cells(granules, "EPSG:32632")
    edges = find_cross_edges(cells, 0.01)
    revised = revise_assignments(
        cells,
        edges,
        slack=0.05,
        center=date(2025, 7, 15),
    )
    return granules, cells, edges, revised


def test_adjacent_cells_share_slack_valid_date(tmp_path):
    granules, cells, edges, revised = _plan(
        tmp_path,
        {"2025-07-12": 0.01, "2025-07-18": 0.03},
        {"2025-07-12": 0.04, "2025-07-18": 0.01},
    )
    assert len(edges) == 1
    assert set(revised.values()) == {"2025-07-18"}

    audit = build_audit(
        granules,
        cells,
        edges,
        revised,
        common_crs="EPSG:32632",
        tolerance_m=0.01,
        slack=0.05,
        center=date(2025, 7, 15),
        audit_only=False,
    )
    assert audit["summary"]["n_boundary_cuts_before"] == 1
    assert audit["summary"]["n_boundary_cuts_after"] == 0
    assert audit["boundaries"][0]["resolved"] is True


def test_never_forces_date_missing_from_other_cell(tmp_path):
    _granules, cells, edges, revised = _plan(
        tmp_path,
        {"2025-07-12": 0.01},
        {"2025-07-18": 0.01},
    )
    assert len(edges) == 1
    assert revised == {key: cell.original_date for key, cell in cells.items()}


def test_shared_date_must_pass_slack_in_both_cells(tmp_path):
    _granules, cells, edges, revised = _plan(
        tmp_path,
        {"2025-07-12": 0.01, "2025-07-18": 0.02},
        {"2025-07-12": 0.20, "2025-07-18": 0.20},
    )
    assert revised == {key: cell.original_date for key, cell in cells.items()}


def test_audit_only_writes_no_revised_plans(tmp_path):
    left = _write_inputs(
        tmp_path,
        "32VAA",
        x_origin=500000,
        assigned="2025-07-12",
        date_clouds={"2025-07-12": 0.01},
    )
    right = _write_inputs(
        tmp_path,
        "32VAB",
        x_origin=500010,
        assigned="2025-07-18",
        date_clouds={"2025-07-18": 0.01},
    )
    input_manifest = tmp_path / "inputs.json"
    input_manifest.write_text(json.dumps({"granules": [left, right]}))
    report = tmp_path / "audit.json"
    args = Namespace(
        granule=None,
        input=None,
        inputs_manifest=input_manifest,
        common_crs="EPSG:32632",
        adjacency_tolerance_m=0.01,
        slack_cloud=0.05,
        center_date="2025-07-15",
        out_dir=tmp_path / "must_not_exist",
        audit_only=True,
        report=report,
    )
    run(args)
    assert report.is_file()
    assert not args.out_dir.exists()


def test_georef_recovers_from_tile_s2_metadata(tmp_path):
    specs = []
    for mgrs, origin, assigned in (
        ("32VAA", 500000, "2025-07-12"),
        ("32VAB", 500010, "2025-07-18"),
    ):
        spec = _write_inputs(
            tmp_path,
            mgrs,
            x_origin=origin,
            assigned=assigned,
            date_clouds={assigned: 0.01},
        )
        tile_dir = tmp_path / mgrs / "tile"
        tile_dir.mkdir()
        meta_path = Path(spec["s2_meta"])
        source_meta = json.loads(meta_path.read_text())
        (tile_dir / "meta.json").write_text(json.dumps(source_meta))
        meta_path.write_text(json.dumps({"mgrs_tile": mgrs, "frames": []}))
        manifest_path = Path(spec["granule_manifest"])
        manifest = json.loads(manifest_path.read_text())
        manifest["tiles"][0]["s2_dir"] = str(tile_dir)
        manifest_path.write_text(json.dumps(manifest))
        specs.append(spec)

    cells = build_cells(load_granules(specs), "EPSG:32632")
    assert len(find_cross_edges(cells, 0.01)) == 1


def test_writes_provenance_rich_revised_plans_without_touching_sources(tmp_path):
    specs = [
        _write_inputs(
            tmp_path,
            "32VAA",
            x_origin=500000,
            assigned="2025-07-12",
            date_clouds={"2025-07-12": 0.01, "2025-07-18": 0.03},
        ),
        _write_inputs(
            tmp_path,
            "32VAB",
            x_origin=500010,
            assigned="2025-07-18",
            date_clouds={"2025-07-12": 0.04, "2025-07-18": 0.01},
        ),
    ]
    inputs_path = tmp_path / "inputs.json"
    inputs_path.write_text(json.dumps({"granules": specs}))
    source_before = Path(specs[0]["identity_plan"]).read_bytes()
    out_dir = tmp_path / "revised"
    run(
        Namespace(
            granule=None,
            input=None,
            inputs_manifest=inputs_path,
            common_crs="EPSG:32632",
            adjacency_tolerance_m=0.01,
            slack_cloud=0.05,
            center_date="2025-07-15",
            out_dir=out_dir,
            audit_only=False,
            report=None,
        )
    )

    revised = json.loads((out_dir / "32VAA" / "identity_plan.json").read_text())
    assert set(revised["assignment"].values()) == {"2025-07-18"}
    assert revised["cross_mgrs_identity"]["unavailable_dates_forced"] is False
    assert (out_dir / "cross_mgrs_identity_audit.json").is_file()
    assert Path(specs[0]["identity_plan"]).read_bytes() == source_before
