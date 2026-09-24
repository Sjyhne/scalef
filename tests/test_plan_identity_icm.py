from datetime import date

import json

from scripts.plan_identity_icm import (
    assign_icm,
    independent_date,
    write_identity_manifests,
)


def test_independent_prefers_clear_closer_day():
    clouds = {"2025-07-15": 0.12, "2025-07-12": 0.0, "2025-08-12": 0.0}
    assert (
        independent_date(clouds, center=date(2025, 7, 15), max_cloud=0.02)
        == "2025-07-12"
    )


def test_icm_joins_primary_at_slack():
    # Three cells in a row. Middle has July 12 at 4% (fails 2%, passes 5%).
    date_clouds = {
        (0, 0): {"2025-07-12": 0.0, "2025-08-12": 0.10},
        (0, 1): {"2025-07-12": 0.04, "2025-08-12": 0.0},
        (0, 2): {"2025-07-12": 0.0},
    }
    plan = assign_icm(
        date_clouds, center=date(2025, 7, 15), hard_cloud=0.02, slack_cloud=0.05
    )
    assert plan["primary"] == "2025-07-12"
    assert plan["independent"]["32VNM_t512_y00_x01"] == "2025-08-12"
    assert plan["assignment"]["32VNM_t512_y00_x01"] == "2025-07-12"
    assert plan["n_joined_primary"] == 1
    assert plan["n_leftover"] == 0


def test_icm_does_not_force_missing_primary():
    date_clouds = {
        (0, 0): {"2025-07-12": 0.0},
        (0, 1): {"2025-07-18": 0.0},  # no July 12 in stack
    }
    plan = assign_icm(
        date_clouds, center=date(2025, 7, 15), hard_cloud=0.02, slack_cloud=0.05
    )
    assert plan["assignment"]["32VNM_t512_y00_x01"] == "2025-07-18"
    assert plan["n_leftover"] == 1


def test_boundary_polish_can_move_a_primary_cell_to_local_region():
    date_clouds = {
        (0, 0): {"2025-07-12": 0.0},
        (0, 1): {"2025-07-12": 0.0, "2025-08-12": 0.01},
        (0, 2): {"2025-08-12": 0.0},
        (1, 1): {"2025-08-12": 0.0},
    }

    plan = assign_icm(
        date_clouds,
        center=date(2025, 7, 15),
        hard_cloud=0.02,
        slack_cloud=0.05,
    )

    assert plan["assignment"]["32VNM_t512_y00_x01"] == "2025-08-12"
    assert plan["n_boundary_polished"] >= 1


def test_icm_uses_requested_parent_in_tile_ids():
    plan = assign_icm(
        {(0, 0): {"2025-07-12": 0.0}},
        center=date(2025, 7, 15),
        hard_cloud=0.02,
        slack_cloud=0.05,
        parent="33WXS",
    )
    assert set(plan["assignment"]) == {"33WXS_t512_y00_x00"}


def test_icm_preserves_overlap_tile_ids_and_geometry():
    tile_id = "32VNM_t512_ovl12_y01_x02"
    plan = assign_icm(
        {(1, 2): {"2025-07-12": 0.0}},
        center=date(2025, 7, 15),
        hard_cloud=0.02,
        slack_cloud=0.05,
        tile_ids={(1, 2): tile_id},
        geometry={(1, 2): {"row_off": 448, "col_off": 896, "stride": 448}},
    )

    assert plan["assignment"] == {tile_id: "2025-07-12"}
    assert plan["cells"][0]["row_off"] == 448
    assert plan["cells"][0]["stride"] == 448


def test_write_identity_manifests_keeps_only_changed_subset(tmp_path):
    plan = assign_icm(
        {
            (0, 0): {"2025-07-12": 0.0},
            (0, 1): {"2025-07-12": 0.04, "2025-08-12": 0.0},
        },
        center=date(2025, 7, 15),
        hard_cloud=0.02,
        slack_cloud=0.05,
    )
    source = {
        "parent": "32VNM",
        "tiles": [
            {
                "tile_id": "32VNM_t512_y00_x00",
                "iy": 0,
                "ix": 0,
                "s2_dir": "a",
            },
            {
                "tile_id": "32VNM_t512_y00_x01",
                "iy": 0,
                "ix": 1,
                "s2_dir": "b",
            },
        ],
    }
    source_path = tmp_path / "granule.json"
    source_path.write_text(json.dumps(source))

    all_path, changed_path = write_identity_manifests(
        plan,
        granule_manifest_path=source_path,
        out_dir=tmp_path / "out",
    )

    all_tiles = json.loads(all_path.read_text())["tiles"]
    changed_tiles = json.loads(changed_path.read_text())["tiles"]
    assert len(all_tiles) == 2
    assert len(changed_tiles) == 1
    assert changed_tiles[0]["tile_id"] == "32VNM_t512_y00_x01"
    assert changed_tiles[0]["force_base_date"] == "2025-07-12"
