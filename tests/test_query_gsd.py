import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.query_gsd import parse_query_gsd_m, query_grid_hw  # noqa: E402


def test_parse_query_gsd_m():
    assert parse_query_gsd_m(None) == []
    assert parse_query_gsd_m("") == []
    assert parse_query_gsd_m("5,1") == [5.0, 1.0]
    assert parse_query_gsd_m(" 1 ") == [1.0]


def test_query_grid_hw_integer_scales():
    assert query_grid_hw(512, 512, 10.0, 2.5) == (2048, 2048, 4)
    assert query_grid_hw(512, 512, 10.0, 5.0) == (1024, 1024, 2)
    assert query_grid_hw(512, 512, 10.0, 1.0) == (5120, 5120, 10)
