import pytest

from ._b13_cases import run_case
from ._port_test_utils import check_case


@pytest.mark.parametrize("variant", [0, 1])
def test_index_fill(monkeypatch, variant):
    check_case(monkeypatch, run_case, "index_fill", variant)
