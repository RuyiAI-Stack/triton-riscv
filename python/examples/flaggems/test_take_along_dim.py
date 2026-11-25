import pytest

from ._b13_cases import run_case
from ._port_test_utils import check_case


@pytest.mark.parametrize("variant", [0, 1])
def test_take_along_dim(monkeypatch, variant):
    check_case(monkeypatch, run_case, "take_along_dim", variant)
