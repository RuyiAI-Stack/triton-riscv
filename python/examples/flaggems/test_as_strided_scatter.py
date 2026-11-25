import pytest

from ._b13_cases import run_case
from ._port_test_utils import check_case


@pytest.mark.parametrize("variant", [0, 1])
def test_as_strided_scatter(monkeypatch, variant):
    check_case(monkeypatch, run_case, "as_strided_scatter", variant)
