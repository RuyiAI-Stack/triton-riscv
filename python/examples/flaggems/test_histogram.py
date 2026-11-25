import pytest

from ._b12_cases import run_case
from ._port_test_utils import check_case


@pytest.mark.parametrize("variant", [0, 1])
def test_histogram(monkeypatch, variant):
    check_case(monkeypatch, run_case, "histogram", variant)
