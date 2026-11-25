import pytest

from ._b12_cases import run_case
from ._port_test_utils import check_case


@pytest.mark.parametrize("variant", [0, 1])
def test__histogramdd_from_bin_cts(monkeypatch, variant):
    check_case(monkeypatch, run_case, "_histogramdd_from_bin_cts", variant)
