import pytest

from ._b14_cases import run_case
from ._port_test_utils import check_case


@pytest.mark.parametrize("variant", [0, 1])
def test_fractional_max_pool2d(monkeypatch, variant):
    check_case(monkeypatch, run_case, "fractional_max_pool2d", variant)
