import pytest

from ._b14_cases import run_case
from ._port_test_utils import check_case


@pytest.mark.parametrize("variant", [0, 1])
def test_adaptive_max_pool2d_backward(monkeypatch, variant):
    check_case(monkeypatch, run_case, "adaptive_max_pool2d_backward", variant)
