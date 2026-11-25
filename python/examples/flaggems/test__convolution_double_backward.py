import pytest

from ._b14_cases import run_case
from ._port_test_utils import check_case


@pytest.mark.parametrize("variant", [0, 1])
def test__convolution_double_backward(monkeypatch, variant):
    check_case(monkeypatch, run_case, "_convolution_double_backward", variant)
