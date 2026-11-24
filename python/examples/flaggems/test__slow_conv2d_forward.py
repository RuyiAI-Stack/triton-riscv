import pytest

from ._b14_cases import run_case
from ._port_test_utils import check_case


@pytest.mark.parametrize("variant", [0, 1])
def test__slow_conv2d_forward(monkeypatch, variant):
    check_case(monkeypatch, run_case, "_slow_conv2d_forward", variant)
