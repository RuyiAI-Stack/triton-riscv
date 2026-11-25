import pytest

from ._b14_cases import run_case
from ._port_test_utils import check_case


@pytest.mark.parametrize("variant", [0, 1])
def test_mkldnn_reorder_conv2d_weight(monkeypatch, variant):
    check_case(monkeypatch, run_case, "mkldnn_reorder_conv2d_weight", variant)
