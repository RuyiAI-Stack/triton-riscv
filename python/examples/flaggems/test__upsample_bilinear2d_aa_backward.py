import pytest

from ._b13_cases import run_case
from ._port_test_utils import check_case


@pytest.mark.parametrize("variant", [0, 1])
def test__upsample_bilinear2d_aa_backward(monkeypatch, variant):
    check_case(monkeypatch, run_case, "_upsample_bilinear2d_aa_backward", variant)
