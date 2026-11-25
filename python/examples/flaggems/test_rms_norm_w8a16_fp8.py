import pytest

from ._b12_cases import run_case
from ._port_test_utils import check_case


@pytest.mark.parametrize("variant", [0, 1])
def test_rms_norm_w8a16_fp8(monkeypatch, variant):
    check_case(monkeypatch, run_case, "rms_norm_w8a16_fp8", variant)
