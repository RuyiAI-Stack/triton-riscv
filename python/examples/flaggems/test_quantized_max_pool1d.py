import pytest

from ._b14_cases import run_case
from ._port_test_utils import check_case


@pytest.mark.parametrize("variant", [0, 1])
def test_quantized_max_pool1d(monkeypatch, variant):
    check_case(monkeypatch, run_case, "quantized_max_pool1d", variant)
