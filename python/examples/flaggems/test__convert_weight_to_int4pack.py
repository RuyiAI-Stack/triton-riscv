import pytest

from ._b14_cases import run_case
from ._port_test_utils import check_case


@pytest.mark.parametrize("variant", [0, 1])
def test__convert_weight_to_int4pack(monkeypatch, variant):
    check_case(monkeypatch, run_case, "_convert_weight_to_int4pack", variant)
