import pytest

from ._b12_cases import run_case
from ._port_test_utils import check_case


@pytest.mark.parametrize("variant", [0, 1])
def test__native_batch_norm_legit_functional(monkeypatch, variant):
    check_case(monkeypatch, run_case, "_native_batch_norm_legit_functional", variant)
