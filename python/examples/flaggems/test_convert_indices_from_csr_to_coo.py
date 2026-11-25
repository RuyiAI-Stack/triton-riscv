import pytest

from ._b14_cases import run_case
from ._port_test_utils import check_case


@pytest.mark.parametrize("variant", [0, 1])
def test_convert_indices_from_csr_to_coo(monkeypatch, variant):
    check_case(monkeypatch, run_case, "convert_indices_from_csr_to_coo", variant)
