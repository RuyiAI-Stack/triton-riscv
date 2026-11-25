import pytest

from ._b14_cases import run_case
from ._port_test_utils import check_case


@pytest.mark.parametrize("variant", [0, 1])
def test_convert_indices_from_coo_to_csr(monkeypatch, variant):
    check_case(monkeypatch, run_case, "convert_indices_from_coo_to_csr", variant)
