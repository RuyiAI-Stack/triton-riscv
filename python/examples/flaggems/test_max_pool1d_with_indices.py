import pytest

from ._b14_cases import run_case
from ._port_test_utils import check_case


@pytest.mark.parametrize("variant", [0, 1])
def test_max_pool1d_with_indices(monkeypatch, variant):
    check_case(monkeypatch, run_case, "max_pool1d_with_indices", variant)
