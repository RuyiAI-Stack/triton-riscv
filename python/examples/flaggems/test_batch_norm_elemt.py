import pytest

from ._b12_cases import run_case
from ._port_test_utils import check_case


@pytest.mark.parametrize("variant", [0, 1])
def test_batch_norm_elemt(monkeypatch, variant):
    check_case(monkeypatch, run_case, "batch_norm_elemt", variant)
