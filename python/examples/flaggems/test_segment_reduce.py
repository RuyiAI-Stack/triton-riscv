import pytest

from ._b12_cases import run_case
from ._port_test_utils import check_case


@pytest.mark.parametrize("variant", [0, 1])
def test_segment_reduce(monkeypatch, variant):
    check_case(monkeypatch, run_case, "segment_reduce", variant)
