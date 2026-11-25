import pytest

from ._b13_cases import run_case
from ._port_test_utils import check_case


@pytest.mark.parametrize("variant", [0, 1])
def test__upsample_nearest_exact3d(monkeypatch, variant):
    check_case(monkeypatch, run_case, "_upsample_nearest_exact3d", variant)
