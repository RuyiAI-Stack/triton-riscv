import pytest

from ._b13_cases import run_case
from ._port_test_utils import check_case


@pytest.mark.parametrize("variant", [0, 1])
def test_grid_sampler_3d_backward(monkeypatch, variant):
    check_case(monkeypatch, run_case, "grid_sampler_3d_backward", variant)
