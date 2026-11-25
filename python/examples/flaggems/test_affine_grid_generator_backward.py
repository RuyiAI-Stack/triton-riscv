import pytest

from ._b13_cases import run_case
from ._port_test_utils import check_case


@pytest.mark.parametrize("variant", [0, 1])
def test_affine_grid_generator_backward(monkeypatch, variant):
    check_case(monkeypatch, run_case, "affine_grid_generator_backward", variant)
