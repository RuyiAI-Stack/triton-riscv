import pytest
import torch

from ._b12_cases import run_case
from ._port_test_utils import check_case
from .linalg_norm import linalg_norm


@pytest.mark.parametrize("variant", [0, 1])
def test_linalg_norm(monkeypatch, variant):
    check_case(monkeypatch, run_case, "linalg_norm", variant)


@pytest.mark.parametrize("ord", ["fro", 1])
@pytest.mark.parametrize("keepdim", [False, True])
def test_linalg_norm_matrix_dimensions(monkeypatch, ord, keepdim):
    # Non-trailing matrix dimensions must keep using the B12 Triton path when
    # the separately maintained public matrix-norm module is also installed.
    x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) - 9

    def matrix_case(name, variant):
        kwargs = dict(ord=ord, dim=(0, 2), keepdim=keepdim, dtype=torch.float64)
        return linalg_norm(x, **kwargs), torch.linalg.norm(x, **kwargs)

    check_case(monkeypatch, matrix_case, "linalg_norm_matrix_dimensions", 0)
