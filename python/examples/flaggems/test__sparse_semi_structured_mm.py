import pytest
import torch

from ._sparse_semi_structured_mm import (
    _sparse_semi_structured_mm,
)


def _torch_sparse_semi_structured_mm(mat1, meta, mat2):
    selected = torch.stack((meta, meta, ~meta, ~meta), dim=-1).reshape_as(mat1)
    return torch.matmul(torch.where(selected, mat1, torch.zeros_like(mat1)), mat2)


@pytest.mark.parametrize("m, k4, n", [(2, 2, 3), (3, 35, 7), (33, 35, 33)])
@pytest.mark.parametrize("meta_value", [False, True, None])
def test_sparse_semi_structured_mm_matches_torch(m, k4, n, meta_value):
    torch.manual_seed(0)
    mat1 = torch.randn(m, 4 * k4, dtype=torch.float32, device="cpu")
    if meta_value is None:
        meta = torch.randint(0, 2, (m, k4), dtype=torch.bool, device="cpu")
    else:
        meta = torch.full((m, k4), meta_value, dtype=torch.bool, device="cpu")
    mat2 = torch.randn(4 * k4, n, dtype=torch.float32, device="cpu")

    out = _sparse_semi_structured_mm(mat1, meta, mat2)
    ref = _torch_sparse_semi_structured_mm(mat1, meta, mat2)

    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


def test_sparse_semi_structured_mm_out_dtype():
    torch.manual_seed(0)
    mat1 = torch.randn(3, 140, dtype=torch.float32, device="cpu")
    meta = torch.randint(0, 2, (3, 35), dtype=torch.bool, device="cpu")
    mat2 = torch.randn(140, 7, dtype=torch.float32, device="cpu")

    out = _sparse_semi_structured_mm(mat1, meta, mat2, out_dtype=torch.float64)
    ref = _torch_sparse_semi_structured_mm(mat1, meta, mat2).to(torch.float64)

    assert out.dtype is torch.float64
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
