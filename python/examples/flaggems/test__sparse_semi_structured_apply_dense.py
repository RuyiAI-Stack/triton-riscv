import torch

from ._sparse_semi_structured_apply_dense import _sparse_semi_structured_apply_dense


def test__sparse_semi_structured_apply_dense():
    torch.manual_seed(0)
    dense = torch.randn(2, 4)
    mask = dense.abs() > 0
    torch.testing.assert_close(
        _sparse_semi_structured_apply_dense(dense, mask),
        dense * mask,
        rtol=1e-4,
        atol=1e-4,
    )
