import torch

from .sparse_csc_tensor import sparse_csc_tensor


def test_sparse_csc_tensor():
    dense = torch.tensor([[0.0, 1.0], [2.0, 0.0]])
    csc = dense.to_sparse_csc()
    rebuilt = sparse_csc_tensor(
        csc.ccol_indices(), csc.row_indices(), csc.values(), dense.shape
    )
    torch.testing.assert_close(rebuilt.to_dense(), dense, rtol=1e-4, atol=1e-4)
