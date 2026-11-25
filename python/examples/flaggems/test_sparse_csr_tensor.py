import torch

from .sparse_csr_tensor import sparse_csr_tensor


def test_sparse_csr_tensor():
    dense = torch.tensor([[0.0, 1.0], [2.0, 0.0]])
    csr = dense.to_sparse_csr()
    rebuilt = sparse_csr_tensor(
        csr.crow_indices(), csr.col_indices(), csr.values(), dense.shape
    )
    torch.testing.assert_close(rebuilt.to_dense(), dense, rtol=1e-4, atol=1e-4)
