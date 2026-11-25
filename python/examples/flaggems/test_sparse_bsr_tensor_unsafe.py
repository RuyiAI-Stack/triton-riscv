import torch

from .sparse_bsr_tensor_unsafe import sparse_bsr_tensor_unsafe


def test_sparse_bsr_tensor_unsafe():
    dense = torch.tensor([[1.0, 0.0], [0.0, 2.0]]).to_sparse_csr()
    rebuilt = sparse_bsr_tensor_unsafe(
        dense.crow_indices(), dense.col_indices(), dense.values(), dense.shape
    )
    torch.testing.assert_close(
        rebuilt.to_dense(), dense.to_dense(), rtol=1e-4, atol=1e-4
    )
