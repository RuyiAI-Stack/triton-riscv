import torch

from .sparse_compressed_tensor import sparse_compressed_tensor


def test_sparse_compressed_tensor():
    dense = torch.tensor([[0.0, 1.0], [2.0, 0.0]]).to_sparse_csr()
    rebuilt = sparse_compressed_tensor(
        dense.crow_indices(), dense.col_indices(), dense.values(), (2, 2)
    )
    torch.testing.assert_close(
        rebuilt.to_dense(), dense.to_dense(), rtol=1e-4, atol=1e-4
    )
