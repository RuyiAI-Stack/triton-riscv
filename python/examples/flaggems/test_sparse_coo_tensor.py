import torch

from .sparse_coo_tensor import sparse_coo_tensor


def test_sparse_coo_tensor():
    indices = torch.tensor([[0, 1], [1, 0]])
    values = torch.tensor([1.0, 3.0])
    tensor = sparse_coo_tensor(indices, values, (2, 2))
    torch.testing.assert_close(
        tensor.to_dense(), torch.tensor([[0.0, 1.0], [3.0, 0.0]]), rtol=1e-4, atol=1e-4
    )
