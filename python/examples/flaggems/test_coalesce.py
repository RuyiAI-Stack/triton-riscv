import torch

from .coalesce import coalesce


def test_coalesce():
    indices = torch.tensor([[0, 0, 1], [1, 1, 0]])
    values = torch.tensor([1.0, 2.0, 3.0])
    tensor = torch.sparse_coo_tensor(indices, values, (2, 2))
    merged = coalesce(tensor)
    assert merged.is_coalesced() is True
    torch.testing.assert_close(
        merged.to_dense(), torch.tensor([[0.0, 3.0], [3.0, 0.0]]), rtol=1e-4, atol=1e-4
    )
