import torch

from .padded_dense_to_jagged_forward import padded_dense_to_jagged_forward


def test_padded_dense_to_jagged_forward():
    dense = torch.tensor([[1.0, 2.0], [3.0, 0.0]])
    offsets = torch.tensor([0, 2, 3])
    got = padded_dense_to_jagged_forward(dense, offsets)
    torch.testing.assert_close(got, torch.tensor([1.0, 2.0, 3.0]), rtol=1e-4, atol=1e-4)
