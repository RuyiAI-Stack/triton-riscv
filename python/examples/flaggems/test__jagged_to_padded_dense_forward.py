import torch

from ._jagged_to_padded_dense_forward import _jagged_to_padded_dense_forward


def test__jagged_to_padded_dense_forward():
    values = torch.tensor([[1.0], [2.0], [3.0]])
    offsets = torch.tensor([0, 2, 3])
    out = _jagged_to_padded_dense_forward(values, offsets, 2, 0.0)
    assert out.shape[0] == 2
