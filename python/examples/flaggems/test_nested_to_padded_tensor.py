import torch

from .nested_to_padded_tensor import nested_to_padded_tensor


def test_nested_to_padded_tensor():
    src = torch.ones(2, 2)
    out = nested_to_padded_tensor(src, 0.0, (2, 3))
    assert out.shape == (2, 3) and float(out[0, 2]) == 0.0
