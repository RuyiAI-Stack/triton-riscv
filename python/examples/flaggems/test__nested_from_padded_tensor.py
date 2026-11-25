import torch

from ._nested_from_padded_tensor import _nested_from_padded_tensor


def test__nested_from_padded_tensor():
    padded = torch.arange(6.0)
    got = _nested_from_padded_tensor(padded, torch.tensor([0, 4]))
    assert got.numel() == 4
