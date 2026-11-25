import torch

from ._nested_from_padded import _nested_from_padded


def test__nested_from_padded():
    padded = torch.arange(6.0)
    got = _nested_from_padded(padded, torch.tensor([0, 4]))
    assert got.numel() == 4
