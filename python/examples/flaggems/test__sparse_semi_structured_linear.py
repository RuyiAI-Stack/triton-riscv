import torch

from ._sparse_semi_structured_linear import _sparse_semi_structured_linear


def test__sparse_semi_structured_linear():
    torch.manual_seed(0)
    weight = torch.randn(4, 8)
    inputs = torch.randn(2, 8)
    out = _sparse_semi_structured_linear(inputs, weight)
    assert out.shape == (2, 4)
