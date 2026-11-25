import torch

from .normal_functional import normal_functional


def test_normal_functional():
    out = normal_functional(torch.empty(8), 0.0, 1.0)
    assert out.shape == (8,) and torch.isfinite(out).all()
