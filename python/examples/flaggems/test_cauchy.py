import torch

from .cauchy import cauchy


def test_cauchy():
    out = cauchy(8, median=0.0, sigma=1.0)
    assert out.shape == (8,) and torch.isfinite(out).all()
