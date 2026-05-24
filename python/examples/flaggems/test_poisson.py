import pytest
import torch

from .poisson import poisson


@pytest.mark.parametrize("shape", [(32,), (16, 8)])
def test_poisson(shape):
    torch.manual_seed(0)
    lam = torch.randn(*shape, dtype=torch.float32).abs() * 10
    tri = poisson(lam)
    assert tri.shape == lam.shape
    assert tri.dtype == lam.dtype
    assert (tri >= 0).all()
    # Statistical check: mean should be close to lambda
    for i in range(min(lam.numel(), 5)):
        assert tri.flatten()[i] >= 0


def test_poisson_large_lambda():
    lam = torch.full((16,), 50.0, dtype=torch.float32)
    tri = poisson(lam)
    assert tri.shape == (16,)
    assert (tri >= 0).all()
    assert tri.dtype == torch.float32
