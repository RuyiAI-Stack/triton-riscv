import pytest
import torch

from .bernoulli_ import bernoulli_


@pytest.mark.parametrize("shape", [(1024,), (64, 64), (128,)])
@pytest.mark.parametrize("p", [0.0, 0.5, 1.0])
def test_bernoulli_inplace(shape, p):
    torch.manual_seed(0)
    x = torch.zeros(shape, dtype=torch.float32, device="cpu")
    result = bernoulli_(x, p=p)

    assert result is x, "bernoulli_ must return self"

    if p == 0.0:
        assert (x == 0.0).all(), "p=0 should produce all zeros"
    elif p == 1.0:
        assert (x == 1.0).all(), "p=1 should produce all ones"
    else:
        mean = x.mean().item()
        assert 0.3 < mean < 0.7, f"Expected mean ~0.5 for p=0.5, got {mean}"


@pytest.mark.parametrize("shape", [(10000,)])
@pytest.mark.parametrize("p", [0.3, 0.7])
def test_bernoulli_statistical(shape, p):
    torch.manual_seed(0)
    x = torch.zeros(shape, dtype=torch.float32, device="cpu")
    bernoulli_(x, p=p)

    observed_p = x.mean().item()
    assert abs(observed_p - p) < 0.05, f"Expected ~{p}, got {observed_p}"
