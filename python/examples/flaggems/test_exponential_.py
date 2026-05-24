import pytest
import torch

from .exponential_ import exponential_


@pytest.mark.parametrize(
    "shape",
    [(16, 256), (4, 128), (512,), (1023,), (1024,)],
)
@pytest.mark.parametrize("lambd", [1.0, 0.5, 2.0])
def test_exponential_(shape, lambd):
    torch.manual_seed(0)
    x = torch.zeros(shape, dtype=torch.float32, device="cpu")

    # Run triton implementation
    out = exponential_(x, lambd=lambd)

    # We can't do assert_close since it's a random generation
    # But we can verify statistical properties

    # Check mean: expected value is 1/lambd
    mean_val = out.mean().item()
    expected_mean = 1.0 / lambd
    # Allow 10% variance
    assert abs(mean_val - expected_mean) / expected_mean < 0.1

    # Check minimum value is >= 0
    assert out.min().item() >= 0

    # Verify no NaN or Inf
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
