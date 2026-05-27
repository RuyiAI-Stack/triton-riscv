import pytest
import torch

from .uniform import uniform, uniform_


@pytest.mark.parametrize("size", [512, 1023, 1024])
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.float32, torch.float64]
)
def test_uniform(size, dtype):
    out = uniform(size, from_=0.0, to=1.0, dtype=dtype)
    assert out.shape == (size,)
    assert out.dtype == dtype
    assert (out >= 0.0).all() and (out <= 1.0).all()


@pytest.mark.parametrize("size", [512, 1023, 1024])
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.float32, torch.float64]
)
def test_uniform_(size, dtype):
    x = torch.empty(size, dtype=dtype, device="cpu")
    uniform_(x, from_=0.0, to=1.0)
    assert (x >= 0.0).all() and (x <= 1.0).all()


@pytest.mark.parametrize("size", [512, 1023, 1024])
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.float32, torch.float64]
)
def test_uniform_range(size, dtype):
    out = uniform(size, from_=-5.0, to=5.0, dtype=dtype)
    assert out.dtype == dtype
    assert (out >= -5.0).all() and (out <= 5.0).all()
