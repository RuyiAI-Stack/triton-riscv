import pytest
import torch

from .fmod_ import fmod_


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_fmod_scalar_(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu") * 10.0
    x_ref = x.clone()

    x_ref.fmod_(2.5)
    fmod_(x, 2.5)

    torch.testing.assert_close(x, x_ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_fmod_tensor_(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu") * 10.0
    y = torch.randn(shape, dtype=torch.float32, device="cpu") + 2.0
    x_ref = x.clone()

    x_ref.fmod_(y)
    fmod_(x, y)

    torch.testing.assert_close(x, x_ref, rtol=1e-5, atol=1e-5)
