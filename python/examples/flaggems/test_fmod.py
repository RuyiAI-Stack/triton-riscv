import pytest
import torch

from .fmod import fmod_scalar, fmod_scalar_, fmod_tensor, fmod_tensor_


@pytest.mark.parametrize(
    "shape",
    [(16, 256), (4, 128), (512,), (1023,), (1024,)],
)
def test_fmod_tensor(shape):
    torch.manual_seed(0)
    device = "cpu"

    x = torch.randn(shape, dtype=torch.float32, device=device) * 10.0
    y = torch.randn(shape, dtype=torch.float32, device=device) * 5.0 + 1.0

    ref_out = torch.fmod(x, y)
    tri_out = fmod_tensor(x, y)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    "shape",
    [(16, 256), (4, 128)],
)
def test_fmod_scalar(shape):
    torch.manual_seed(0)
    device = "cpu"

    x = torch.randn(shape, dtype=torch.float32, device=device) * 10.0
    y = 5.0

    ref_out = torch.fmod(x, y)
    tri_out = fmod_scalar(x, y)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    "shape",
    [(16, 256)],
)
def test_fmod_inplace(shape):
    torch.manual_seed(0)
    device = "cpu"

    x = torch.randn(shape, dtype=torch.float32, device=device) * 10.0
    y = torch.randn(shape, dtype=torch.float32, device=device) * 5.0 + 1.0

    x_ref = x.clone()

    x_ref.fmod_(y)
    fmod_tensor_(x, y)

    torch.testing.assert_close(x, x_ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("shape", [(16, 256)])
def test_fmod_scalar_inplace(shape):
    torch.manual_seed(0)
    device = "cpu"

    x = torch.randn(shape, dtype=torch.float32, device=device) * 10.0
    y = 5.0

    x_ref = x.clone()
    x_ref.fmod_(y)

    result = fmod_scalar_(x, y)

    torch.testing.assert_close(x, x_ref, rtol=1e-3, atol=1e-3)
    assert result is x, "fmod_scalar_ must return the input tensor"
