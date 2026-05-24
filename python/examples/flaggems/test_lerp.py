import pytest
import torch

from .lerp import lerp_scalar, lerp_scalar_, lerp_tensor


@pytest.mark.parametrize("shape", [(16, 256), (4, 128)])
@pytest.mark.parametrize("weight", [0.3, 0.7])
def test_lerp_scalar(shape, weight):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    end = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref_out = torch.lerp(x, end, weight)
    tri_out = lerp_scalar(x, end, weight)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(16, 256), (4, 128)])
def test_lerp_tensor(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    end = torch.randn(shape, dtype=torch.float32, device="cpu")
    weight = torch.rand(shape, dtype=torch.float32, device="cpu")

    ref_out = torch.lerp(x, end, weight)
    tri_out = lerp_tensor(x, end, weight)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(16, 256), (4, 128)])
def test_lerp_broadcast(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    end = torch.randn(shape[1], dtype=torch.float32, device="cpu")
    weight = 0.5

    ref_out = torch.lerp(x, end, weight)
    tri_out = lerp_scalar(x, end, weight)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(16, 256)])
@pytest.mark.parametrize("weight", [0.3, 0.7])
def test_lerp_inplace(shape, weight):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    end = torch.randn(shape, dtype=torch.float32, device="cpu")
    x_ref = x.clone()

    x_ref.lerp_(end, weight)
    lerp_scalar_(x, end, weight)

    torch.testing.assert_close(x, x_ref, rtol=1e-4, atol=1e-4)
