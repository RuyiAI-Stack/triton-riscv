import pytest
import torch

from .rms_norm import rms_norm, rms_norm_backward, rms_norm_forward


def torch_rms_norm(x, weight, eps=1e-5):
    variance = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight


@pytest.mark.parametrize("M", [1, 16, 64])
@pytest.mark.parametrize("N", [512, 1023, 1024, 4097])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_rms_norm_forward(M, N, dtype):
    x = torch.randn((M, N), dtype=dtype, device="cpu")
    weight = torch.randn((N,), dtype=dtype, device="cpu")
    normalized_shape = (N,)
    eps = 1e-5

    out_triton = rms_norm(x, normalized_shape, weight, eps)
    out_torch = torch_rms_norm(x, weight, eps)

    torch.testing.assert_close(out_triton, out_torch, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("M", [1, 16, 64])
@pytest.mark.parametrize("N", [512, 1023, 1024, 4097])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_rms_norm_backward(M, N, dtype):
    x_triton = torch.randn(
        (M, N), dtype=dtype, device="cpu", requires_grad=True
    )
    weight_triton = torch.randn(
        (N,), dtype=dtype, device="cpu", requires_grad=True
    )

    x_torch = x_triton.clone().detach().requires_grad_(True)
    weight_torch = weight_triton.clone().detach().requires_grad_(True)

    normalized_shape = (N,)
    eps = 1e-5

    out_triton = rms_norm(x_triton, normalized_shape, weight_triton, eps)
    out_torch = torch_rms_norm(x_torch, weight_torch, eps)

    grad_output = torch.randn_like(out_triton)

    out_triton.backward(grad_output)
    out_torch.backward(grad_output)

    torch.testing.assert_close(
        x_triton.grad, x_torch.grad, rtol=1e-3, atol=1e-3
    )
    torch.testing.assert_close(
        weight_triton.grad, weight_torch.grad, rtol=1e-3, atol=1e-3
    )


@pytest.mark.parametrize("M", [1, 16, 64])
@pytest.mark.parametrize("N", [512, 1023, 1024])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_rms_norm_forward_direct(M, N, dtype):
    x = torch.randn((M, N), dtype=dtype, device="cpu")
    weight = torch.randn((N,), dtype=dtype, device="cpu")
    normalized_shape = (N,)
    eps = 1e-5

    y, inv_rms = rms_norm_forward(x, normalized_shape, weight, eps)
    out_torch = torch_rms_norm(x, weight, eps)

    torch.testing.assert_close(y, out_torch, rtol=1e-4, atol=1e-4)
    assert inv_rms.shape == (M,)
    assert inv_rms.dtype == torch.float32


@pytest.mark.parametrize("M", [1, 16, 64])
@pytest.mark.parametrize("N", [512, 1023, 1024])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_rms_norm_backward_direct(M, N, dtype):
    x = torch.randn((M, N), dtype=dtype, device="cpu")
    weight = torch.randn((N,), dtype=dtype, device="cpu")
    normalized_shape = (N,)
    eps = 1e-5
    dy = torch.randn((M, N), dtype=dtype, device="cpu")

    y, inv_rms = rms_norm_forward(x, normalized_shape, weight, eps)
    dx, dw = rms_norm_backward(dy, x, inv_rms, normalized_shape, weight, eps)

    # Verify shapes
    assert dx.shape == x.shape
    assert dw.shape == weight.shape
