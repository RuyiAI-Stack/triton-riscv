import pytest
import torch

from .layernorm import layer_norm


@pytest.mark.parametrize(
    "shape, normalized_shape",
    [
        ((512,), (512,)),
        ((1023,), (1023,)),
        ((1024,), (1024,)),
        ((2, 512), (512,)),
        ((2, 1023), (1023,)),
        ((2, 1024), (1024,)),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("eps", [1e-5])
def test_layer_norm(shape, normalized_shape, dtype, eps):
    a = torch.randn(shape, dtype=dtype, requires_grad=True)
    weight = torch.randn(normalized_shape, dtype=dtype, requires_grad=True)
    bias = torch.randn(normalized_shape, dtype=dtype, requires_grad=True)

    y, mean, rstd = layer_norm(a, normalized_shape, weight, bias, eps)
    out_torch = torch.nn.functional.layer_norm(
        a, normalized_shape, weight, bias, eps
    )

    torch.testing.assert_close(y, out_torch, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "shape, normalized_shape",
    [
        ((2, 512), (512,)),
        ((2, 1024), (1024,)),
    ],
)
def test_layer_norm_backward(shape, normalized_shape):
    torch.manual_seed(0)
    a = torch.randn(shape, dtype=torch.float32, requires_grad=True)
    weight = torch.randn(
        normalized_shape, dtype=torch.float32, requires_grad=True
    )
    bias = torch.randn(
        normalized_shape, dtype=torch.float32, requires_grad=True
    )

    # Reference backward via torch
    ref_out = torch.nn.functional.layer_norm(
        a, normalized_shape, weight, bias, 1e-5
    )
    grad_out = torch.randn_like(ref_out)
    ref_out.backward(grad_out)
    ref_dx = a.grad.clone()
    ref_dw = weight.grad.clone()
    ref_db = bias.grad.clone()

    # Triton backward
    a.grad = None
    weight.grad = None
    bias.grad = None
    from .layernorm import layer_norm_backward

    y, mean, rstd = layer_norm(a, normalized_shape, weight, bias, 1e-5)
    tri_dx, tri_dw, tri_db = layer_norm_backward(
        grad_out,
        a,
        normalized_shape,
        mean,
        rstd,
        weight=weight,
        bias=bias,
        output_mask=(True, True, True),
    )

    torch.testing.assert_close(tri_dx, ref_dx, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(tri_dw, ref_dw, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(tri_db, ref_db, rtol=1e-3, atol=1e-3)
