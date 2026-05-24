import pytest
import torch

from .scaled_softmax import (
    scaled_softmax,
    scaled_softmax_backward,
    scaled_softmax_forward,
)


def torch_scaled_softmax(x, scale_factor):
    return torch.nn.functional.softmax(x * scale_factor, dim=-1)


@pytest.mark.parametrize("query_seq_len", [16, 64])
@pytest.mark.parametrize("key_seq_len", [512, 1024, 2048])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_scaled_softmax_forward(query_seq_len, key_seq_len, dtype):
    shape = (1, 2, query_seq_len, key_seq_len)
    x = torch.randn(shape, dtype=dtype, device="cpu")
    scale_factor = 0.5

    out_triton = scaled_softmax(x, scale_factor)
    out_torch = torch_scaled_softmax(x, scale_factor)

    torch.testing.assert_close(out_triton, out_torch, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    "shape, scale_factor",
    [
        ((2, 4, 32, 128), 0.3),
        ((1, 1, 8, 256), 1.0),
        ((1, 2, 16, 512), 0.1),
        ((2, 2, 32, 256), 0.8),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_scaled_softmax_varied(shape, scale_factor, dtype):
    x = torch.randn(shape, dtype=dtype, device="cpu")

    out_triton = scaled_softmax(x, scale_factor)
    out_torch = torch_scaled_softmax(x, scale_factor)

    torch.testing.assert_close(out_triton, out_torch, rtol=1e-3, atol=1e-3)


def test_scaled_softmax_edge_cases():
    x = torch.randn((1, 1, 1, 16), dtype=torch.float32, device="cpu")

    with pytest.raises(AssertionError):
        scaled_softmax(x, 1.0)


@pytest.mark.parametrize("query_seq_len", [16, 64])
@pytest.mark.parametrize("key_seq_len", [512, 1024, 2048])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_scaled_softmax_backward(query_seq_len, key_seq_len, dtype):
    shape = (1, 2, query_seq_len, key_seq_len)
    x = torch.randn(shape, dtype=dtype, device="cpu", requires_grad=True)
    scale_factor = 0.5

    out = torch_scaled_softmax(x, scale_factor)
    grad_output = torch.randn_like(out)
    out.backward(grad_output)
    ref_grad = x.grad.clone()

    x.grad.zero_()
    out = scaled_softmax_forward(x, scale_factor)
    tri_grad = scaled_softmax_backward(grad_output, out, scale_factor)

    torch.testing.assert_close(tri_grad, ref_grad, rtol=1e-3, atol=1e-3)
