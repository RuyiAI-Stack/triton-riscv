import pytest
import torch

from .elu import elu, elu_, elu_backward


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_elu(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref = torch.nn.functional.elu(x)
    tri = elu(x)

    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64])
def test_elu_dtype(dtype):
    torch.manual_seed(0)
    x = torch.randn(128, dtype=dtype, device="cpu")

    ref = torch.nn.functional.elu(x)
    tri = elu(x)

    rtol = 1e-2 if dtype == torch.float16 else 1e-4
    atol = 1e-3 if dtype == torch.float16 else 1e-4
    torch.testing.assert_close(tri, ref, rtol=rtol, atol=atol)


def test_elu_alpha():
    torch.manual_seed(0)
    x = torch.randn(128, dtype=torch.float32, device="cpu")

    ref = torch.nn.functional.elu(x, alpha=2.0)
    tri = elu(x, alpha=2.0)

    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


def test_elu_scale_input_scale():
    torch.manual_seed(0)
    x = torch.randn(128, dtype=torch.float32, device="cpu")

    tri = elu(x, alpha=1.0, scale=2.0, input_scale=0.5)
    ref = torch.mul(torch.nn.functional.elu(torch.mul(x, 0.5)), 2.0)

    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


def test_elu_inplace():
    x = torch.tensor([-1.0, 0.0, 2.0], dtype=torch.float32, device="cpu")
    x_ref = x.clone()
    torch.nn.functional.elu(x_ref, inplace=True)
    elu_(x)
    torch.testing.assert_close(x, x_ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("is_result", [True, False])
def test_elu_backward(is_result):
    torch.manual_seed(0)
    x = torch.randn(128, dtype=torch.float32, device="cpu")

    grad_output = torch.randn(128, dtype=torch.float32, device="cpu")

    if is_result:
        y = torch.nn.functional.elu(x)
        ref_grad = torch.ops.aten.elu_backward.default(
            grad_output, 1.0, 1.0, 1.0, is_result, y
        )
        tri_grad = elu_backward(grad_output, 1.0, 1.0, 1.0, is_result, y)
    else:
        ref_grad = torch.ops.aten.elu_backward.default(
            grad_output, 1.0, 1.0, 1.0, is_result, x
        )
        tri_grad = elu_backward(grad_output, 1.0, 1.0, 1.0, is_result, x)

    torch.testing.assert_close(tri_grad, ref_grad, rtol=1e-4, atol=1e-4)
