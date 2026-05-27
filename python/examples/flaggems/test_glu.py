import pytest
import torch

from .glu import glu, glu_backward


@pytest.mark.parametrize(
    "shape, dim",
    [
        ((512,), -1),
        ((1023,), -1),
        ((1024,), -1),
        ((16, 256), -1),
        ((4, 128), -1),
        ((8, 512), 1),
        ((512, 128), 0),
    ],
)
def test_glu_forward(shape, dim):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    # Ensure even split along the tested dim
    if x.shape[dim] % 2 != 0:
        shape_list = list(shape)
        shape_list[dim] = shape_list[dim] + 1
        x = torch.randn(shape_list, dtype=torch.float32, device="cpu")

    ref_out = torch.nn.functional.glu(x, dim=dim)
    tri_out = glu(x, dim=dim)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "shape, dim",
    [
        ((512,), -1),
        ((1023,), -1),
        ((1024,), -1),
        ((16, 256), -1),
        ((4, 128), -1),
        ((8, 512), 1),
        ((512, 128), 0),
    ],
)
def test_glu_backward(shape, dim):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    if x.shape[dim] % 2 != 0:
        shape_list = list(shape)
        shape_list[dim] = shape_list[dim] + 1
        x = torch.randn(shape_list, dtype=torch.float32, device="cpu")

    # Reference
    x_ref = x.clone().requires_grad_()
    ref_out = torch.nn.functional.glu(x_ref, dim=dim)
    grad_output = torch.randn_like(ref_out)
    ref_out.backward(grad_output)

    # Our implementation
    tri_grad = glu_backward(grad_output, x, dim=dim)

    torch.testing.assert_close(tri_grad, x_ref.grad, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "shape, dim",
    [
        ((512,), -1),
        ((1023,), -1),
        ((1024,), -1),
        ((16, 256), -1),
    ],
)
def test_glu_autograd(shape, dim):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    if x.shape[dim] % 2 != 0:
        shape_list = list(shape)
        shape_list[dim] = shape_list[dim] + 1
        x = torch.randn(shape_list, dtype=torch.float32, device="cpu")

    # Test that our forward can be used with autograd
    x_ref = x.clone().requires_grad_()
    x_tri = x.clone().requires_grad_()

    ref_out = torch.nn.functional.glu(x_ref, dim=dim)
    tri_out = glu(x_tri, dim=dim)

    grad_output = torch.randn_like(ref_out)

    # Reference backward (torch's native autograd for glu)
    ref_out.backward(grad_output)

    # Our forward + manual backward
    tri_grad = glu_backward(grad_output, x_tri, dim=dim)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(tri_grad, x_ref.grad, rtol=1e-4, atol=1e-4)
