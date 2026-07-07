import pytest
import torch

from .unfold_backward import unfold_backward


def ref_unfold_backward(grad_in, input_sizes, dim, size, step):
    """PyTorch reference for unfold_backward using unfold + backward."""
    x = torch.randn(
        input_sizes,
        device=grad_in.device,
        dtype=grad_in.dtype,
        requires_grad=True,
    )
    x_unfolded = x.unfold(dim, size, step)
    fake_grad = grad_in.to(x_unfolded.dtype)
    x_unfolded.backward(fake_grad, retain_graph=True)
    return x.grad


@pytest.mark.parametrize("size", [10, 16, 32])
def test_unfold_backward(size):
    torch.manual_seed(0)
    window_size = 4
    step = 2
    D = size
    input_sizes = [D]
    L = (D - window_size) // step + 1
    grad_in = torch.randn(L, window_size, dtype=torch.float32, device="cpu")

    ref_grad = ref_unfold_backward(grad_in, input_sizes, 0, window_size, step)
    tri_grad = unfold_backward(grad_in, input_sizes, 0, window_size, step)

    torch.testing.assert_close(tri_grad, ref_grad, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("size", [10, 16, 32])
def test_unfold_backward_2d(size):
    torch.manual_seed(0)
    window_size = 4
    step = 2
    D = size
    input_sizes = [2, D]
    L = (D - window_size) // step + 1
    grad_in = torch.randn(2, L, window_size, dtype=torch.float32, device="cpu")

    ref_grad = ref_unfold_backward(grad_in, input_sizes, 1, window_size, step)
    tri_grad = unfold_backward(grad_in, input_sizes, 1, window_size, step)

    torch.testing.assert_close(tri_grad, ref_grad, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("size", [10, 16, 32])
def test_unfold_backward_basic(size):
    torch.manual_seed(0)
    window_size = 3
    step = 1
    D = size
    input_sizes = [D]
    grad_in = torch.randn(
        D - window_size + 1, window_size, dtype=torch.float32, device="cpu"
    )

    tri_grad = unfold_backward(grad_in, input_sizes, 0, window_size, step)
    assert tri_grad.shape == (D,)
    assert not torch.isnan(tri_grad).any()


def test_unfold_backward_compare_ref():
    window_size = 3
    step = 1
    D = 10
    input_sizes = [D]
    grad_in = torch.randn(D - window_size + 1, window_size, dtype=torch.float32)

    ref = ref_unfold_backward(grad_in, input_sizes, 0, window_size, step)
    tri = unfold_backward(grad_in, input_sizes, 0, window_size, step)

    torch.testing.assert_close(tri, ref, atol=1e-4, rtol=1e-4)
