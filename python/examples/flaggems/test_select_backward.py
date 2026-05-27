import pytest
import torch

from .select_backward import select_backward


@pytest.mark.parametrize("size", [512, 1023, 1024])
@pytest.mark.parametrize("dim", [0, 1])
def test_select_backward(size, dim):
    torch.manual_seed(0)
    if dim == 0:
        input_sizes = (size, 8)
        index = size // 2
    else:
        input_sizes = (8, size)
        index = size // 2

    grad = torch.randn(
        input_sizes[:dim] + input_sizes[dim + 1 :],
        dtype=torch.float32,
        device="cpu",
    )

    ref_out = torch.zeros(input_sizes, dtype=torch.float32, device="cpu")
    ref_grad = grad
    if dim == 0:
        ref_out[index, :] = ref_grad
    else:
        ref_out[:, index] = ref_grad

    tri_out = select_backward(grad, input_sizes, dim, index)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-5, atol=1e-5)
