import pytest
import torch

from .slice_backward import slice_backward


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_slice_backward(size):
    torch.manual_seed(0)
    input_sizes = (size,)
    dim = 0
    start = 0
    end = size // 2
    step = 1

    grad_output = torch.randn(end - start, dtype=torch.float32, device="cpu")

    ref_grad_input = torch.zeros(
        input_sizes, dtype=torch.float32, device="cpu"
    )
    ref_grad_input[start:end:step] = grad_output

    tri_grad_input = slice_backward(
        grad_output, input_sizes, dim, start, end, step
    )

    torch.testing.assert_close(
        tri_grad_input, ref_grad_input, rtol=1e-4, atol=1e-4
    )


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_slice_backward_2d(size):
    torch.manual_seed(0)
    input_sizes = (size, size)
    dim = 0
    start = 0
    end = size // 2
    step = 1

    grad_output_shape = list(input_sizes)
    grad_output_shape[dim] = end - start
    grad_output = torch.randn(
        grad_output_shape, dtype=torch.float32, device="cpu"
    )

    ref_grad_input = torch.zeros(
        input_sizes, dtype=torch.float32, device="cpu"
    )
    ref_grad_input[start:end:step, :] = grad_output

    tri_grad_input = slice_backward(
        grad_output, input_sizes, dim, start, end, step
    )

    torch.testing.assert_close(
        tri_grad_input, ref_grad_input, rtol=1e-4, atol=1e-4
    )
