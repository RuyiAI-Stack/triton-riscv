import pytest
import torch

from .diagonal import diagonal_backward


@pytest.mark.parametrize(
    "shape, offset, dim1, dim2",
    [
        ((3, 3), 0, 0, 1),
        ((3, 3), 1, 0, 1),
        ((3, 3), -1, 0, 1),
        ((2, 3, 4), 0, 1, 2),
        ((2, 3, 4), 1, 1, 2),
        ((2, 3, 4), -1, 0, 2),
        ((5, 5, 5, 5), 0, 0, 1),
        ((5, 5, 5, 5), 2, 2, 3),
        # Test required sizes: 512, 1023, 1024
        ((512, 512), 0, 0, 1),
        ((1023, 1023), 0, 0, 1),
        ((1024, 1024), 0, 0, 1),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_diagonal_backward(shape, offset, dim1, dim2, dtype):
    torch.manual_seed(0)
    device = "cpu"

    x = torch.randn(shape, dtype=dtype, device=device, requires_grad=True)
    x_ref = x.clone().detach().requires_grad_(True)

    out = torch.diagonal(x, offset, dim1, dim2)
    out_ref = torch.diagonal(x_ref, offset, dim1, dim2)

    grad_output = torch.randn_like(out)

    # Run PyTorch backward
    out_ref.backward(grad_output)

    # Run Triton backward manually since we only have diagonal_backward
    grad_input = diagonal_backward(grad_output, shape, offset, dim1, dim2)

    torch.testing.assert_close(grad_input, x_ref.grad, rtol=1e-3, atol=1e-3)
