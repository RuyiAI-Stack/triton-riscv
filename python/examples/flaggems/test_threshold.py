import pytest
import torch

from .threshold import threshold, threshold_backward


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
@pytest.mark.parametrize("threshold_val,value", [(0.5, 0.0), (-0.5, -1.0)])
def test_threshold(shape, threshold_val, value):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref = torch.threshold(x, threshold_val, value)
    tri = threshold(x, threshold_val, value)

    assert torch.allclose(tri, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(512,), (1024,)])
@pytest.mark.parametrize("threshold_val", [0.5, -0.5])
def test_threshold_backward(shape, threshold_val):
    torch.manual_seed(0)
    x = torch.randn(
        shape, dtype=torch.float32, device="cpu", requires_grad=True
    )
    grad_output = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref = torch.threshold(x, threshold_val, 0.0)
    ref.backward(grad_output)
    ref_grad = x.grad.clone()

    tri_grad = threshold_backward(grad_output, x.detach(), threshold_val)

    torch.testing.assert_close(tri_grad, ref_grad, rtol=1e-4, atol=1e-4)
