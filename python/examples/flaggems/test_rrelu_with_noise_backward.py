import pytest
import torch

from .rrelu_with_noise_backward import rrelu_with_noise_backward


@pytest.mark.parametrize("size", [512, 1023, 1024])
@pytest.mark.parametrize("training", [True, False])
def test_rrelu_with_noise_backward(size, training):
    torch.manual_seed(0)
    grad_output = torch.randn(size, dtype=torch.float32, device="cpu")
    inp = torch.randn(size, dtype=torch.float32, device="cpu")
    noise = torch.rand(size, dtype=torch.float32, device="cpu")
    lower = 0.125
    upper = 0.33333

    ref_out = torch.ops.aten.rrelu_with_noise_backward.default(
        grad_output, inp, noise, lower, upper, training, False
    )

    tri_out = rrelu_with_noise_backward(grad_output, inp, noise, lower, upper, training)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)
