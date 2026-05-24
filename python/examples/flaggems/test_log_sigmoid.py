import pytest
import torch

from .log_sigmoid import log_sigmoid


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_log_sigmoid(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref_out = torch.nn.functional.logsigmoid(x)
    tri_out = log_sigmoid(x)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)
