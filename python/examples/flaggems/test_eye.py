import pytest
import torch

from .eye import eye


@pytest.mark.parametrize(
    "size",
    [
        5,
        16,
        128,
        # Test required sizes: 512, 1023, 1024
        512,
        1023,
        1024,
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_eye(size, dtype):
    torch.manual_seed(0)
    device = "cpu"

    ref_out = torch.eye(size, dtype=dtype, device=device)
    tri_out = eye(size, dtype=dtype, device=device)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-5, atol=1e-5)
