import pytest
import torch

from .eye_m import eye_m


@pytest.mark.parametrize(
    "n, m",
    [
        (5, 5),
        (3, 8),
        (8, 3),
        (16, 256),
        (256, 16),
        # Test required sizes: 512, 1023, 1024
        (512, 512),
        (1023, 1023),
        (1024, 1024),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_eye_m(n, m, dtype):
    torch.manual_seed(0)
    device = "cpu"

    ref_out = torch.eye(n, m, dtype=dtype, device=device)
    tri_out = eye_m(n, m, dtype=dtype, device=device)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-5, atol=1e-5)
