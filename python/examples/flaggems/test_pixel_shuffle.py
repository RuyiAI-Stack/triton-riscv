import pytest
import torch
import torch.nn.functional as F

from .pixel_shuffle import pixel_shuffle


@pytest.mark.parametrize(
    "batch, c, h, w, scale",
    [
        (2, 16, 4, 4, 2),
        (2, 512, 2, 2, 2),
        (2, 1020, 2, 2, 2),
        (2, 1024, 2, 2, 2),
    ],
)
def test_pixel_shuffle(batch, c, h, w, scale):
    x = torch.randn(batch, c, h, w, dtype=torch.float32, device="cpu")
    ref = F.pixel_shuffle(x, scale)
    tri = pixel_shuffle(x, scale)
    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)
