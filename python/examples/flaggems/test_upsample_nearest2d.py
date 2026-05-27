import pytest
import torch

from .upsample_nearest2d import upsample_nearest2d


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_upsample_nearest2d(size):
    torch.manual_seed(0)
    N, C = 2, 4
    x = torch.randn(N, C, size, size, device="cpu", dtype=torch.float32)
    output_size = (size * 2, size * 2)

    ref_out = torch.nn.functional.interpolate(
        x, size=output_size, mode="nearest"
    )
    tri_out = upsample_nearest2d(x, output_size=output_size)

    torch.testing.assert_close(tri_out, ref_out, rtol=0, atol=0)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_upsample_nearest2d_with_scale(size):
    torch.manual_seed(0)
    N, C = 2, 4
    x = torch.randn(N, C, size, size, device="cpu", dtype=torch.float32)

    ref_out = torch.nn.functional.interpolate(
        x, size=(size * 2, size * 2), mode="nearest"
    )
    tri_out = upsample_nearest2d(x, output_size=(size * 2, size * 2))

    torch.testing.assert_close(tri_out, ref_out, rtol=0, atol=0)
