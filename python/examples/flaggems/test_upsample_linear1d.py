import pytest
import torch

from .upsample_linear1d import upsample_linear1d


@pytest.mark.parametrize("size", [512, 1023, 1024])
@pytest.mark.parametrize("align_corners", [True, False])
def test_upsample_linear1d(size, align_corners):
    torch.manual_seed(0)
    N, C = 2, 4
    x = torch.randn(N, C, size, device="cpu", dtype=torch.float32)
    output_size = (size * 2,)

    ref_out = torch.nn.functional.interpolate(
        x, size=output_size, mode="linear", align_corners=align_corners
    )
    tri_out = upsample_linear1d(x, output_size, align_corners=align_corners)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_upsample_linear1d_downsample(size):
    torch.manual_seed(0)
    N, C = 2, 4
    x = torch.randn(N, C, size, device="cpu", dtype=torch.float32)
    output_size = (size // 2,)

    ref_out = torch.nn.functional.interpolate(
        x, size=output_size, mode="linear", align_corners=False
    )
    tri_out = upsample_linear1d(x, output_size, align_corners=False)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-3, atol=1e-3)
