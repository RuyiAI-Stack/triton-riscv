import pytest
import torch

from .upsample_nearest3d import upsample_nearest3d


@pytest.mark.parametrize("size", [4, 8, 16])
def test_upsample_nearest3d(size):
    torch.manual_seed(0)
    N, C = 1, 2
    x = torch.randn(N, C, size, size, size, device="cpu", dtype=torch.float32)
    output_size = (size * 2, size * 2, size * 2)

    ref_out = torch.nn.functional.interpolate(x, size=output_size, mode="nearest")
    tri_out = upsample_nearest3d(x, output_size=output_size)

    torch.testing.assert_close(tri_out, ref_out, rtol=0, atol=0)


def test_upsample_nearest3d_scale():
    torch.manual_seed(0)
    x = torch.randn(1, 2, 8, 8, 8, device="cpu", dtype=torch.float32)

    ref_out = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
    tri_out = upsample_nearest3d(
        x, (16, 16, 16), scales_d=2.0, scales_h=2.0, scales_w=2.0
    )

    torch.testing.assert_close(tri_out, ref_out, rtol=0, atol=0)
