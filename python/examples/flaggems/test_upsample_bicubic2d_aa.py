import pytest
import torch

from .upsample_bicubic2d_aa import _upsample_bicubic2d_aa


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("channels", [1, 3])
def test_upsample_bicubic2d_aa_correctness(batch_size, channels):
    torch.manual_seed(0)
    x = torch.randn(batch_size, channels, 4, 4, dtype=torch.float32, device="cpu")

    ref = torch.nn.functional.interpolate(
        x, size=(8, 8), mode="bicubic", align_corners=False, antialias=True
    )
    tri = _upsample_bicubic2d_aa(x, (8, 8), align_corners=False)

    torch.testing.assert_close(tri, ref, rtol=1e-3, atol=1e-3)


def test_upsample_bicubic2d_aa_align_corners():
    torch.manual_seed(0)
    x = torch.randn(1, 1, 4, 4, dtype=torch.float32, device="cpu")

    ref = torch.nn.functional.interpolate(
        x, size=(8, 8), mode="bicubic", align_corners=True, antialias=True
    )
    tri = _upsample_bicubic2d_aa(x, (8, 8), align_corners=True)

    torch.testing.assert_close(tri, ref, rtol=1e-3, atol=1e-3)


def test_upsample_bicubic2d_aa_downsample():
    torch.manual_seed(0)
    x = torch.randn(1, 1, 8, 8, dtype=torch.float32, device="cpu")

    ref = torch.nn.functional.interpolate(
        x, size=(4, 4), mode="bicubic", align_corners=False, antialias=True
    )
    tri = _upsample_bicubic2d_aa(x, (4, 4), align_corners=False)

    torch.testing.assert_close(tri, ref, rtol=1e-3, atol=1e-3)
