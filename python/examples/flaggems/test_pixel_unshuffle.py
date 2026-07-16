import pytest
import torch
import torch.nn.functional as F

from .pixel_unshuffle import pixel_unshuffle, pixel_unshuffle_out


@pytest.mark.parametrize(
    "shape, factor",
    [
        ((2, 4, 8, 8), 2),
        ((1, 16, 6, 6), 3),
        ((4, 3, 12, 12), 2),
        ((2, 8, 16, 16), 4),
        ((3, 12, 10, 10), 5),
        ((2, 512, 4, 4), 2),
        ((2, 1024, 4, 4), 2),
    ],
)
def test_pixel_unshuffle(shape, factor):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    ref = F.pixel_unshuffle(x, factor)
    tri = pixel_unshuffle(x, factor)
    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "shape, factor",
    [
        ((1, 9, 6, 6), 3),
        ((2, 12, 8, 8), 2),
    ],
)
def test_pixel_unshuffle_out(shape, factor):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    ref = F.pixel_unshuffle(x, factor)
    N, C, H, W = shape
    r = factor
    out = torch.empty(N, C * r * r, H // r, W // r, dtype=x.dtype, device=x.device)
    tri = pixel_unshuffle_out(x, factor, out)
    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)
    assert tri is out, "pixel_unshuffle_out must return the same tensor as out"


@pytest.mark.parametrize(
    "shape, factor",
    [
        ((2, 4, 8, 8), 2),
        ((2, 8, 16, 16), 4),
    ],
)
def test_pixel_unshuffle_non_contiguous(shape, factor):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    x_t = x.permute(0, 1, 3, 2).contiguous().permute(0, 1, 3, 2)
    ref = F.pixel_unshuffle(x_t, factor)
    tri = pixel_unshuffle(x_t, factor)
    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)
