import pytest
import torch
import torch.nn.functional as F

from .reflection_pad2d import reflection_pad2d, reflection_pad2d_out


@pytest.mark.parametrize(
    "shape, padding",
    [
        ((1, 5, 5), (1, 1, 1, 1)),
        ((2, 10, 10), (2, 3, 1, 2)),
        ((2, 3, 20, 20), (5, 4, 3, 2)),
        ((2, 3, 512, 64), (1, 1, 1, 1)),
        ((2, 3, 1023, 64), (1, 1, 1, 1)),
        ((2, 3, 1024, 64), (1, 1, 1, 1)),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_reflection_pad2d(shape, padding, dtype):
    x = torch.randn(shape, dtype=dtype)

    out_triton = reflection_pad2d(x, padding)
    out_torch = F.pad(x, padding, mode="reflect")

    torch.testing.assert_close(out_triton, out_torch, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    "shape, padding",
    [
        ((1, 5, 5), (1, 1, 1, 1)),
        ((2, 10, 10), (2, 3, 1, 2)),
        ((2, 3, 20, 20), (5, 4, 3, 2)),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_reflection_pad2d_out(shape, padding, dtype):
    x = torch.randn(shape, dtype=dtype)

    ref_out = F.pad(x, padding, mode="reflect")
    out = torch.empty(ref_out.shape, dtype=dtype)
    reflection_pad2d_out(x, padding, out)

    torch.testing.assert_close(out, ref_out, rtol=1e-3, atol=1e-3)
