import pytest
import torch

from ._upsample_nearest_exact1d import (
    _upsample_nearest_exact1d,
    _upsample_nearest_exact1d_out,
)


def ref_upsample(x, size=None, scale_factor=None):
    if size is not None:
        return torch.nn.functional.interpolate(x, size=size, mode="nearest")
    else:
        return torch.nn.functional.interpolate(
            x, scale_factor=scale_factor, mode="nearest"
        )


@pytest.mark.parametrize("shape", [(1, 1, 512), (2, 3, 1023), (1, 4, 1024)])
@pytest.mark.parametrize("out_size", [256, 2000, 1024])
def test__upsample_nearest_exact1d_size(shape, out_size):
    torch.manual_seed(0)
    x = torch.randn(shape, device="cpu", dtype=torch.float32)

    ref_out = ref_upsample(x, size=out_size)
    tri_out = _upsample_nearest_exact1d(x, output_size=out_size)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(1, 1, 512), (2, 3, 1023), (1, 4, 1024)])
@pytest.mark.parametrize("scale", [0.5, 2.0, 1.5])
def test__upsample_nearest_exact1d_scale(shape, scale):
    torch.manual_seed(0)
    x = torch.randn(shape, device="cpu", dtype=torch.float32)

    ref_out = ref_upsample(x, scale_factor=scale)
    tri_out = _upsample_nearest_exact1d(x, scale_factor=scale)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(2, 3, 512)])
@pytest.mark.parametrize("out_size", [1024])
def test__upsample_nearest_exact1d_out(shape, out_size):
    torch.manual_seed(0)
    x = torch.randn(shape, device="cpu", dtype=torch.float32)
    out = torch.empty(
        (shape[0], shape[1], out_size), device="cpu", dtype=torch.float32
    )

    ref_out = ref_upsample(x, size=out_size)
    _upsample_nearest_exact1d_out(x, out_size, out=out)

    torch.testing.assert_close(out, ref_out, rtol=1e-4, atol=1e-4)
