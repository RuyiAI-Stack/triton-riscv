import pytest
import torch

from .tile import tile


@pytest.mark.parametrize(
    "shape, dims",
    [
        ((16, 32), (1, 2)),
        ((4, 8), (2, 3)),
        ((512,), (2,)),
        ((1023,), (1,)),
        ((1024,), (3,)),
    ],
)
def test_tile(shape, dims):
    torch.manual_seed(0)
    x = torch.randn(shape, device="cpu", dtype=torch.float32)

    ref_out = torch.tile(x, dims)
    tri_out = tile(x, dims)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)
