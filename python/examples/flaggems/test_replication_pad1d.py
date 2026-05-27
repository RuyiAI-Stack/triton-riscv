import pytest
import torch
import torch.nn.functional as F

from .replication_pad1d import replication_pad1d, replication_pad1d_out


@pytest.mark.parametrize(
    "shape, padding",
    [
        ((5, 10), (1, 1)),
        ((2, 5, 10), (2, 3)),
        ((3, 3, 20), (5, 4)),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_replication_pad1d(shape, padding, dtype):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype)

    out_triton = replication_pad1d(x, padding)
    out_torch = F.pad(x, padding, mode="replicate")

    torch.testing.assert_close(out_triton, out_torch, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    "shape, padding",
    [
        ((5, 10), (1, 1)),
        ((2, 5, 10), (2, 3)),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_replication_pad1d_out(shape, padding, dtype):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype)
    out = torch.empty(F.pad(x, padding, mode="replicate").shape, dtype=dtype)

    ref_out = F.pad(x, padding, mode="replicate")
    replication_pad1d_out(x, padding, out)

    torch.testing.assert_close(out, ref_out, rtol=1e-3, atol=1e-3)
