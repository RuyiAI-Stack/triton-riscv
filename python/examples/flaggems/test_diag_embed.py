import pytest
import torch

from .diag_embed import diag_embed


@pytest.mark.parametrize("n", [512, 1023, 1024])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_diag_embed(n, dtype):
    torch.manual_seed(0)
    x = torch.randn(n, dtype=dtype, device="cpu")

    ref = torch.diag_embed(x)
    tri = diag_embed(x)

    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("n", [512, 1024])
@pytest.mark.parametrize("offset", [0, 1, -1, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_diag_embed_offset(n, offset, dtype):
    torch.manual_seed(0)
    x = torch.randn(n, dtype=dtype, device="cpu")

    ref = torch.diag_embed(x, offset=offset)
    tri = diag_embed(x, offset=offset)

    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)
