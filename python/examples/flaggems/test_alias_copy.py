import pytest
import torch

from .alias_copy import alias_copy, alias_copy_out


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_alias_copy(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32)

    ref_out = x.clone()
    tri_out = alias_copy(x)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_alias_copy_out(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32)
    out = torch.empty_like(x)

    ref_out = x.clone()
    alias_copy_out(x, out)

    torch.testing.assert_close(out, ref_out, rtol=1e-4, atol=1e-4)
