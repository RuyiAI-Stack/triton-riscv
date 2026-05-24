import pytest
import torch

from .isin import isin


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_isin(size):
    torch.manual_seed(0)
    x = torch.randint(0, 10, (size,), device="cpu", dtype=torch.int64)
    y = torch.randint(0, 10, (50,), device="cpu", dtype=torch.int64)

    ref_out = torch.isin(x, y)
    tri_out = isin(x, y)

    torch.testing.assert_close(tri_out, ref_out)
