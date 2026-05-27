import pytest
import torch

from .select_scatter import select_scatter


@pytest.mark.parametrize("size", [512, 1023, 1024])
@pytest.mark.parametrize("dim", [0, 1])
def test_select_scatter(size, dim):
    torch.manual_seed(0)
    if dim == 0:
        inp = torch.randn(size, 8, dtype=torch.float32, device="cpu")
        src = torch.randn(8, dtype=torch.float32, device="cpu")
        index = size // 2
    else:
        inp = torch.randn(8, size, dtype=torch.float32, device="cpu")
        src = torch.randn(8, dtype=torch.float32, device="cpu")
        index = size // 2

    inp_ref = inp.clone()
    ref_out = torch.select_scatter(inp_ref, src, dim, index)

    tri_out = select_scatter(inp, src, dim, index)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-5, atol=1e-5)
