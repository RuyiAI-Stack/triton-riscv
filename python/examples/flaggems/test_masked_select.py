import pytest
import torch

from .masked_select import masked_select


@pytest.mark.parametrize(
    "shape",
    [(16, 256), (4, 128), (512,), (1023,), (1024,)],
)
def test_masked_select(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    mask = torch.randint(0, 2, shape, dtype=torch.bool, device="cpu")

    ref_out = torch.masked_select(x, mask)
    tri_out = masked_select(x, mask)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)
