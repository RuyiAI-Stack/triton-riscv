import pytest
import torch

from .nonzero_numpy import nonzero_numpy


@pytest.mark.parametrize(
    "shape",
    [(16, 256), (4, 128), (512,), (1023,), (1024,)],
)
def test_nonzero_numpy(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    threshold = 0.5
    x = (x > threshold).to(torch.float32)

    ref_out = torch.nonzero(x, as_tuple=False).unbind(dim=1)
    tri_out = nonzero_numpy(x)

    assert len(tri_out) == len(ref_out)
    for t, r in zip(tri_out, ref_out):
        torch.testing.assert_close(t, r)
