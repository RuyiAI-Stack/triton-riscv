import pytest
import torch

from .nonzero import nonzero


@pytest.mark.parametrize(
    "shape",
    [(16, 256), (4, 128), (512,), (1023,), (1024,)],
)
def test_nonzero(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    threshold = 0.5
    x = (x > threshold).to(torch.float32)

    ref_out = torch.nonzero(x)
    tri_out = nonzero(x)

    if ref_out.numel() == 0 and tri_out.numel() == 0:
        return
    torch.testing.assert_close(tri_out, ref_out)


@pytest.mark.parametrize(
    "shape",
    [(16, 256), (4, 128)],
)
def test_nonzero_as_tuple(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    threshold = 0.5
    x = (x > threshold).to(torch.float32)

    ref_out = torch.nonzero(x, as_tuple=True)
    tri_out = nonzero(x, as_tuple=True)

    for t, r in zip(tri_out, ref_out):
        torch.testing.assert_close(t, r)
