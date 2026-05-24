import pytest
import torch

from .fmin import fmin, fmin_out


@pytest.mark.parametrize(
    "shape",
    [(16, 256), (4, 128), (512,), (1023,), (1024,)],
)
def test_fmin(shape):
    torch.manual_seed(0)
    device = "cpu"

    a = torch.randn(shape, dtype=torch.float32, device=device)
    b = torch.randn(shape, dtype=torch.float32, device=device)

    ref_out = torch.fmin(a, b)
    tri_out = fmin(a, b)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "shape_a, shape_b",
    [
        ((16, 256), (1, 256)),
        ((4, 128), (4, 1)),
        ((256,), (16, 256)),
    ],
)
def test_fmin_broadcast(shape_a, shape_b):
    torch.manual_seed(0)
    device = "cpu"

    a = torch.randn(shape_a, dtype=torch.float32, device=device)
    b = torch.randn(shape_b, dtype=torch.float32, device=device)

    ref_out = torch.fmin(a, b)
    tri_out = fmin(a, b)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-5, atol=1e-5)


def test_fmin_scalar():
    torch.manual_seed(0)
    device = "cpu"

    a = torch.randn((16, 256), dtype=torch.float32, device=device)
    b = 0.5

    ref_out = torch.fmin(a, torch.tensor(b, device=device))
    tri_out = fmin(a, b)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-5, atol=1e-5)


def test_fmin_out():
    torch.manual_seed(0)
    device = "cpu"
    shape = (16, 256)

    a = torch.randn(shape, dtype=torch.float32, device=device)
    b = torch.randn(shape, dtype=torch.float32, device=device)
    out = torch.empty(shape, dtype=torch.float32, device=device)

    ref_out = torch.fmin(a, b)
    fmin_out(a, b, out)

    torch.testing.assert_close(out, ref_out, rtol=1e-5, atol=1e-5)
