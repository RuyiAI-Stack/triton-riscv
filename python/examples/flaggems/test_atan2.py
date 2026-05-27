import pytest
import torch

from .atan2 import atan2, atan2_out


@pytest.mark.parametrize("shape", [(16, 256), (4, 128)])
def test_atan2_tt(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    y = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref_out = torch.atan2(x, y)
    tri_out = atan2(x, y)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(16, 256), (4, 128)])
def test_atan2_ts(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    y = 0.5

    ref_out = torch.atan2(x, torch.tensor(y))
    tri_out = atan2(x, y)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(16, 256), (4, 128)])
def test_atan2_st(shape):
    torch.manual_seed(0)
    x = 0.5
    y = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref_out = torch.atan2(torch.tensor(x), y)
    tri_out = atan2(x, y)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "shape_A, shape_B", [((16, 256), (256,)), ((4, 1), (4, 128))]
)
def test_atan2_broadcast(shape_A, shape_B):
    torch.manual_seed(0)
    x = torch.randn(shape_A, dtype=torch.float32, device="cpu")
    y = torch.randn(shape_B, dtype=torch.float32, device="cpu")

    ref_out = torch.atan2(x, y)
    tri_out = atan2(x, y)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(16, 256), (4, 128)])
def test_atan2_out(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    y = torch.randn(shape, dtype=torch.float32, device="cpu")
    out = torch.empty(shape, dtype=torch.float32, device="cpu")

    ref_out = torch.atan2(x, y)
    atan2_out(x, y, out)

    torch.testing.assert_close(out, ref_out, rtol=1e-4, atol=1e-4)
