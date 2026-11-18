import pytest
import torch

from .arcsin import arcsin, arcsin_, arcsin_out


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_arcsin(size):
    x = torch.linspace(-0.9, 0.9, size, dtype=torch.float32, device="cpu")

    out = arcsin(x)
    ref = torch.asin(x)

    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_arcsin_inplace_and_out():
    x = torch.tensor([-0.5, 0.0, 0.5], dtype=torch.float32, device="cpu")
    out = torch.empty_like(x)

    arcsin_out(x, out=out)
    arcsin_(x)

    torch.testing.assert_close(out, torch.asin(torch.tensor([-0.5, 0.0, 0.5])))
    torch.testing.assert_close(x, out)


def test_arcsin_out_noncontiguous():
    x = torch.linspace(-0.75, 0.75, 6, dtype=torch.float32, device="cpu").reshape(2, 3)
    out_base = torch.empty((3, 2), dtype=torch.float32, device="cpu")
    out = out_base.transpose(0, 1)

    ret = arcsin_out(x, out=out)

    assert ret is out
    assert not out.is_contiguous()
    torch.testing.assert_close(out, torch.asin(x), rtol=1e-5, atol=1e-5)
