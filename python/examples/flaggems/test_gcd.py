import pytest
import torch

from .gcd import gcd, gcd_out


@pytest.mark.parametrize("dtype", [torch.int16, torch.int32, torch.int64])
@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_gcd(dtype, size):
    torch.manual_seed(0)
    a = torch.randint(-1000, 1000, (size,), dtype=dtype, device="cpu")
    b = torch.randint(-1000, 1000, (size,), dtype=dtype, device="cpu")

    ref = torch.gcd(a, b)
    tri = gcd(a, b)

    torch.testing.assert_close(tri, ref)


@pytest.mark.parametrize("dtype", [torch.int16, torch.int32, torch.int64])
@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_gcd_out(dtype, size):
    torch.manual_seed(0)
    a = torch.randint(-1000, 1000, (size,), dtype=dtype, device="cpu")
    b = torch.randint(-1000, 1000, (size,), dtype=dtype, device="cpu")

    ref = torch.gcd(a, b)

    out = torch.empty_like(ref)
    result = gcd_out(a, b, out=out)

    torch.testing.assert_close(result, ref)
    torch.testing.assert_close(out, ref)


@pytest.mark.parametrize("dtype", [torch.int16, torch.int32, torch.int64])
@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_gcd_out_none(dtype, size):
    torch.manual_seed(0)
    a = torch.randint(-1000, 1000, (size,), dtype=dtype, device="cpu")
    b = torch.randint(-1000, 1000, (size,), dtype=dtype, device="cpu")

    ref = torch.gcd(a, b)
    result = gcd_out(a, b)

    torch.testing.assert_close(result, ref)
