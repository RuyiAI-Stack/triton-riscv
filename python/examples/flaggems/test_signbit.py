import pytest
import torch

from .signbit import signbit, signbit_out


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_signbit(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32)
    ref_out = torch.signbit(x)
    tri_out = signbit(x)
    torch.testing.assert_close(tri_out, ref_out)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_signbit_fp16(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float16)
    ref_out = torch.signbit(x)
    tri_out = signbit(x)
    torch.testing.assert_close(tri_out, ref_out)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_signbit_bf16(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.bfloat16)
    ref_out = torch.signbit(x)
    tri_out = signbit(x)
    torch.testing.assert_close(tri_out, ref_out)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_signbit_out(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32)
    out = torch.empty(size, device="cpu", dtype=torch.bool)
    ref_out = torch.signbit(x)
    signbit_out(x, out=out)
    torch.testing.assert_close(out, ref_out)


def test_signbit_special_values():
    torch.manual_seed(0)
    x = torch.tensor(
        [0.0, -0.0, float("inf"), float("-inf"), float("nan"), -1.0, 1.0],
        dtype=torch.float32,
    )
    ref_out = torch.signbit(x)
    tri_out = signbit(x)
    torch.testing.assert_close(tri_out, ref_out)


def test_signbit_out_different_dtype():
    x = torch.randn(128, device="cpu", dtype=torch.float32)
    out = torch.empty(128, device="cpu", dtype=torch.bool)
    signbit_out(x, out=out)
    ref = torch.signbit(x)
    torch.testing.assert_close(out, ref)


def test_signbit_out_none():
    x = torch.randn(128, device="cpu", dtype=torch.float32)
    tri_out = signbit_out(x)
    ref_out = torch.signbit(x)
    torch.testing.assert_close(tri_out, ref_out)
