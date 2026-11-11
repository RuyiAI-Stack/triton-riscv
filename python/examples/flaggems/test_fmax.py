import pytest
import torch

from .fmax import fmax, fmax_out


def _make_pair(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32)
    y = torch.randn(size, device="cpu", dtype=torch.float32)
    # Cover the three NaN cases: only x NaN, only y NaN, both NaN.
    x[0::8] = float("nan")
    y[1::8] = float("nan")
    x[3::16] = float("nan")
    y[3::16] = float("nan")
    return x, y


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_fmax(size):
    x, y = _make_pair(size)

    ref = torch.fmax(x, y)
    tri = fmax(x, y)

    torch.testing.assert_close(tri, ref, equal_nan=True)


@pytest.mark.parametrize("size", [512, 1024])
def test_fmax_scalar(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32)
    x[::8] = float("nan")

    ref = torch.fmax(x, torch.tensor(0.5))
    tri = fmax(x, 0.5)

    torch.testing.assert_close(tri, ref, equal_nan=True)


@pytest.mark.parametrize("size", [512, 1024])
def test_fmax_int(size):
    torch.manual_seed(0)
    x = torch.randint(-1000, 1001, (size,), dtype=torch.int32, device="cpu")
    y = torch.randint(-1000, 1001, (size,), dtype=torch.int32, device="cpu")

    ref = torch.fmax(x, y)
    tri = fmax(x, y)

    torch.testing.assert_close(tri, ref)


@pytest.mark.parametrize("size", [512, 1024])
def test_fmax_out(size):
    x, y = _make_pair(size)

    ref = torch.fmax(x, y)
    out = torch.empty_like(x)
    ret = fmax_out(x, y, out)

    assert ret is out
    torch.testing.assert_close(out, ref, equal_nan=True)


@pytest.mark.parametrize(
    "shape_a, shape_b",
    [
        ((16, 256), (1, 256)),
        ((4, 128), (4, 1)),
        ((256,), (16, 256)),
    ],
)
def test_fmax_broadcast(shape_a, shape_b):
    torch.manual_seed(0)
    a = torch.randn(shape_a)
    b = torch.randn(shape_b)

    ref_out = torch.fmax(a, b)
    tri_out = fmax(a, b)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


BF16_XFAIL = pytest.mark.xfail(
    strict=True,
    reason="backend: fp32 to bfloat16 conversion miscompiles (RuyiAI-Stack/triton-riscv#114)",
)


@BF16_XFAIL
def test_fmax_bf16():
    torch.manual_seed(0)
    x = torch.randn(128, device="cpu", dtype=torch.bfloat16)
    y = torch.randn(128, device="cpu", dtype=torch.bfloat16)

    ref = torch.fmax(x, y)
    tri = fmax(x, y)

    torch.testing.assert_close(tri, ref, equal_nan=True)


def test_fmax_int64():
    torch.manual_seed(0)
    x = torch.randint(-1000, 1001, (512,), dtype=torch.int64, device="cpu")
    y = torch.randint(-1000, 1001, (512,), dtype=torch.int64, device="cpu")

    ref = torch.fmax(x, y)
    tri = fmax(x, y)

    torch.testing.assert_close(tri, ref)


def test_fmax_scalar_nan():
    torch.manual_seed(0)
    x = torch.randn(512, device="cpu", dtype=torch.float32)
    x[::8] = float("nan")

    ref = torch.fmax(x, torch.tensor(float("nan")))
    tri = fmax(x, float("nan"))

    torch.testing.assert_close(tri, ref, equal_nan=True)


def test_fmax_fp16():
    torch.manual_seed(0)
    x = torch.randn(128, dtype=torch.float16)
    y = torch.randn(128, dtype=torch.float16)
    torch.testing.assert_close(fmax(x, y), torch.fmax(x, y), rtol=1e-3, atol=1e-3)


def test_fmax_out_small_raises():
    x = torch.randn(8)
    y = torch.randn(8)
    with pytest.raises(ValueError):
        fmax_out(x, y, out=torch.empty(2))


def test_fmax_mixed_dtype_promotes():
    # torch promotes (float32, float64) -> float64.
    x = torch.tensor([1.0, float("nan")])
    y = torch.tensor([2.0, 3.0], dtype=torch.float64)
    out = fmax(x, y)
    assert out.dtype == torch.float64
    torch.testing.assert_close(out, torch.fmax(x, y))
