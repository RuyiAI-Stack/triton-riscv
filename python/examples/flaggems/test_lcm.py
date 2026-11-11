import pytest
import torch

from .lcm import lcm


def _make_pair(size, dtype):
    torch.manual_seed(0)
    # Small magnitudes keep |x|/gcd*|y| comfortably inside the dtype.
    x = torch.randint(-100, 101, (size,), dtype=dtype, device="cpu")
    y = torch.randint(-100, 101, (size,), dtype=dtype, device="cpu")
    # lcm(x, 0) == lcm(0, x) == 0.
    x[::9] = 0
    y[1::11] = 0
    return x, y


@pytest.mark.parametrize("size", [512, 1023])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_lcm(size, dtype):
    x, y = _make_pair(size, dtype)

    ref = torch.lcm(x, y)
    tri = lcm(x, y)

    torch.testing.assert_close(tri, ref)


def test_lcm_i64_large_primes():
    # The product overflows int64; torch wraps modulo 2^64 and the kernel's
    # uint64 arithmetic must wrap identically.
    x = torch.tensor(
        [9223372036854775783, 4611686018427387847, 9223372036854775783],
        dtype=torch.int64,
    )
    y = torch.tensor(
        [4611686018427387847, 9223372036854775783, 1],
        dtype=torch.int64,
    )

    ref = torch.lcm(x, y)
    tri = lcm(x, y)

    torch.testing.assert_close(tri, ref)


@pytest.mark.parametrize(
    "shape_a, shape_b",
    [
        ((16, 256), (1, 256)),
        ((4, 128), (4, 1)),
        ((256,), (16, 256)),
    ],
)
def test_lcm_broadcast(shape_a, shape_b):
    torch.manual_seed(0)
    a = torch.randint(-100, 101, shape_a, dtype=torch.int32)
    b = torch.randint(-100, 101, shape_b, dtype=torch.int32)

    ref_out = torch.lcm(a, b)
    tri_out = lcm(a, b)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


def test_lcm_int32_min_edge():
    # lcm(INT_MIN, 1) = 2^31 wraps to INT_MIN in int32; the kernel's uint32
    # arithmetic must wrap identically.
    x = torch.tensor([-(2**31)], dtype=torch.int32)
    y = torch.tensor([1], dtype=torch.int32)

    ref = torch.lcm(x, y)
    tri = lcm(x, y)

    torch.testing.assert_close(tri, ref)


@pytest.mark.parametrize("dtype", [torch.int8, torch.int16])
def test_lcm_small_int_dtypes(dtype):
    lo, hi = (-128, 127) if dtype == torch.int8 else (-32768, 32767)
    a = torch.randint(lo, hi, (64,), dtype=dtype)
    b = torch.randint(lo, hi, (64,), dtype=dtype)
    torch.testing.assert_close(lcm(a, b), torch.lcm(a, b))


def test_lcm_float_raises_like_torch():
    a = torch.randn(64)
    b = torch.randn(64)
    with pytest.raises(RuntimeError):
        lcm(a, b)
    with pytest.raises(RuntimeError):
        torch.lcm(a, b)


def test_lcm_mixed_int_float_raises():
    x = torch.tensor([12, 18], dtype=torch.int64)
    y = torch.tensor([6.0, 6.0])
    with pytest.raises(NotImplementedError):
        lcm(x, y)
    with pytest.raises(NotImplementedError):
        lcm(y, x)
