import pytest
import torch

from .gcd_ import gcd_


def _make_pair(size, dtype):
    torch.manual_seed(0)
    x = torch.randint(-1000, 1001, (size,), dtype=dtype, device="cpu")
    y = torch.randint(-1000, 1001, (size,), dtype=dtype, device="cpu")
    # Zeros are a defined corner: gcd(x, 0) == |x|, gcd(0, 0) == 0.
    x[::9] = 0
    y[1::11] = 0
    return x, y


@pytest.mark.parametrize("size", [512, 1023])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_gcd_inplace(size, dtype):
    x, y = _make_pair(size, dtype)

    ref = x.clone()
    torch.gcd_(ref, y)

    tri = x.clone()
    out = gcd_(tri, y)

    assert out is tri, "gcd_ should return the same tensor"
    torch.testing.assert_close(tri, ref)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_gcd_inplace_non_contiguous(dtype):
    torch.manual_seed(0)
    x = torch.randint(-1000, 1001, (32, 64), dtype=dtype, device="cpu")
    y = torch.randint(-1000, 1001, (32, 64), dtype=dtype, device="cpu")
    x_sliced = x[::2, ::2]  # non-contiguous

    ref = x_sliced.clone()
    torch.gcd_(ref, y[::2, ::2])

    tri = x_sliced.clone()
    out = gcd_(tri, y[::2, ::2])

    assert out is tri
    torch.testing.assert_close(tri, ref)


def test_gcd_inplace_i64_large_primes():
    # Coprime magnitudes near 2^63 stress the worst-case iteration bound.
    x = torch.tensor(
        [9223372036854775783, 4611686018427387847, 9223372036854775783, 7],
        dtype=torch.int64,
    )
    y = torch.tensor(
        [4611686018427387847, 9223372036854775783, 1, 9223372036854775783],
        dtype=torch.int64,
    )

    ref = x.clone()
    torch.gcd_(ref, y)

    tri = x.clone()
    out = gcd_(tri, y)

    assert out is tri
    torch.testing.assert_close(tri, ref)


@pytest.mark.parametrize(
    "shape_a, shape_b",
    [
        ((16, 256), (1, 256)),
        ((4, 128), (4, 1)),
    ],
)
def test_gcd__broadcast(shape_a, shape_b):
    torch.manual_seed(0)
    ref_src = torch.randint(-100, 101, shape_a, dtype=torch.int32)
    a = ref_src
    b = torch.randint(-100, 101, shape_b, dtype=torch.int32)
    ref = a.clone()
    torch.gcd_(ref, b)

    tri = a.clone()
    out = gcd_(tri, b)

    assert out is tri
    torch.testing.assert_close(tri, ref)


def test_gcd_inplace_small_int_dtypes():
    torch.manual_seed(0)
    x = torch.randint(-50, 51, (128,), dtype=torch.int8, device="cpu")
    y = torch.randint(-50, 51, (128,), dtype=torch.int8, device="cpu")
    x[::9] = 0

    ref = x.clone()
    torch.gcd_(ref, y)

    tri = x.clone()
    gcd_(tri, y)

    torch.testing.assert_close(tri, ref)

    x16 = torch.randint(-1000, 1001, (128,), dtype=torch.int16, device="cpu")
    y16 = torch.randint(-1000, 1001, (128,), dtype=torch.int16, device="cpu")

    ref16 = x16.clone()
    torch.gcd_(ref16, y16)

    tri16 = x16.clone()
    gcd_(tri16, y16)

    torch.testing.assert_close(tri16, ref16)


def test_gcd_inplace_int32_min_edges():
    # gcd(INT_MIN, 0/1/3) is well-defined; the pair (INT_MIN, 6) diverges
    # because of torch's own overflow artifact and is deliberately excluded.
    x = torch.tensor([-(2**31), -(2**31), -(2**31)], dtype=torch.int32)
    y = torch.tensor([0, 1, 3], dtype=torch.int32)

    ref = x.clone()
    torch.gcd_(ref, y)

    tri = x.clone()
    out = gcd_(tri, y)

    assert out is tri
    torch.testing.assert_close(tri, ref)


def test_gcd_float_raises_like_torch():
    # Non-integer inputs take the aten fallback, which rejects them exactly
    # like torch.gcd does.
    a = torch.randn(64)
    b = torch.randn(64)
    with pytest.raises(RuntimeError):
        gcd_(a, b)
    with pytest.raises(RuntimeError):
        torch.gcd(a, b)


def test_gcd__mixed_int_float_raises():
    x = torch.tensor([12, 18], dtype=torch.int64)
    y = torch.tensor([6.0, 6.0])
    with pytest.raises(NotImplementedError):
        gcd_(x, y)
    with pytest.raises(NotImplementedError):
        gcd_(y, x)
