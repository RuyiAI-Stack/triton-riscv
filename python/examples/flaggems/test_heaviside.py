import pytest
import torch

from .heaviside import heaviside, heaviside_out


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_heaviside_scalar_values(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32)
    x[::7] = 0.0  # exact zeros take the `values` branch

    ref = torch.heaviside(x, torch.tensor(0.5))
    tri = heaviside(x, 0.5)

    torch.testing.assert_close(tri, ref)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_heaviside_tensor_values(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32)
    x[::7] = 0.0
    values = torch.rand(size, device="cpu", dtype=torch.float32)

    ref = torch.heaviside(x, values)
    tri = heaviside(x, values)

    torch.testing.assert_close(tri, ref)


def test_heaviside_out():
    torch.manual_seed(0)
    x = torch.randn(1024, device="cpu", dtype=torch.float32)
    x[::7] = 0.0
    values = torch.rand(1024, device="cpu", dtype=torch.float32)

    ref = torch.heaviside(x, values)
    out = torch.empty_like(x)
    ret = heaviside_out(x, values, out)

    assert ret is out
    torch.testing.assert_close(out, ref)


@pytest.mark.parametrize(
    "shape_a, shape_b",
    [
        ((16, 256), (1, 256)),
        ((4, 128), (4, 1)),
        ((256,), (16, 256)),
    ],
)
def test_heaviside_broadcast(shape_a, shape_b):
    torch.manual_seed(0)
    a = torch.randn(shape_a)
    b = torch.rand(shape_b)

    ref_out = torch.heaviside(a, b)
    tri_out = heaviside(a, b)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


def test_heaviside_bf16():
    # tl.where miscompiles on bf16 (issue #119); the wrapper falls back to
    # torch.heaviside, so bf16 must match torch exactly.
    torch.manual_seed(0)
    x = torch.randn(128, device="cpu", dtype=torch.bfloat16)
    x[::7] = 0.0

    ref = torch.heaviside(x, torch.tensor(0.5, dtype=torch.bfloat16))
    tri = heaviside(x, 0.5)

    torch.testing.assert_close(tri, ref)


def test_heaviside_specials():
    # torch CPU semantics: +inf -> 1, -inf -> 0, NaN -> 0 (anchored by probe
    # on torch 2.15 CPU, which differs from the propagating doc form).
    x = torch.tensor([float("inf"), float("-inf"), float("nan"), 0.0, 2.0, -2.0])
    v = torch.full_like(x, 0.5)
    torch.testing.assert_close(heaviside(x, v), torch.heaviside(x, v))


def test_heaviside_fp16():
    torch.manual_seed(0)
    x = torch.randn(128, dtype=torch.float16)
    v = torch.full_like(x, 0.5)
    torch.testing.assert_close(
        heaviside(x, v), torch.heaviside(x, v), rtol=1e-3, atol=1e-3
    )


def test_heaviside_out_small_raises():
    x = torch.randn(8)
    v = torch.full_like(x, 0.5)
    small = torch.empty(2)
    with pytest.raises(ValueError):
        heaviside_out(x, v, small)


def test_heaviside_scalar_values_nonfinite():
    # A scalar values only enters the x==0 branch; the torch reference also
    # requires a tensor values.
    x = torch.tensor([2.0, -2.0, 0.0, float("inf"), float("-inf"), float("nan")])
    for v in (float("inf"), float("nan"), -1.0):
        ref = torch.heaviside(x, torch.tensor(v))
        torch.testing.assert_close(heaviside(x, v), ref, equal_nan=True)


def test_heaviside_tensor_values_nonfinite():
    # values only feeds the x==0 output; x!=0 entries must stay 1/0 even
    # when values holds inf/nan.
    x = torch.tensor([2.0, -2.0, 0.0, float("inf"), float("-inf"), float("nan")])
    v = torch.tensor([float("inf"), float("inf"), float("inf"), 3.0, 3.0, 3.0])
    torch.testing.assert_close(heaviside(x, v), torch.heaviside(x, v))
    v2 = torch.tensor([float("nan"), float("nan"), float("nan"), 3.0, 3.0, 3.0])
    torch.testing.assert_close(heaviside(x, v2), torch.heaviside(x, v2), equal_nan=True)


def test_heaviside_mixed_dtype_raises():
    x = torch.ones(4)
    v = torch.ones(4, dtype=torch.float64)
    with pytest.raises(RuntimeError):
        heaviside(x, v)
