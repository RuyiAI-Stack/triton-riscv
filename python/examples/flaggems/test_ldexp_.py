import pytest
import torch

from .ldexp_ import ldexp_


def _assert_ldexp_inplace_bitexact(x, e):
    ref = x.clone()
    torch.ldexp_(ref, e)

    tri = x.clone()
    out = ldexp_(tri, e)

    assert out is tri, "ldexp_ should return the same tensor"
    torch.testing.assert_close(tri, ref, rtol=0.0, atol=0.0, equal_nan=True)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_ldexp_inplace(size):
    torch.manual_seed(0)
    x = torch.randn(size, dtype=torch.float32) * torch.pow(
        2.0, torch.randint(-60, 60, (size,)).float()
    )
    e = torch.randint(-100, 100, (size,), dtype=torch.int32)
    _assert_ldexp_inplace_bitexact(x, e)


@pytest.mark.parametrize("self_dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("e_dtype", [torch.int16, torch.int32, torch.int64])
def test_ldexp_inplace_dtype_matrix(self_dtype, e_dtype):
    torch.manual_seed(0)
    x = (
        torch.randn(512, dtype=torch.float32)
        * torch.pow(2.0, torch.randint(-12, 12, (512,)).float())
    ).to(self_dtype)
    e = torch.randint(-30, 30, (512,), dtype=torch.int64).to(e_dtype)
    _assert_ldexp_inplace_bitexact(x, e)


def test_ldexp_inplace_float_exponent():
    torch.manual_seed(0)
    x = torch.randn(512, dtype=torch.float32)
    e = torch.randn(512, dtype=torch.float32) * 3

    ref = x.clone()
    torch.ldexp_(ref, e)

    tri = x.clone()
    out = ldexp_(tri, e)

    # A float exponent degrades ldexp to x * 2**e; torch.ldexp_ rounds that
    # product once while the kernel rounds exp2 and the product, so allow
    # one fp32 ulp of difference.
    assert out is tri
    torch.testing.assert_close(tri, ref, rtol=1e-6, atol=1e-6)


def test_ldexp_inplace_exponent_extremes():
    # e near +-2**31 saturates to +-inf / signed zero (torch 2.11+ wraps
    # out-of-int32 int64 exponents: 2**32 + 5 acts as 5).
    x = torch.tensor([1.5, -1.5, 0.75, -0.75, 6e-8], dtype=torch.float32)
    e = torch.tensor(
        [2**31 - 1, 2**31 - 1, -(2**31), -(2**31), 2**32 + 5], dtype=torch.int64
    )
    _assert_ldexp_inplace_bitexact(x, e)


def test_ldexp_inplace_special_values():
    x = torch.tensor(
        [0.0, -0.0, float("inf"), float("-inf"), float("nan"), 2.0**-149, 2.0**127],
        dtype=torch.float32,
    )
    e = torch.tensor([10, -10, 3, -3, 0, 30, 1], dtype=torch.int32)
    _assert_ldexp_inplace_bitexact(x, e)


def test_ldexp_inplace_non_contiguous():
    torch.manual_seed(0)
    base = torch.randn(64, 32, dtype=torch.float32)
    e = torch.randint(-20, 20, (32, 64), dtype=torch.int32)

    ref_view = base.t().clone()
    torch.ldexp_(ref_view, e)

    tri_view = base.t()
    assert not tri_view.is_contiguous()
    out = ldexp_(tri_view, e)

    assert out is tri_view
    # the write went through the view into the base storage
    torch.testing.assert_close(tri_view, ref_view, rtol=0.0, atol=0.0, equal_nan=True)
    assert torch.equal(base.t(), ref_view)


@pytest.mark.parametrize(
    "shape_a, shape_e", [((16, 256), (16, 1)), ((16, 256), (16, 256))]
)
def test_ldexp_inplace_broadcast(shape_a, shape_e):
    torch.manual_seed(0)
    x = torch.randn(shape_a, dtype=torch.float32)
    e = torch.randint(-5, 5, shape_e, dtype=torch.int32)
    _assert_ldexp_inplace_bitexact(x, e)


def test_ldexp_inplace_broadcast_grow_raises():
    # self smaller than the broadcast result is not writable in place.
    x = torch.zeros(4, 1, dtype=torch.float32)
    e = torch.ones(4, 8, dtype=torch.int32)
    with pytest.raises(RuntimeError):
        ldexp_(x, e)


@pytest.mark.parametrize("self_dtype", [torch.int32, torch.int64])
def test_ldexp_inplace_integer_self_raises(self_dtype):
    # a float result cannot be written back into an integer self
    x = torch.ones(8, dtype=self_dtype)
    e = torch.ones(8, dtype=torch.int32)
    with pytest.raises(RuntimeError, match="can't be cast to the desired output type"):
        ldexp_(x, e)


def test_ldexp_inplace_scalar_other_raises():
    x = torch.ones(8, dtype=torch.float32)
    with pytest.raises(TypeError):
        ldexp_(x, 3)


def test_ldexp_inplace_float64_fallback():
    torch.manual_seed(0)
    x = torch.randn(128, dtype=torch.float64)
    e = torch.randint(-40, 40, (128,), dtype=torch.int32)
    _assert_ldexp_inplace_bitexact(x, e)


@pytest.mark.parametrize("shape", [(0,), (0, 37)])
def test_ldexp_inplace_empty(shape):
    x = torch.randn(shape, dtype=torch.float32)
    e = torch.randint(-5, 5, shape, dtype=torch.int32)
    out = ldexp_(x, e)
    assert out is x
    assert x.shape == shape


def test_ldexp_inplace_triton_path(monkeypatch):
    # Smoke: the float32 + int32 path must launch the Triton kernel.
    import importlib

    mod = importlib.import_module("flaggems.ldexp_")

    launches = []
    real_run = mod.ldexp_inplace_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod.ldexp_inplace_kernel, "run", spy_run)
    x = torch.randn(1024, dtype=torch.float32)
    e = torch.randint(-10, 10, (1024,), dtype=torch.int32)
    ref = x.clone()
    torch.ldexp_(ref, e)

    out = mod.ldexp_(x, e)
    assert len(launches) == 1, "ldexp_(A, B) must go through the Triton kernel"
    assert out is x
    torch.testing.assert_close(x, ref, rtol=0.0, atol=0.0)
