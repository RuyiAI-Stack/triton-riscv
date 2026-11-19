import math

import pytest
import torch

from .float_power_ import (
    float_power_scalar_tensor,
    float_power_scalar_tensor_out,
    float_power_tensor_scalar,
    float_power_tensor_scalar_,
    float_power_tensor_scalar_out,
    float_power_tensor_tensor,
    float_power_tensor_tensor_,
    float_power_tensor_tensor_out,
)


def _inplace(base, exp):
    ref_base = base.clone()
    ref_base.float_power_(exp)

    tri = base.clone()
    ret = float_power_tensor_tensor_(tri, exp)

    assert ret is tri, "float_power_ must return the base tensor"
    return tri, ref_base


def test_float_power_special_contract():
    base = torch.tensor(
        [0.0, 0.0, 0.0, 0.0, 0.0, -0.0, -0.0, -0.0, 1.0, -2.0, -2.0, -2.0, -2.0, -2.0],
        dtype=torch.float64,
    )
    exp = torch.tensor(
        [
            0.0,
            -1.0,
            -2.5,
            2.5,
            float("inf"),
            3.0,
            0.5,
            0.0,
            3.7,
            3.0,
            2.0,
            0.5,
            -3.0,
            0.0,
        ],
        dtype=torch.float64,
    )
    tri, ref = _inplace(base, exp)
    torch.testing.assert_close(tri, ref, rtol=0.0, atol=0.0, equal_nan=True)

    # pin the torch-probed contract literals
    expected = [
        1.0,  # 0**0 == 1
        float("inf"),  # 0**-1
        float("inf"),  # 0**-2.5
        0.0,  # 0**2.5
        0.0,  # 0**inf
        -0.0,  # (-0)**3 keeps the sign
        0.0,  # (-0)**0.5 loses it
        1.0,  # (-0)**0
        1.0,  # 1**anything
        -8.0,  # (-2)**3
        4.0,  # (-2)**2
        float("nan"),  # (-2)**0.5
        -0.125,  # (-2)**-3
        1.0,  # anything**0
    ]
    for got, want in zip(tri.tolist(), expected):
        if math.isnan(want):
            assert math.isnan(got)
        else:
            assert got == want, f"{got!r} != {want!r}"


def test_float_power_nan_base_fallback():
    # nan is outside the triton domain; the wrapper delegates to aten, which
    # still honors pow(nan, 0) == 1.
    base = torch.tensor([float("nan"), float("nan")], dtype=torch.float64)
    exp = torch.tensor([0.0, 2.0], dtype=torch.float64)
    tri, ref = _inplace(base, exp)
    torch.testing.assert_close(tri, ref, rtol=0.0, atol=0.0, equal_nan=True)
    assert tri[0].item() == 1.0
    assert math.isnan(tri[1].item())


def test_float_power_integer_exponents_bitexact():
    bases = [2.0, 3.0, 5.0, 10.0, -2.0, -3.0, 1.0, 0.0]
    exps = list(range(0, 7))
    base = torch.tensor(bases * len(exps), dtype=torch.float64)
    exp = torch.tensor(exps * len(bases), dtype=torch.float64)
    tri, ref = _inplace(base, exp)
    assert torch.equal(tri, ref), "non-negative integer exponents must be bit-exact"


def test_float_power_negative_integer_exponents():
    # the kernel takes a float64 reciprocal (correctly rounded) while torch
    # goes through libm pow (<= 1 ulp off for some bases, e.g. 10**-5), so
    # allow a few ulps instead of claiming bit-equality
    bases = [2.0, 3.0, 5.0, 10.0, -2.0, -3.0, 1.0]
    exps = list(range(-6, 0))
    base = torch.tensor(bases * len(exps), dtype=torch.float64)
    exp = torch.tensor(exps * len(bases), dtype=torch.float64)
    tri, ref = _inplace(base, exp)
    torch.testing.assert_close(tri, ref, rtol=1e-14, atol=0.0)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_float_power_fractional(size):
    # tolerance source: the kernel computes exp(e * log(x)) in fp32 while
    # torch works in fp64 (measured <= 1.1e-6 relative error on this domain,
    # rtol=1e-4 gives ~100x headroom)
    torch.manual_seed(0)
    base = torch.randn(size, dtype=torch.float64).abs() * 4 + 0.25
    exp = torch.randn(size, dtype=torch.float64) * 3
    tri, ref = _inplace(base, exp)
    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


def test_float_power_int_inputs():
    # integer inputs are promoted to float64 (torch.float_power semantics)
    base = torch.tensor([2, 3, -2, 5], dtype=torch.int64)
    exp = torch.tensor([3, 2, 3, -2], dtype=torch.int64)
    ref = torch.float_power(base, exp)

    tri = base.to(torch.float64).clone()
    ret = float_power_tensor_tensor_(tri, exp)
    assert ret is tri
    assert torch.equal(tri, ref)


def test_float_power_inplace_non_contiguous():
    torch.manual_seed(0)
    base = torch.randn(8, 4, dtype=torch.float64).abs() + 0.5
    exp = torch.tensor([[2.0], [3.0], [0.5], [-1.0]], dtype=torch.float64)

    ref_view = base.t().clone()
    ref_view.float_power_(exp)

    tri_view = base.t()
    assert not tri_view.is_contiguous()
    ret = float_power_tensor_tensor_(tri_view, exp)

    assert ret is tri_view
    torch.testing.assert_close(tri_view, ref_view, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(base.t(), ref_view, rtol=1e-4, atol=1e-4)


def test_float_power_inplace_broadcast():
    torch.manual_seed(0)
    base = torch.randn(4, 8, dtype=torch.float64).abs() + 0.5
    exp = torch.tensor([[2.0], [3.0], [0.5], [-1.0]], dtype=torch.float64)
    tri, ref = _inplace(base, exp)
    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


def test_float_power_inplace_broadcast_grow_raises():
    base = torch.ones(4, 1, dtype=torch.float64)
    exp = torch.ones(4, 8, dtype=torch.float64)
    with pytest.raises(RuntimeError):
        float_power_tensor_tensor_(base, exp)


@pytest.mark.parametrize(
    "self_dtype",
    [torch.float32, torch.float16, torch.bfloat16, torch.int32, torch.int64],
)
def test_float_power_non_double_base_raises(self_dtype):
    # in-place float_power requires a float64 base (ints are promoted, so an
    # int self can never receive the Double result)
    base = torch.ones(8, dtype=self_dtype)
    exp = torch.ones(8, dtype=torch.float64)
    with pytest.raises(RuntimeError, match="the base given to float_power_ has dtype"):
        float_power_tensor_tensor_(base, exp)


def test_float_power_scalar_exponent_inplace():
    torch.manual_seed(0)
    base = torch.tensor([2.0, 3.0, -2.0, 0.5], dtype=torch.float64)

    ref = base.clone()
    ref.float_power_(2.0)

    tri = base.clone()
    ret = float_power_tensor_scalar_(tri, 2.0)
    assert ret is tri
    assert torch.equal(tri, ref)

    tri_int = base.clone()
    float_power_tensor_scalar_(tri_int, 3)
    assert torch.equal(tri_int, base**3)


def test_float_power_out_variants():
    base = torch.tensor([2.0, 3.0, -2.0], dtype=torch.float64)
    exp = torch.tensor([3.0, 2.0, 3.0], dtype=torch.float64)
    ref = torch.float_power(base, exp)

    out = torch.empty(3, dtype=torch.float64)
    ret = float_power_tensor_tensor_out(base, exp, out=out)
    assert ret is out
    assert torch.equal(out, ref)

    wrong = torch.empty(3, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="the output given to float_power has dtype"):
        float_power_tensor_tensor_out(base, exp, out=wrong)

    out_s = torch.empty(3, dtype=torch.float64)
    ret_s = float_power_tensor_scalar_out(base, 3.0, out=out_s)
    assert ret_s is out_s
    assert torch.equal(out_s, base**3.0)

    wrong_s = torch.empty(3, dtype=torch.float64)
    with pytest.raises(RuntimeError, match="Expected out tensor to have device"):
        float_power_tensor_scalar_out(base, 3.0, out=wrong_s.to("meta"))


def test_float_power_scalar_base_variants():
    exp = torch.tensor([3.0, 0.5, -1.0, 0.0], dtype=torch.float64)
    ref = torch.float_power(2.0, exp)

    out = float_power_scalar_tensor(2.0, exp)
    assert out.dtype == torch.float64
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)

    out_int = float_power_scalar_tensor(2, exp.to(torch.int64))
    assert torch.equal(out_int, torch.float_power(2, exp.to(torch.int64)))

    buf = torch.empty(4, dtype=torch.float64)
    ret = float_power_scalar_tensor_out(2.0, exp, out=buf)
    assert ret is buf
    torch.testing.assert_close(buf, ref, rtol=1e-4, atol=1e-4)


def test_float_power_non_inplace_entries():
    torch.manual_seed(0)
    base = torch.randn(512, dtype=torch.float64).abs() * 4 + 0.25
    exp = torch.randn(512, dtype=torch.float64) * 3
    ref = torch.float_power(base, exp)
    torch.testing.assert_close(
        float_power_tensor_tensor(base, exp), ref, rtol=1e-4, atol=1e-4
    )
    torch.testing.assert_close(
        float_power_tensor_scalar(base, 2.0), base**2, rtol=1e-4, atol=1e-4
    )

    ints = torch.arange(-5, 5, dtype=torch.int64)
    assert torch.equal(
        float_power_tensor_tensor(ints, torch.full_like(ints, 3)),
        torch.float_power(ints, torch.tensor(3)),
    )


def test_float_power_out_of_domain_fallback():
    # values the fp32-domain kernel cannot represent delegate to torch and
    # stay bit-identical to it
    base = torch.tensor([1e300, 1e-300, 2.0], dtype=torch.float64)
    exp = torch.tensor([2.0, 3.0, 1e60], dtype=torch.float64)
    tri, ref = _inplace(base, exp)
    torch.testing.assert_close(tri, ref, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("shape", [(0,), (0, 37)])
def test_float_power_empty(shape):
    base = torch.randn(shape, dtype=torch.float64).abs() + 0.5
    exp = torch.randn(shape, dtype=torch.float64)
    ret = float_power_tensor_tensor_(base, exp)
    assert ret is base
    assert base.shape == shape

    out = float_power_tensor_tensor(base, exp)
    assert out.shape == shape


def test_float_power_triton_path(monkeypatch):
    # Smoke: the in-place float64 route must launch the Triton kernel.
    import importlib

    mod = importlib.import_module("flaggems.float_power_")

    launches = []
    real_run = mod.float_power_kernel_tt.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod.float_power_kernel_tt, "run", spy_run)
    base = torch.tensor([2.0, 3.0, -2.0, 0.5], dtype=torch.float64)
    exp = torch.tensor([3.0, 2.0, 3.0, -1.0], dtype=torch.float64)
    ref = base.clone()
    ref.float_power_(exp)

    ret = mod.float_power_tensor_tensor_(base, exp)
    assert len(launches) == 1, "float_power_ must go through the Triton kernel"
    assert ret is base
    assert torch.equal(base, ref)
