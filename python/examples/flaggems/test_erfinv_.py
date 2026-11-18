import math

import pytest
import torch

from .erfinv_ import erfinv, erfinv_, erfinv_out


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_erfinv_inplace(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32).clamp(-0.999999, 0.999999)

    ref = torch.erfinv(x)

    tri = x.clone()
    out = erfinv_(tri)

    assert out is tri, "erfinv_ should return the same tensor"
    # rtol/atol source: the kernel runs the Newton refinement in fp64 and
    # narrows to fp32 at the end; measured max relative error vs
    # torch.erfinv over dense and random sweeps is 1.2e-7 (this file's
    # development probe), 1.3e-6 gives an order-of-magnitude margin.
    torch.testing.assert_close(tri, ref, rtol=1.3e-6, atol=1.3e-6)


def test_erfinv_inplace_boundary():
    x = torch.tensor(
        [
            1.0,  # -> exactly +inf
            -1.0,  # -> exactly -inf
            2.0,  # out of domain -> NaN
            -1.5,  # out of domain -> NaN
            float("inf"),  # -> NaN (|x| > 1)
            float("-inf"),  # -> NaN
            float("nan"),  # -> NaN
            0.0,  # -> exactly 0
            -0.0,  # -> exactly -0 (erfinv is odd)
            0.99999994,  # 1 ulp below 1.0: extreme tail, stays finite
            -0.99999994,
        ],
        dtype=torch.float32,
    )

    ref = torch.erfinv(x)

    tri = x.clone()
    out = erfinv_(tri)

    assert out is tri
    assert tri[0].item() == float("inf")
    assert tri[1].item() == float("-inf")
    for i in range(2, 7):  # out-of-domain, +-inf and NaN inputs -> NaN
        assert torch.isnan(tri[i]), f"element {i} should be NaN"
    assert tri[7].item() == 0.0
    assert tri[8].item() == 0.0
    assert math.copysign(1.0, tri[8].item()) == -1.0, "erfinv_(-0.0) must be -0.0"
    torch.testing.assert_close(tri[9:11], ref[9:11], rtol=1.3e-6, atol=1.3e-6)


def test_erfinv_inplace_non_contiguous():
    torch.manual_seed(0)
    x = torch.randn(32, 64, device="cpu", dtype=torch.float32).clamp(
        -0.999999, 0.999999
    )
    x_sliced = x[::2, ::2]  # non-contiguous

    ref = torch.erfinv(x_sliced)

    tri = x_sliced.t()
    assert not tri.is_contiguous()
    out = erfinv_(tri)

    assert out is tri
    torch.testing.assert_close(tri, ref.t(), rtol=1.3e-6, atol=1.3e-6)


def test_erfinv_inplace_fp16():
    torch.manual_seed(0)
    # Tolerance source: repo precedent test_erf.py fp16 (rtol=1e-2,
    # atol=1e-3); the kernel computes in fp64 internally and narrows once
    # to fp16, so the only extra error vs the fp32 path is the final
    # fp16 quantisation (1 ulp ~ 1e-3 near |erfinv| ~ 4).
    x = (torch.rand(128, device="cpu", dtype=torch.float32) * 1.9 - 0.95).to(
        torch.float16
    )

    ref = torch.erfinv(x)

    tri = x.clone()
    out = erfinv_(tri)

    assert out is tri
    torch.testing.assert_close(tri, ref, rtol=1e-2, atol=1e-3)


def test_erfinv_inplace_fp64_aten_fallback(monkeypatch):
    # fp64 goes through torch.ops.aten.erfinv_ (full double precision,
    # matching torch.erfinv bit-for-bit).  Spy on the aten op to assert
    # the fallback is the code path actually taken.
    real = torch.ops.aten.erfinv_
    calls = []

    def spy(t):
        calls.append(1)
        return real(t)

    monkeypatch.setattr(torch.ops.aten, "erfinv_", spy)

    torch.manual_seed(0)
    x = torch.randn(128, device="cpu", dtype=torch.float64).clamp(
        -0.999999, 0.999999
    )

    ref = torch.erfinv(x)

    tri = x.clone()
    out = erfinv_(tri)

    assert out is tri
    assert calls, "expected the fp64 path to call torch.ops.aten.erfinv_"
    torch.testing.assert_close(tri, ref, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("size", [512, 1023])
def test_erfinv_out_of_place(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32).clamp(-0.999999, 0.999999)
    x_snapshot = x.clone()

    ref = torch.erfinv(x)
    tri = erfinv(x)

    assert tri is not x
    assert torch.equal(x, x_snapshot), "out-of-place erfinv must not modify the input"
    torch.testing.assert_close(tri, ref, rtol=1.3e-6, atol=1.3e-6)


def test_erfinv_out_variant():
    torch.manual_seed(0)
    x = torch.randn(256, device="cpu", dtype=torch.float32).clamp(-0.999999, 0.999999)

    ref = torch.erfinv(x)

    out = torch.empty_like(x)
    res = erfinv_out(x, out=out)

    assert res is out
    torch.testing.assert_close(out, ref, rtol=1.3e-6, atol=1.3e-6)

    fresh = erfinv_out(x)
    assert fresh is not x
    torch.testing.assert_close(fresh, ref, rtol=1.3e-6, atol=1.3e-6)


@pytest.mark.parametrize("shape", [(0,), (3, 0)])
def test_erfinv_inplace_empty(shape):
    x = torch.randn(shape, device="cpu", dtype=torch.float32)
    out = erfinv_(x)
    assert out is x
    assert x.numel() == 0
