import pytest
import torch

from .trunc_ import trunc, trunc_, trunc_out


def _bitview(t):
    if t.dtype == torch.float16:
        return t.view(torch.int16)
    if t.dtype == torch.bfloat16:
        return t.view(torch.int16)
    if t.dtype == torch.float32:
        return t.view(torch.int32)
    return t.view(torch.int64)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_trunc_inplace(size):
    torch.manual_seed(0)
    # Fractional values with mixed signs: trunc rounds toward zero, which
    # only differs from floor on the negative side.  +-inf and a huge
    # magnitude pin the IEEE edge cases (trunc is the identity on them).
    x = torch.randn(size, device="cpu", dtype=torch.float32) * 5.0
    x[0] = float("inf")
    x[1] = float("-inf")
    x[2] = 1e30

    ref = torch.trunc(x)

    tri = x.clone()
    out = trunc_(tri)

    assert out is tri, "trunc_ should return the same tensor"
    # trunc is exact for every float dtype, so compare the raw bit patterns
    # (this also pins the sign of -0.0 from trunc(-0.5)).
    assert torch.equal(_bitview(tri), _bitview(ref))


def test_trunc_inplace_non_contiguous():
    torch.manual_seed(0)
    x = torch.randn(32, 64, device="cpu", dtype=torch.float32) * 5.0
    x_sliced = x[::2, ::2]  # non-contiguous

    ref = torch.trunc(x_sliced)

    # .t() keeps the view genuinely non-contiguous (a .clone() of a
    # strided view re-packs contiguous, which would only exercise
    # the fast path).
    tri = x_sliced.t()
    assert not tri.is_contiguous()
    out = trunc_(tri)

    assert out is tri
    assert torch.equal(_bitview(tri), _bitview(ref.t()))


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64, torch.bfloat16])
def test_trunc_inplace_other_float(dtype):
    torch.manual_seed(0)
    x = (torch.randn(128, device="cpu", dtype=torch.float32) * 5.0).to(dtype)

    ref = torch.trunc(x)

    tri = x.clone()
    out = trunc_(tri)

    assert out is tri
    assert torch.equal(_bitview(tri), _bitview(ref))


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_trunc_inplace_int_aten_fallback(dtype, monkeypatch):
    # Integer trunc is the identity; the wrapper routes ints to
    # torch.ops.aten.trunc_ (the fix_ precedent).  Spy on the aten op to
    # assert the fallback is the code path actually taken.
    real = torch.ops.aten.trunc_
    calls = []

    def spy(t):
        calls.append(1)
        return real(t)

    monkeypatch.setattr(torch.ops.aten, "trunc_", spy)

    if dtype == torch.int64:
        x = torch.tensor([-7, -1, 0, 3, 1024, -(2 ** 40)], dtype=dtype)
    else:
        x = torch.tensor([-7, -1, 0, 3, 1024, -65000], dtype=dtype)
    ref = torch.trunc(x)

    tri = x.clone()
    out = trunc_(tri)

    assert out is tri
    assert calls, "expected the int path to call torch.ops.aten.trunc_"
    assert torch.equal(tri, ref)


@pytest.mark.parametrize("size", [512, 1023])
def test_trunc_out_of_place(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32) * 5.0
    x[-1] = -0.5  # trunc(-0.5) == -0.0: pin the sign bit through the kernel
    x_snapshot = x.clone()

    ref = torch.trunc(x)
    tri = trunc(x)

    assert tri is not x
    assert torch.equal(x, x_snapshot), "out-of-place trunc must not modify the input"
    assert torch.equal(_bitview(tri), _bitview(ref))


def test_trunc_out_variant():
    torch.manual_seed(0)
    x = (torch.randn(256, device="cpu", dtype=torch.float32) * 5.0)

    ref = torch.trunc(x)

    out = torch.empty_like(x)
    res = trunc_out(x, out=out)

    assert res is out
    assert torch.equal(_bitview(out), _bitview(ref))

    # out=None falls back to allocating a fresh result tensor.
    fresh = trunc_out(x)
    assert fresh is not x
    assert torch.equal(_bitview(fresh), _bitview(ref))


@pytest.mark.parametrize("shape", [(0,), (3, 0)])
def test_trunc_inplace_empty(shape):
    x = torch.randn(shape, device="cpu", dtype=torch.float32)
    out = trunc_(x)
    assert out is x
    assert x.numel() == 0
