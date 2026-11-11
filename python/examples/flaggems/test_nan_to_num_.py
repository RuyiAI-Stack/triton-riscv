import pytest
import torch

from .nan_to_num_ import nan_to_num_


def _make_input(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32) * 3.0
    x[0::8] = float("nan")
    x[1::8] = float("inf")
    x[2::8] = float("-inf")
    return x


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_nan_to_num_inplace(size):
    x = _make_input(size)

    ref = torch.nan_to_num(x.clone())

    tri = x.clone()
    out = nan_to_num_(tri)

    assert out is tri, "nan_to_num_ should return the same tensor"
    torch.testing.assert_close(tri, ref)


def test_nan_to_num_inplace_non_contiguous():
    torch.manual_seed(0)
    x = torch.randn(32, 64, device="cpu", dtype=torch.float32) * 3.0
    x_sliced = x[::2, ::2]  # non-contiguous
    x_sliced[0, 0] = float("nan")
    x_sliced[1, 1] = float("inf")
    x_sliced[2, 2] = float("-inf")

    ref = torch.nan_to_num(x_sliced.clone())

    # .t() keeps the view genuinely non-contiguous (a .clone() of a strided
    # view re-packs contiguous, which would only exercise the fast path).
    tri = x_sliced.t()
    assert not tri.is_contiguous()
    out = nan_to_num_(tri)

    assert out is tri
    torch.testing.assert_close(tri, ref.t())
    assert not (tri != tri).any()  # the caller's storage was actually cleaned


@pytest.mark.parametrize("dtype", [torch.float16])
def test_nan_to_num_inplace_fp16(dtype):
    torch.manual_seed(0)
    x = (torch.randn(256, device="cpu", dtype=torch.float32) * 3.0).to(dtype)
    x.view(-1)[0::8] = float("nan")
    x.view(-1)[1::8] = float("inf")
    x.view(-1)[2::8] = float("-inf")

    ref = torch.nan_to_num(x.clone())

    tri = x.clone()
    out = nan_to_num_(tri)

    assert out is tri
    torch.testing.assert_close(tri, ref)


def test_nan_to_num_inplace_fp64_fallback():
    # fp64 exceeds the fp32 kernel-scalar range, so the wrapper delegates to
    # the aten op; this test pins that fallback.
    torch.manual_seed(0)
    x = torch.randn(256, device="cpu", dtype=torch.float64) * 3.0
    x[0::8] = float("nan")
    x[1::8] = float("inf")
    x[2::8] = float("-inf")

    ref = torch.nan_to_num(x.clone())

    tri = x.clone()
    out = nan_to_num_(tri)

    assert out is tri
    torch.testing.assert_close(tri, ref)


BF16_XFAIL = pytest.mark.xfail(
    strict=True,
    reason="backend: fp32 to bfloat16 conversion miscompiles (RuyiAI-Stack/triton-riscv#114)",
)


@BF16_XFAIL
def test_nan_to_num_bf16():
    torch.manual_seed(0)
    x = (torch.randn(256, device="cpu", dtype=torch.float32) * 3.0).to(torch.bfloat16)
    x.view(-1)[0::8] = float("nan")
    x.view(-1)[1::8] = float("inf")
    x.view(-1)[2::8] = float("-inf")

    ref = torch.nan_to_num(x)

    tri = x.clone()
    out = nan_to_num_(tri)

    assert out is tri
    torch.testing.assert_close(tri, ref, rtol=1e-2, atol=1e-2)


def test_nan_to_num_kwargs_respected():
    # nan/posinf/neginf must reach the computation (aten fallback).
    t = torch.tensor([float("nan"), 1.0, float("inf"), float("-inf")])
    r = nan_to_num_(t.clone(), nan=5.0)
    # aten defaults: posinf/neginf None -> largest finite values.
    ref = torch.nan_to_num(t, nan=5.0)
    torch.testing.assert_close(r, ref)


def test_nan_to_num_posinf_neginf_kwargs():
    t = torch.tensor([float("nan"), float("inf"), float("-inf"), 2.0])
    r = nan_to_num_(t.clone(), nan=1.0, posinf=99.0, neginf=-99.0)
    torch.testing.assert_close(r, torch.tensor([1.0, 99.0, -99.0, 2.0]))


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_nan_to_num_inplace_integer_noop(dtype):
    # aten semantics: integer inputs have no NaN/inf, so nan_to_num_ is a no-op.
    x = torch.tensor([1, 2, 3], dtype=dtype)
    out = nan_to_num_(x)
    assert out is x
    torch.testing.assert_close(x, torch.tensor([1, 2, 3], dtype=dtype))
