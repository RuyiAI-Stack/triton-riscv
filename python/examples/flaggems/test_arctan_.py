import pytest
import torch

from .arctan_ import arctan_

# Tolerances (source: the kernel always computes atan in fp32 via
# libdevice.atan, like upstream arctan_func (tl_extra_shim.atan has no fp64
# entry) and the repo atan.py kernel, then casts back to the storage dtype;
# torch's CPU in-place arctan also computes in fp32 opmath):
#   fp32: rtol=atol=1e-4, repo test_atan.py / test_asinh_.py convention.
#   fp16: rtol=atol=1e-3, fp16 eps ~ 9.8e-4; both sides round fp32 -> fp16.
#   fp64: rtol=atol=1e-5; the fp32-computed result tracks torch's fp64
#         reference to ~1e-7 on x86_64, but the riscv64 CI's atan lowering
#         measured ~3.7e-6 relative error (2026-11-28), so the pin keeps
#         headroom for the platform gap.


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_arctan_inplace(size):
    torch.manual_seed(0)
    x = torch.randn(size, dtype=torch.float32, device="cpu")
    ref = x.clone()
    torch.arctan_(ref)

    ret = arctan_(x)

    assert ret is x
    torch.testing.assert_close(x, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "dtype,rtol,atol",
    [(torch.float16, 1e-3, 1e-3), (torch.float64, 1e-5, 1e-5)],
)
def test_arctan_inplace_other_float(dtype, rtol, atol):
    torch.manual_seed(0)
    x = torch.randn(128, dtype=dtype, device="cpu")
    ref = x.clone()
    torch.arctan_(ref)

    ret = arctan_(x)

    assert ret is x
    torch.testing.assert_close(x, ref, rtol=rtol, atol=atol)


def test_arctan_inplace_non_contiguous():
    torch.manual_seed(0)
    x = torch.randn(32, 64, dtype=torch.float32, device="cpu")
    x_sliced = x[::2, ::2]  # non-contiguous
    ref = x_sliced.clone()
    torch.arctan_(ref)

    # .t() keeps the view genuinely non-contiguous (a .clone() of a
    # strided view re-packs contiguous, which would only exercise
    # the fast path).
    tri = x_sliced.t()
    assert not tri.is_contiguous()

    out = arctan_(tri)

    assert out is tri
    torch.testing.assert_close(tri, ref.t(), rtol=1e-4, atol=1e-4)


def test_arctan_inplace_empty():
    x = torch.empty(0, dtype=torch.float32, device="cpu")

    ret = arctan_(x)

    assert ret is x
    assert x.numel() == 0


def test_arctan_inplace_boundaries():
    # arctan(+-inf) = +-pi/2, arctan(+-0) = +-0, arctan(nan) = nan.
    x = torch.tensor(
        [float("inf"), float("-inf"), float("nan"), 0.0, -0.0, 1.0, -1.0],
        dtype=torch.float32,
        device="cpu",
    )
    ref = x.clone()
    torch.arctan_(ref)

    ret = arctan_(x)

    assert ret is x
    torch.testing.assert_close(x, ref, rtol=1e-6, atol=1e-6, equal_nan=True)


def test_arctan_inplace_integer_falls_back_to_aten():
    # Non-float dtypes leave the Triton path (repo convention, cf. asinh_ /
    # acosh_) and delegate to torch.ops.aten.arctan_, which rejects in-place
    # arctan on integer tensors ("result type Float can't be cast ...").
    x = torch.randint(-9, 9, (16,), dtype=torch.int64, device="cpu")

    with pytest.raises(RuntimeError, match="can't be cast"):
        arctan_(x)
