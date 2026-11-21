import pytest
import torch

from .clamp_max import clamp_max, clamp_max_, clamp_max_out


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_clamp_max_scalar(size):
    torch.manual_seed(0)
    # Values below, above and exactly at the bound; NaN input must
    # propagate (torch.clamp_max semantics).
    x = torch.randn(size, device="cpu", dtype=torch.float32) * 2.0
    x[0] = 0.5  # exactly equal to max
    x[1] = 0.5000001  # just above
    x[2] = float("nan")

    ref = torch.clamp_max(x, max=0.5)

    tri = clamp_max(x, 0.5)

    # minimum is exact, so assert_close with equal_nan pins values
    # (default tolerances are trivially met).
    torch.testing.assert_close(tri, ref, equal_nan=True)


def test_clamp_max_scalar_exact():
    # No NaN in this tensor: pin bit-exact equality, including the
    # "input == max" and +-inf boundaries.
    x = torch.tensor(
        [-2.0, -0.5, 0.5, 0.5000001, 2.0, float("inf"), float("-inf")],
        dtype=torch.float32,
    )
    ref = torch.clamp_max(x, max=0.5)
    tri = clamp_max(x, 0.5)
    assert torch.equal(tri, ref)


def test_clamp_max_scalar_nan_bound():
    # A NaN scalar bound clamps every element to NaN, like torch.
    x = torch.tensor([-2.0, 0.7, 3.0], dtype=torch.float32)
    ref = torch.clamp_max(x, max=float("nan"))
    tri = clamp_max(x, float("nan"))
    torch.testing.assert_close(tri, ref, equal_nan=True)
    assert torch.isnan(tri).all()


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_clamp_max_scalar_int(dtype):
    # Integer clamp is bit-exact through the Triton integer minimum.
    if dtype == torch.int64:
        x = torch.tensor([-5, 0, 3, 1024, -(2**40), 2**40], dtype=dtype)
    else:
        x = torch.tensor([-5, 0, 3, 1024, -65000, 65000], dtype=dtype)

    ref = torch.clamp_max(x, max=0)

    tri = clamp_max(x, 0)

    assert torch.equal(tri, ref)


def test_clamp_max_tensor():
    torch.manual_seed(0)
    x = torch.randn(16, 256, dtype=torch.float32, device="cpu") * 2.0
    maxi = torch.rand(16, 256, dtype=torch.float32, device="cpu") * 2.0 - 1.0
    maxi[0, 0] = float("nan")  # NaN bound at one position -> NaN there
    x[0, 1] = float("nan")  # NaN input propagates

    ref = torch.clamp_max(x, max=maxi)

    tri = clamp_max(x, maxi)

    torch.testing.assert_close(tri, ref, equal_nan=True)


def test_clamp_max_tensor_broadcast():
    torch.manual_seed(0)
    x = torch.randn(16, 256, dtype=torch.float32, device="cpu") * 2.0
    maxi = (
        torch.rand(16, 1, dtype=torch.float32, device="cpu") * 2.0 - 1.0
    )  # broadcasts along rows

    ref = torch.clamp_max(x, max=maxi)

    tri = clamp_max(x, maxi)

    torch.testing.assert_close(tri, ref, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float64])
def test_clamp_max_scalar_dtype(dtype):
    torch.manual_seed(0)
    x = (torch.randn(128, dtype=torch.float32) * 2.0).to(dtype)

    ref = torch.clamp_max(x, max=0.5)

    tri = clamp_max(x, 0.5)

    # minimum is exact in every dtype (fp16/bf16 compute in fp32 and
    # narrow once, also exact), so bit-exact equality holds.
    assert torch.equal(tri, ref)


def test_clamp_max_non_contiguous():
    torch.manual_seed(0)
    x = torch.randn(32, 64, dtype=torch.float32, device="cpu") * 2.0
    x_sliced = x[::2, ::2]

    ref = torch.clamp_max(x_sliced, max=0.25)

    tri_view = x_sliced.t()
    assert not tri_view.is_contiguous()
    tri = clamp_max(tri_view, 0.25)

    torch.testing.assert_close(tri, ref.t(), rtol=1e-6, atol=1e-6)


def test_clamp_max_inplace():
    torch.manual_seed(0)
    x = torch.randn(16, 256, dtype=torch.float32, device="cpu") * 2.0
    x[0, 0] = 0.5  # exactly at the bound
    x_ref = x.clone()
    x_ref.clamp_max_(0.5)

    out = clamp_max_(x, 0.5)

    assert out is x, "clamp_max_ should return the same tensor"
    torch.testing.assert_close(x, x_ref, rtol=1e-6, atol=1e-6)


def test_clamp_max_inplace_non_contiguous():
    torch.manual_seed(0)
    x = torch.randn(32, 64, dtype=torch.float32, device="cpu") * 2.0
    x_sliced = x[::2, ::2]

    ref = x_sliced.clone()
    ref.clamp_max_(0.25)

    tri = x_sliced.t()
    assert not tri.is_contiguous()
    out = clamp_max_(tri, 0.25)

    assert out is tri
    # The caller's storage behind the view must be rewritten.
    torch.testing.assert_close(tri, ref.t(), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(x_sliced, ref, rtol=1e-6, atol=1e-6)


def test_clamp_max_inplace_int():
    x = torch.tensor([-5, 0, 3, 1024], dtype=torch.int64)

    ref = x.clone()
    ref.clamp_max_(2)

    out = clamp_max_(x, 2)

    assert out is x
    assert torch.equal(x, ref)


def test_clamp_max_out_variant():
    torch.manual_seed(0)
    x = torch.randn(256, dtype=torch.float32) * 2.0
    maxi = torch.rand(256, dtype=torch.float32) * 2.0 - 1.0

    ref = torch.clamp_max(x, max=maxi)

    out = torch.empty_like(x)
    res = clamp_max_out(x, maxi, out=out)

    assert res is out
    torch.testing.assert_close(out, ref, rtol=1e-6, atol=1e-6)

    out2 = torch.empty_like(x)
    res2 = clamp_max_out(x, 0.5, out=out2)
    assert res2 is out2
    torch.testing.assert_close(out2, torch.clamp_max(x, max=0.5), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("shape", [(0,), (3, 0)])
def test_clamp_max_empty(shape):
    x = torch.randn(shape, dtype=torch.float32)
    tri = clamp_max(x, 0.5)
    assert tri.shape == x.shape

    maxi = torch.randn(shape, dtype=torch.float32)
    tri_t = clamp_max(x, maxi)
    assert tri_t.shape == x.shape
