import math
import sys

import pytest
import torch

from ._masked_scale import _masked_scale

# Semantics (upstream flag_gems/ops/_masked_scale.py + aten, probed):
# out = where(mask != 0, input * scale, 0.0). input must be floating point
# (ValueError otherwise); output dtype == input dtype. mask may be any numeric
# dtype (bool/uint8/int), compared against 0, broadcast to input's shape.
# scale is a Python scalar (aten schema: Scalar). aten._masked_scale has no
# CPU kernel (CUDA/Meta only, probed NotImplementedError), so references are
# composed with torch.where using the exact formula from the upstream FlagGems
# test suite (tests/test_masked_scale.py).
#
# Tolerance: every scale below (2.0, 0.5, -1.25, 8.0, 0.0) is exactly
# representable in fp16/fp32/fp64. fp16/bf16 mantissa products fit exactly in
# fp32, so the kernel (compute in fp32 via the fp32 scale arg, round once at
# the store) and torch (same opmath) are correctly rounded alike -> bit-exact;
# fp32/fp64 multiply once in their own precision -> bit-exact. All float
# comparisons therefore use rtol=0, atol=0.

# The op module object, captured at collection time. pytest 9.2 evicts the
# flaggens package from sys.modules after collection, and this environment's
# FileFinder cache for python/examples nondeterministically forgets the
# package (probed: 'flaggens' absent from FileFinder._path_cache while
# present on disk), which breaks re-imports by name at test runtime.
_MASKED_SCALE_MOD = sys.modules[__name__.rsplit(".", 1)[0] + "._masked_scale"]


def _spy_module():
    """Fetch the op module via importlib (S1 rule: package __init__ exports
    shadow submodule attributes). Re-seed sys.modules from the object the
    collection-time import kept alive before the mandated import_module call
    (a direct sys.modules hit needs no parent lookup)."""
    import importlib

    name = "flaggens._masked_scale"
    if name not in sys.modules:
        sys.modules[name] = _MASKED_SCALE_MOD
    return importlib.import_module(name)


def _ref(input, mask, scale):
    return torch.where(
        mask.to(torch.bool), input * scale, torch.zeros((), dtype=input.dtype)
    )


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_masked_scale_float32(size):
    torch.manual_seed(0)
    x = torch.randn(size, dtype=torch.float32)
    mask = torch.randint(0, 2, (size,), dtype=torch.uint8)

    tri = _masked_scale(x, mask, 2.0)

    assert tri.dtype == torch.float32
    torch.testing.assert_close(tri, _ref(x, mask, 2.0), rtol=0, atol=0)


@pytest.mark.parametrize("size", [512, 1023])
def test_masked_scale_bool_mask(size):
    torch.manual_seed(0)
    x = torch.randn(size, dtype=torch.float32)
    mask = torch.rand(size) < 0.5

    tri = _masked_scale(x, mask, 0.5)

    assert tri.dtype == torch.float32
    torch.testing.assert_close(tri, _ref(x, mask, 0.5), rtol=0, atol=0)


def test_masked_scale_int_mask_nonzero_is_true():
    # kernel semantic: mask != 0, so any nonzero int value counts as true
    torch.manual_seed(0)
    x = torch.randn(128, dtype=torch.float32)
    mask = torch.tensor([5, 0, -3, 2] * 32, dtype=torch.int32)

    torch.testing.assert_close(
        _masked_scale(x, mask, 2.0), _ref(x, mask, 2.0), rtol=0, atol=0
    )


@pytest.mark.parametrize("scale", [2.0, 0.5, -1.25, 8.0, 2])
def test_masked_scale_scale_matrix(scale):
    torch.manual_seed(0)
    x = torch.randn(1024, dtype=torch.float32)
    mask = torch.rand(1024) < 0.5

    torch.testing.assert_close(
        _masked_scale(x, mask, scale), _ref(x, mask, scale), rtol=0, atol=0
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float64])
def test_masked_scale_dtype_matrix(dtype):
    torch.manual_seed(0)
    x = torch.randn(1024, dtype=torch.float32).to(dtype)
    mask = torch.randint(0, 2, (1024,), dtype=torch.uint8)

    tri = _masked_scale(x, mask, 2.0)

    assert tri.dtype == dtype
    torch.testing.assert_close(tri, _ref(x, mask, 2.0), rtol=0, atol=0)


def test_masked_scale_boundary_values():
    # masked-out positions collapse to +0.0 regardless of the input value
    # (the softmax-masking purpose of this op); masked-in propagate inf/NaN
    x = torch.tensor(
        [1.0, float("-inf"), float("inf"), float("nan"), -0.0, 2.0, float("nan")]
    )
    mask = torch.tensor([1, 0, 0, 0, 0, 1, 1], dtype=torch.uint8)

    tri = _masked_scale(x, mask, 2.0)

    assert tri[0].item() == 2.0
    for i in range(1, 5):
        assert tri[i].item() == 0.0, f"masked-out element {i} must be 0.0"
        assert (
            math.copysign(1.0, tri[i].item()) == 1.0
        ), f"masked-out element {i} must be +0.0, got {tri[i].item()!r}"
    assert tri[5].item() == 4.0
    assert torch.isnan(tri[6]), "masked-in NaN must propagate"
    torch.testing.assert_close(tri[:1], _ref(x, mask, 2.0)[:1], rtol=0, atol=0)
    torch.testing.assert_close(tri[5:6], _ref(x, mask, 2.0)[5:6], rtol=0, atol=0)


def test_masked_scale_zero_scale_signed_zero():
    # x * 0.0 keeps the sign of x (IEEE): -3.0 -> -0.0, 5.0 -> +0.0
    x = torch.tensor([-3.0, 5.0, -2.0])
    mask = torch.tensor([1, 1, 0], dtype=torch.uint8)

    tri = _masked_scale(x, mask, 0.0)

    ref = _ref(x, mask, 0.0)
    torch.testing.assert_close(tri, ref, rtol=0, atol=0)
    assert math.copysign(1.0, tri[0].item()) == -1.0
    assert math.copysign(1.0, tri[1].item()) == 1.0
    assert tri[2].item() == 0.0


def test_masked_scale_2d_broadcast_mask():
    torch.manual_seed(0)
    x = torch.randn(8, 64, dtype=torch.float32)
    mask = (torch.rand(8, 1) < 0.5).to(torch.uint8)

    tri = _masked_scale(x, mask, 2.0)

    assert tri.shape == x.shape
    torch.testing.assert_close(tri, _ref(x, mask, 2.0), rtol=0, atol=0)


def test_masked_scale_non_contiguous():
    torch.manual_seed(0)
    x = torch.randn(32, 64, dtype=torch.float32)
    mask = torch.randint(0, 2, (32, 64), dtype=torch.uint8)
    x_t = x.t()
    mask_t = mask.t()
    assert not x_t.is_contiguous()

    tri = _masked_scale(x_t, mask_t, 2.0)

    assert tri.shape == x_t.shape
    torch.testing.assert_close(tri, _ref(x_t, mask_t, 2.0), rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(0,), (3, 0)])
def test_masked_scale_empty(shape):
    x = torch.randn(shape, dtype=torch.float32)
    mask = (torch.rand(shape) < 0.5).to(torch.uint8)

    tri = _masked_scale(x, mask, 2.0)

    assert tri.shape == x.shape
    assert tri.dtype == x.dtype
    assert tri.numel() == 0


def test_masked_scale_zero_dim():
    x = torch.tensor(3.5)
    mask = torch.tensor(True)

    tri = _masked_scale(x, mask, 2.0)

    assert tri.shape == x.shape
    torch.testing.assert_close(tri, _ref(x, mask, 2.0), rtol=0, atol=0)


def test_masked_scale_int_input_raises():
    x = torch.randint(-5, 5, (64,), dtype=torch.int32)
    mask = torch.rand(64) < 0.5
    with pytest.raises(ValueError, match="Only floating-point dtype"):
        _masked_scale(x, mask, 2.0)


def test_masked_scale_incompatible_mask_raises():
    x = torch.randn(8, dtype=torch.float32)
    mask = (torch.rand(7) < 0.5).to(torch.uint8)
    with pytest.raises(RuntimeError):
        _masked_scale(x, mask, 2.0)


def test_masked_scale_input_not_modified():
    torch.manual_seed(0)
    x = torch.randn(1024, dtype=torch.float32)
    mask = torch.randint(0, 2, (1024,), dtype=torch.uint8)
    snap = x.clone()

    _masked_scale(x, mask, 2.0)

    assert torch.equal(x, snap), "out-of-place _masked_scale must not touch input"


def test_masked_scale_triton_path(monkeypatch):
    # Smoke: the main path must launch the Triton kernel.
    mod = _spy_module()

    launches = []
    real_run = mod._masked_scale_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._masked_scale_kernel, "run", spy_run)
    x = torch.randn(1024, dtype=torch.float32)
    mask = torch.randint(0, 2, (1024,), dtype=torch.uint8)
    out = mod._masked_scale(x, mask, 2.0)
    assert len(launches) == 1, "_masked_scale must go through the Triton kernel"
    assert launches[0] == (1,)
    torch.testing.assert_close(out, _ref(x, mask, 2.0), rtol=0, atol=0)
