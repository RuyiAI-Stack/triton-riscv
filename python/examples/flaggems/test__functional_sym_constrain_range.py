import sys

import pytest
import torch

from ._functional_sym_constrain_range import _functional_sym_constrain_range

# Semantics (upstream flag_gems/ops/_functional_sym_constrain_range.py + aten,
# probed on this torch 2.15 CPU build): the symbolic range constraint itself
# is recorded by torch's symbolic tracing, not at runtime. aten's real schema
# is (Scalar size, int? min, int? max, Tensor dep_token); its CPU kernel
# VALIDATES the range (RuntimeError "Invalid value range ...") and returns
# dep_token.clone() (probed: equal values, fresh object). The FlagGems port
# skips validation entirely (deferred to tracing) and returns an identity
# COPY of the first tensor argument: contiguous non-empty -> Triton copy
# kernel into empty_like; non-contiguous/empty -> host clone. No tensor in
# args/kwargs -> args[0] (or None when there are no args at all).

# The op module object, captured at collection time. pytest 9.2 evicts the
# flaggens package from sys.modules after collection, and this environment's
# FileFinder cache for python/examples nondeterministically forgets the
# package (probed: 'flaggens' absent from FileFinder._path_cache while
# present on disk), which breaks re-imports by name at test runtime.
_SCR_MOD = sys.modules[__name__.rsplit(".", 1)[0] + "._functional_sym_constrain_range"]


def _spy_module():
    """Fetch the op module via importlib (S1 rule: package __init__ exports
    shadow submodule attributes). Re-seed sys.modules from the object the
    collection-time import kept alive before the mandated import_module call
    (a direct sys.modules hit needs no parent lookup)."""
    import importlib

    name = "flaggens._functional_sym_constrain_range"
    if name not in sys.modules:
        sys.modules[name] = _SCR_MOD
    return importlib.import_module(name)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_identity_copy_float32(size):
    torch.manual_seed(0)
    dep_token = torch.randn(size, dtype=torch.float32)
    snap = dep_token.clone()

    out = _functional_sym_constrain_range(5, 1, 10, dep_token)

    assert out is not dep_token, "must return a fresh tensor, not the input"
    assert out.data_ptr() != dep_token.data_ptr()
    assert out.dtype == dep_token.dtype
    assert out.shape == dep_token.shape
    torch.testing.assert_close(out, snap, rtol=0, atol=0)
    assert torch.equal(dep_token, snap), "input must not be modified"


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.float64])
def test_identity_copy_bit_exact(dtype):
    torch.manual_seed(0)
    if dtype.is_floating_point:
        dep_token = torch.randn(512, dtype=dtype)
    else:
        dep_token = torch.randint(-1000, 1000, (512,), dtype=dtype)

    out = _functional_sym_constrain_range(5, 1, 10, dep_token)

    torch.testing.assert_close(out, dep_token, rtol=0, atol=0)


def test_tensor_found_in_kwargs():
    torch.manual_seed(0)
    dep_token = torch.randn(128, dtype=torch.float32)

    out = _functional_sym_constrain_range(5, min=1, max=10, dep_token=dep_token)

    assert out is not dep_token
    torch.testing.assert_close(out, dep_token, rtol=0, atol=0)


@pytest.mark.parametrize(
    "min_val, max_val", [(1, 10), (None, None), (None, 5), (-5, None)]
)
def test_aten_parity_in_range(min_val, max_val):
    # aten's CPU kernel returns dep_token.clone() for in-range sizes; the
    # port's copy must match it bit-for-bit (integers and floats).
    torch.manual_seed(0)
    dep_token = torch.randn(256, dtype=torch.float32)

    ref = torch.ops.aten._functional_sym_constrain_range(5, min_val, max_val, dep_token)
    out = _functional_sym_constrain_range(5, min_val, max_val, dep_token)

    assert out.dtype == ref.dtype and out.shape == ref.shape
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_no_validation_out_of_range():
    # FlagGems file semantics: the constraint check is deferred to symbolic
    # tracing, so an out-of-range scalar must NOT raise here (aten would
    # raise RuntimeError "Invalid value range for 50 between [1, 10].").
    dep_token = torch.randn(64, dtype=torch.float32)

    out = _functional_sym_constrain_range(50, 1, 10, dep_token)

    torch.testing.assert_close(out, dep_token, rtol=0, atol=0)


def test_non_contiguous_clone_path():
    torch.manual_seed(0)
    x = torch.randn(8, 4, dtype=torch.float32)
    x_t = x.t()
    assert not x_t.is_contiguous()

    out = _functional_sym_constrain_range(5, 1, 10, x_t)

    assert out is not x_t
    assert out.shape == x_t.shape
    torch.testing.assert_close(out, x_t, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(0,), (3, 0)])
def test_empty_clone_path(shape):
    dep_token = torch.randn(shape, dtype=torch.float32)

    out = _functional_sym_constrain_range(5, 1, 10, dep_token)

    assert out.shape == shape
    assert out.numel() == 0
    assert out.dtype == dep_token.dtype


def test_zero_dim_kernel_path(monkeypatch):
    # 0-d tensors are contiguous with numel 1 -> Triton kernel path.
    mod = _spy_module()

    dep_token = torch.tensor(3.5)

    out = mod._functional_sym_constrain_range(5, 1, 10, dep_token)

    assert out.item() == 3.5


def test_no_tensor_args_returns_first_arg():
    assert _functional_sym_constrain_range(5, 1, 10) == 5
    assert _functional_sym_constrain_range() is None


def test_triton_path_contiguous(monkeypatch):
    # Smoke: the contiguous non-empty path must launch the Triton copy kernel.
    mod = _spy_module()

    launches = []
    real_run = mod._functional_sym_constrain_range_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._functional_sym_constrain_range_kernel, "run", spy_run)
    dep_token = torch.randn(1024, dtype=torch.float32)
    out = mod._functional_sym_constrain_range(5, 1, 10, dep_token)
    assert len(launches) == 1, "contiguous path must go through the Triton kernel"
    assert launches[0] == (1,)
    torch.testing.assert_close(out, dep_token, rtol=0, atol=0)


def test_clone_path_no_kernel(monkeypatch):
    # The non-contiguous and empty fallbacks must NOT launch the kernel.
    mod = _spy_module()

    launches = []
    real_run = mod._functional_sym_constrain_range_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._functional_sym_constrain_range_kernel, "run", spy_run)

    x_t = torch.randn(8, 4).t()
    mod._functional_sym_constrain_range(5, 1, 10, x_t)
    mod._functional_sym_constrain_range(5, 1, 10, torch.randn(0))
    assert launches == [], "clone fallbacks must not launch the Triton kernel"
