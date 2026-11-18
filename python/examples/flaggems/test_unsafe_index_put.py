import sys

import pytest
import torch

from .unsafe_index_put import unsafe_index_put

# Semantics (upstream flag_gems/ops/unsafe_index_put.py + aten, probed on
# this host): unsafe_index_put(inp, indices, values, accumulate=False)
# mirrors torch._unsafe_index_put(Tensor self, Tensor?[] indices, Tensor
# values, bool accumulate=False): out-of-place scatter (the input is cloned,
# the caller's tensor is untouched) writing values at the positions selected
# by the index tensors.  indices[i] applies to dim i, None means a full
# slice, bool/int8 tensors expand to their nonzero coordinates, and the
# tensor indices broadcast against each other.  values broadcasts to the
# gathered shape (broadcast index shape + slice dims).
# Legal domain: every index value in [-size, 0) U [0, size) per indexed dim;
# negatives wrap (probed: torch._unsafe_index_put writes x[-1] to the last
# element -- the port wraps in-kernel to match; the upstream kernel skips
# negatives instead, but never tests them).  With accumulate=False the index
# positions must be unique for a deterministic result (assignment order is
# undefined for repeats, same as aten); with accumulate=True repeats are
# legal and add (atomic), so float sums compare with a small tolerance while
# every other case is a pure data movement compared bit-exact (rtol=0,
# atol=0).  Safe-correspondent reference: ref.clone().index_put_(tuple(...),
# values, accumulate=...) with None -> slice(None) (probed equal on this
# host).

# The op module object, captured at collection time. pytest 9.2 evicts the
# flaggens package from sys.modules after collection, and this environment's
# FileFinder cache for python/examples nondeterministically forgets the
# package (probed: 'flaggens' absent from FileFinder._path_cache while
# present on disk), which breaks re-imports by name at test runtime.
_UNSAFE_INDEX_PUT_MOD = sys.modules[__name__.rsplit(".", 1)[0] + ".unsafe_index_put"]


def _spy_module():
    """Fetch the op module via importlib (S1 rule: package __init__ exports
    shadow submodule attributes). Re-seed sys.modules from the object the
    collection-time import kept alive before the mandated import_module call
    (a direct sys.modules hit needs no parent lookup)."""
    import importlib

    name = "flaggens.unsafe_index_put"
    if name not in sys.modules:
        sys.modules[name] = _UNSAFE_INDEX_PUT_MOD
    return importlib.import_module(name)


def _ref_put(x, indices, values, accumulate=False):
    # setitem handles slice/None/bool keys natively; index_put_ only takes
    # mutually broadcastable tensors in the key (probed), so it is used for
    # the accumulate oracle only (whose tests always pass pure int tensors).
    ref = x.clone()
    key = tuple(slice(None) if i is None else i for i in indices)
    if accumulate:
        ref.index_put_(key, values, accumulate=True)
    else:
        ref[key] = values
    return ref


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_unsafe_index_put_1d(size):
    torch.manual_seed(0)
    x = torch.randn(size, dtype=torch.float32)
    idx = torch.randperm(size)[: max(size // 8, 4)].to(torch.int64)
    vals = torch.randn(idx.numel(), dtype=torch.float32)

    out = unsafe_index_put(x, [idx], vals)

    torch.testing.assert_close(out, _ref_put(x, [idx], vals), rtol=0, atol=0)


def test_unsafe_index_put_1d_int32_index():
    torch.manual_seed(0)
    x = torch.randn(512, dtype=torch.float32)
    idx = torch.randperm(512)[:64].to(torch.int32)
    vals = torch.randn(64, dtype=torch.float32)

    out = unsafe_index_put(x, [idx], vals)

    torch.testing.assert_close(out, _ref_put(x, [idx], vals), rtol=0, atol=0)


def test_unsafe_index_put_2d_rows():
    torch.manual_seed(0)
    x = torch.randn(8, 6, dtype=torch.float32)
    idx = torch.randperm(8)[:5].to(torch.int64)
    vals = torch.randn(5, 6, dtype=torch.float32)

    out = unsafe_index_put(x, [idx], vals)

    torch.testing.assert_close(out, _ref_put(x, [idx], vals), rtol=0, atol=0)


def test_unsafe_index_put_2d_broadcast_indices():
    torch.manual_seed(0)
    x = torch.randn(32, 32, dtype=torch.float32)
    idx0 = torch.randperm(32)[:8].to(torch.int64)
    idx1 = torch.randperm(32)[:8].to(torch.int64).unsqueeze(0)  # (1, 8)
    vals = torch.randn(8, dtype=torch.float32)

    out = unsafe_index_put(x, [idx0, idx1], vals)

    torch.testing.assert_close(
        out, _ref_put(x, [idx0, idx1], vals), rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "shape, indices_spec",
    [
        ((16, 16), ("t", None)),  # contiguous block at dim 0
        ((16, 16), (None, "t")),  # advanced block at dim 1
        ((8, 6, 4), ("t", None, "t")),  # split: transpose path
    ],
)
def test_unsafe_index_put_none_patterns(shape, indices_spec):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32)
    k = 4
    pool = {
        "t": torch.randperm(shape[0])[:k].to(torch.int64),
        None: None,
    }
    if shape[0] != shape[-1]:
        # the split case needs valid values per indexed dim
        pool["t"] = torch.randperm(min(shape[0], shape[-1]))[:k].to(torch.int64)
    indices = [pool[spec] for spec in indices_spec]
    key = tuple(slice(None) if i is None else i for i in indices)
    vals = torch.randn(x[key].shape, dtype=torch.float32)

    out = unsafe_index_put(x, indices, vals)

    torch.testing.assert_close(
        out, _ref_put(x, indices, vals), rtol=0, atol=0
    )


def test_unsafe_index_put_values_broadcast():
    torch.manual_seed(0)
    x = torch.randn(8, 6, dtype=torch.float32)
    idx = torch.randperm(8)[:3].to(torch.int64)
    vals = torch.randn(1, 6, dtype=torch.float32)  # broadcasts over rows

    out = unsafe_index_put(x, [idx], vals)

    torch.testing.assert_close(out, _ref_put(x, [idx], vals), rtol=0, atol=0)


def test_unsafe_index_put_bool_mask_1d():
    torch.manual_seed(0)
    x = torch.randn(512, dtype=torch.float32)
    mask = torch.rand(512) < 0.3
    k = int(mask.sum())
    vals = torch.randn(k, dtype=torch.float32)

    out = unsafe_index_put(x, [mask], vals)

    torch.testing.assert_close(out, _ref_put(x, [mask], vals), rtol=0, atol=0)


def test_unsafe_index_put_bool_mask_rows():
    torch.manual_seed(0)
    x = torch.randn(8, 6, dtype=torch.float32)
    mask = torch.rand(8) < 0.5
    k = int(mask.sum())
    vals = torch.randn(k, 6, dtype=torch.float32)

    out = unsafe_index_put(x, [mask], vals)

    torch.testing.assert_close(out, _ref_put(x, [mask], vals), rtol=0, atol=0)


def test_unsafe_index_put_bool_mask_2d():
    torch.manual_seed(0)
    x = torch.randn(4, 5, dtype=torch.float32)
    mask = torch.rand(4, 5) < 0.5
    k = int(mask.sum())
    vals = torch.randn(k, dtype=torch.float32)

    out = unsafe_index_put(x, [mask], vals)

    torch.testing.assert_close(out, _ref_put(x, [mask], vals), rtol=0, atol=0)


def test_unsafe_index_put_accumulate_duplicates():
    # accumulate=True makes repeated indices legal (atomic adds); the float
    # sum order is unspecified, hence the small tolerance.
    torch.manual_seed(0)
    x = torch.randn(16, dtype=torch.float32)
    idx = torch.tensor([0, 2, 0, 2, 2, 5], dtype=torch.int64)
    vals = torch.randn(6, dtype=torch.float32)

    out = unsafe_index_put(x, [idx], vals, accumulate=True)

    torch.testing.assert_close(
        out, _ref_put(x, [idx], vals, accumulate=True), rtol=1e-5, atol=1e-5
    )


def test_unsafe_index_put_accumulate_int_exact():
    torch.manual_seed(0)
    x = torch.randint(0, 100, (16,), dtype=torch.int64)
    idx = torch.tensor([0, 2, 0, 2, 2, 5], dtype=torch.int64)
    vals = torch.randint(-50, 50, (6,), dtype=torch.int64)

    out = unsafe_index_put(x, [idx], vals, accumulate=True)

    torch.testing.assert_close(
        out, _ref_put(x, [idx], vals, accumulate=True), rtol=0, atol=0
    )


def test_unsafe_index_put_accumulate_2d_rows():
    torch.manual_seed(0)
    x = torch.randn(8, 6, dtype=torch.float32)
    idx = torch.tensor([0, 3, 0, 3, 7], dtype=torch.int64)
    vals = torch.randn(5, 6, dtype=torch.float32)

    out = unsafe_index_put(x, [idx], vals, accumulate=True)

    torch.testing.assert_close(
        out, _ref_put(x, [idx], vals, accumulate=True), rtol=1e-5, atol=1e-5
    )


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.int32, torch.int64]
)
def test_unsafe_index_put_dtype_matrix(dtype):
    torch.manual_seed(0)
    if dtype.is_floating_point:
        x = torch.randn(256, dtype=dtype)
        vals = torch.randn(16, dtype=dtype)
    else:
        x = torch.randint(-500, 500, (256,), dtype=dtype)
        vals = torch.randint(-50, 50, (16,), dtype=dtype)
    idx = torch.randperm(256)[:16].to(torch.int64)

    out = unsafe_index_put(x, [idx], vals)

    assert out.dtype == dtype
    torch.testing.assert_close(out, _ref_put(x, [idx], vals), rtol=0, atol=0)


def test_unsafe_index_put_non_contiguous_input():
    # The host clones the input first (out-of-place contract), which also
    # normalizes strides; the logical result must match the reference.
    torch.manual_seed(0)
    base = torch.randn(6, 8, dtype=torch.float32)
    inp = base.t()
    idx = torch.randperm(8)[:4].to(torch.int64)
    vals = torch.randn(4, 6, dtype=torch.float32)
    assert not inp.is_contiguous()

    out = unsafe_index_put(inp, [idx], vals)

    torch.testing.assert_close(out, _ref_put(inp, [idx], vals), rtol=0, atol=0)


def test_unsafe_index_put_negative_indices_wrap():
    # torch._unsafe_index_put wraps negatives (probed: -1 hits the last
    # element); the port pins the same behaviour.
    torch.manual_seed(0)
    x = torch.randn(8, dtype=torch.float32)
    idx = torch.tensor([-1, -8], dtype=torch.int64)
    vals = torch.tensor([7.0, -3.0])

    out = unsafe_index_put(x, [idx], vals)

    torch.testing.assert_close(out, _ref_put(x, [idx], vals), rtol=0, atol=0)


def test_unsafe_index_put_empty_index_positions():
    torch.manual_seed(0)
    x = torch.randn(16, dtype=torch.float32)
    idx = torch.empty(0, dtype=torch.int64)
    vals = torch.randn(0, dtype=torch.float32)

    out = unsafe_index_put(x, [idx], vals)

    torch.testing.assert_close(out, x, rtol=0, atol=0)


def test_unsafe_index_put_input_not_modified():
    torch.manual_seed(0)
    x = torch.randn(256, dtype=torch.float32)
    snap = x.clone()
    idx = torch.randperm(256)[:16].to(torch.int64)
    vals = torch.randn(16, dtype=torch.float32)

    out = unsafe_index_put(x, [idx], vals)

    assert torch.equal(x, snap), "unsafe_index_put must not touch the input"
    assert out is not x


def test_unsafe_index_put_error_empty_indices():
    x = torch.randn(32, 64, dtype=torch.float32)
    vals = torch.randn(8, dtype=torch.float32)

    with pytest.raises(ValueError, match="At least one index tensor"):
        unsafe_index_put(x, [], vals)


def test_unsafe_index_put_error_all_none():
    x = torch.randn(32, 64, dtype=torch.float32)
    vals = torch.randn(8, dtype=torch.float32)

    with pytest.raises(
        ValueError, match="At least one non-None index tensor is required"
    ):
        unsafe_index_put(x, [None, None], vals)


def test_unsafe_index_put_error_too_many_indices():
    x = torch.randn(32, 64, dtype=torch.float32)
    indices = [
        torch.randint(0, 32, (8,), dtype=torch.int64),
        torch.randint(0, 64, (8,), dtype=torch.int64),
        torch.randint(0, 32, (8,), dtype=torch.int64),
    ]
    vals = torch.randn(8, dtype=torch.float32)

    with pytest.raises(IndexError, match="too many indices"):
        unsafe_index_put(x, indices, vals)


def test_unsafe_index_put_triton_path(monkeypatch):
    # Smoke: the main path must reach the code-generated Triton kernel.
    # unsafe_index_put is code-generated per (inp_rank, indices_len,
    # index_rank) key: the spy asserts the host dispatches through the
    # UnsafeIndexPutFunction launcher and that a generated kernel module was
    # executed. The bit-exact result proves the kernel ran (only the Triton
    # kernel writes the cloned buffer).
    mod = _spy_module()

    calls = []
    real_func = mod._unsafe_index_put_func

    def spy_dispatch(inp, tensor_indices, values, accumulate):
        calls.append(accumulate)
        return real_func(inp, tensor_indices, values, accumulate)

    monkeypatch.setattr(mod, "_unsafe_index_put_func", spy_dispatch)
    x = torch.randn(16, dtype=torch.float32)
    idx = torch.randperm(16)[:4].to(torch.int64)
    vals = torch.randn(4, dtype=torch.float32)
    out = mod.unsafe_index_put(x, [idx], vals)

    assert len(calls) == 1, "unsafe_index_put must go through the codegen launcher"
    assert calls[0] is False
    assert len(real_func.overloads) >= 1, "a generated kernel must exist"
    torch.testing.assert_close(out, _ref_put(x, [idx], vals), rtol=0, atol=0)
