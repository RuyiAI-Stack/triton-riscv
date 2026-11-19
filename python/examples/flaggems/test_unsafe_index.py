import sys

import pytest
import torch

from .unsafe_index import unsafe_index

# Semantics (upstream flag_gems/ops/unsafe_index.py + aten, probed on this host):
# unsafe_index(inp, indices) mirrors aten::_unsafe_index(Tensor self,
# Tensor?[] indices) -> Tensor: advanced indexing where the entry indices[i]
# applies to input dim i, a None entry means a full slice, and trailing dims
# may be left unspecified (implicitly None).  On the legal domain every index
# value lies in [-size, 0) U [0, size) per indexed dim; negatives wrap like
# Python indexing (probed: aten._unsafe_index(x, [-1]) == x[-1]).  It then
# equals the safe correspondent inp[tuple(slice(None) if i is None else i)]
# (probed equal on this host, including the split-subspace shape rule
# x3[t1, :, t2] -> (2, 2, 6)).  The "unsafe" contract only removes aten's
# front-end checks; out-of-range indices beyond the wrap window are undefined
# behaviour and are not tested.
# Host-visible contract points pinned below: only int32/int64 index tensors
# are accepted (everything else -> RuntimeError "_unsafe_index found
# unexpected index type <LegacyName>", probed text), multiple tensor indices
# broadcast (IndexError "could not be broadcast"), 0-d tensor indices behave
# like integer indices (the dim is removed from the output, and a 0-d index
# separated from the other tensors by a None dim pushes the broadcast
# subspace to the front of the output), an empty index yields an empty
# output, and the output dtype always equals the input dtype.
# Tolerance: the kernel is a pure data movement (gather through explicit
# strides, no arithmetic), so every non-accumulate comparison is bit-exact
# (rtol=0, atol=0), like the S1-S3 landed tests.

# The op module object, captured at collection time. pytest 9.2 evicts the
# flaggens package from sys.modules after collection, and this environment's
# FileFinder cache for python/examples nondeterministically forgets the
# package (probed: 'flaggens' absent from FileFinder._path_cache while
# present on disk), which breaks re-imports by name at test runtime.
_UNSAFE_INDEX_MOD = sys.modules[__name__.rsplit(".", 1)[0] + ".unsafe_index"]


def _spy_module():
    """Fetch the op module via importlib (S1 rule: package __init__ exports
    shadow submodule attributes). Re-seed sys.modules from the object the
    collection-time import kept alive before the mandated import_module call
    (a direct sys.modules hit needs no parent lookup)."""
    import importlib

    name = "flaggens.unsafe_index"
    if name not in sys.modules:
        sys.modules[name] = _UNSAFE_INDEX_MOD
    return importlib.import_module(name)


def _ref(inp, indices):
    return inp[tuple(slice(None) if i is None else i for i in indices)]


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_unsafe_index_1d(size):
    torch.manual_seed(0)
    x = torch.randn(size, dtype=torch.float32)
    idx = torch.randint(0, size, (max(size // 8, 1),), dtype=torch.int64)

    out = unsafe_index(x, [idx])

    assert out.dtype == torch.float32
    torch.testing.assert_close(out, _ref(x, [idx]), rtol=0, atol=0)


def test_unsafe_index_1d_int32_index():
    torch.manual_seed(0)
    x = torch.randn(512, dtype=torch.float32)
    idx = torch.randint(0, 512, (64,), dtype=torch.int32)

    out = unsafe_index(x, [idx])

    assert out.dtype == torch.float32
    torch.testing.assert_close(out, _ref(x, [idx]), rtol=0, atol=0)


def test_unsafe_index_2d_single_index():
    torch.manual_seed(0)
    x = torch.randn(8, 64, dtype=torch.float32)
    idx = torch.randint(0, 8, (5,), dtype=torch.int64)

    out = unsafe_index(x, [idx])

    assert out.shape == (5, 64)
    torch.testing.assert_close(out, _ref(x, [idx]), rtol=0, atol=0)


def test_unsafe_index_2d_broadcast_indices():
    torch.manual_seed(0)
    x = torch.randn(32, 32, dtype=torch.float32)
    idx0 = torch.randint(0, 32, (8,), dtype=torch.int64)
    idx1 = torch.randint(0, 32, (2, 8), dtype=torch.int64)

    out = unsafe_index(x, [idx0, idx1])

    assert out.shape == (2, 8)
    torch.testing.assert_close(out, _ref(x, [idx0, idx1]), rtol=0, atol=0)


@pytest.mark.parametrize(
    "shape, indices_spec",
    [
        ((16, 16), ("t", None)),  # contiguous block: subspace stays in place
        ((16, 16), (None, "t")),  # advanced block at dim 1
        ((4, 6, 4), ("t22", None, "t22")),  # split: subspace to the front
        ((8, 6, 4), (None, "0d", None)),  # 0-d index on the middle dim
    ],
)
def test_unsafe_index_none_and_zero_dim_patterns(shape, indices_spec):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32)
    pool = {
        "t": torch.randint(0, shape[0], (5,), dtype=torch.int64),
        None: None,
    }
    if len(shape) == 3:
        pool["t22"] = torch.randint(
            0, min(shape[0], shape[2]), (2, 2), dtype=torch.int64
        )
        pool["0d"] = torch.randint(0, shape[1], (), dtype=torch.int64)
    indices = [pool[spec] for spec in indices_spec]

    out = unsafe_index(x, indices)
    ref = _ref(x, indices)

    assert out.shape == ref.shape
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_unsafe_index_zero_dim_split_subspace():
    # A 0-d index separated from the tensor index by a None dim: aten
    # counts 0-d tensors as advanced indices, so the broadcast subspace
    # goes to the front (out (2, 6)). Python sugar x[0d, :, t1]
    # disagrees here (it gives (6, 2); probed on this host), so the oracle
    # for this case is aten._unsafe_index itself, matching upstream.
    torch.manual_seed(0)
    x = torch.randn(4, 6, 3, dtype=torch.float32)
    zero_d = torch.randint(0, 4, (), dtype=torch.int64)
    t1 = torch.randint(0, 3, (2,), dtype=torch.int64)
    indices = [zero_d, None, t1]

    out = unsafe_index(x, indices)
    aten_out = torch.ops.aten._unsafe_index(x, list(indices))

    assert out.shape == torch.Size([2, 6])
    torch.testing.assert_close(out, aten_out, rtol=0, atol=0)


def test_unsafe_index_negative_indices_wrap():
    torch.manual_seed(0)
    x = torch.randn(16, 16, dtype=torch.float32)
    idx = torch.tensor([-1, -16, 5, -11], dtype=torch.int64)

    out = unsafe_index(x, [idx])

    assert out.shape == (4, 16)
    torch.testing.assert_close(out, _ref(x, [idx]), rtol=0, atol=0)


def test_unsafe_index_duplicate_indices():
    torch.manual_seed(0)
    x = torch.randn(64, dtype=torch.float32)
    idx = torch.full((7,), 3, dtype=torch.int64)

    out = unsafe_index(x, [idx])

    assert out.shape == (7,)
    torch.testing.assert_close(out, _ref(x, [idx]), rtol=0, atol=0)


@pytest.mark.parametrize(
    "shape, idx_shape",
    [((512,), (0,)), ((0, 32), (0,))],
)
def test_unsafe_index_empty(shape, idx_shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32)
    idx = torch.empty(idx_shape, dtype=torch.int64)

    out = unsafe_index(x, [idx])

    ref = _ref(x, [idx])
    assert out.shape == ref.shape
    assert out.numel() == 0
    assert out.dtype == torch.float32


def test_unsafe_index_non_contiguous_input():
    torch.manual_seed(0)
    base = torch.randn(64, 64, 64, dtype=torch.float32)
    inp = base.permute(1, 2, 0)
    idx = torch.randint(0, 64, (8,), dtype=torch.int64)
    assert not inp.is_contiguous()

    out = unsafe_index(inp, [idx])

    assert out.shape == (8, 64, 64)
    torch.testing.assert_close(out, _ref(inp, [idx]), rtol=0, atol=0)


def test_unsafe_index_transposed_input():
    torch.manual_seed(0)
    base = torch.randn(32, 16, dtype=torch.float32)
    inp = base.t()
    idx = torch.randint(0, 16, (6,), dtype=torch.int64)
    assert not inp.is_contiguous()

    out = unsafe_index(inp, [idx])

    assert out.shape == (6, 32)
    torch.testing.assert_close(out, _ref(inp, [idx]), rtol=0, atol=0)


def test_unsafe_index_non_contiguous_index_tensor():
    torch.manual_seed(0)
    x = torch.randn(32, 32, dtype=torch.float32)
    idx = torch.arange(0, 16, 2, dtype=torch.int64)[::2]  # strided view
    assert not idx.is_contiguous()

    out = unsafe_index(x, [idx])

    torch.testing.assert_close(out, _ref(x, [idx]), rtol=0, atol=0)


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.int32, torch.int64]
)
def test_unsafe_index_dtype_matrix(dtype):
    torch.manual_seed(0)
    if dtype.is_floating_point:
        x = torch.randn(512, dtype=dtype)
    else:
        x = torch.randint(-1000, 1000, (512,), dtype=dtype)
    idx = torch.randint(0, 512, (64,), dtype=torch.int64)

    out = unsafe_index(x, [idx])

    assert out.dtype == dtype
    torch.testing.assert_close(out, _ref(x, [idx]), rtol=0, atol=0)


def test_unsafe_index_matches_aten_reference():
    # Cross-check the safe-composition reference against aten's own
    # _unsafe_index on this host (contract: equal on the legal domain).
    torch.manual_seed(0)
    x = torch.randn(4, 6, 4, dtype=torch.float32)
    t1 = torch.randint(0, 4, (2, 2), dtype=torch.int64)
    t2 = torch.randint(0, 4, (2, 2), dtype=torch.int64)
    indices = [t1, None, t2]

    aten_out = torch.ops.aten._unsafe_index(x, list(indices))

    torch.testing.assert_close(unsafe_index(x, indices), aten_out, rtol=0, atol=0)
    torch.testing.assert_close(aten_out, _ref(x, indices), rtol=0, atol=0)


@pytest.mark.parametrize(
    "index_dtype",
    [torch.bool, torch.int8, torch.uint8, torch.int16, torch.float32],
)
def test_unsafe_index_error_bad_index_dtype(index_dtype):
    # aten rejects non-int32/int64 index tensors with the legacy TypeMeta
    # name in the message; the port mirrors type and text (probed on aten).
    x = torch.randn(32, 64, dtype=torch.float32)
    idx = torch.randint(0, 32, (8,)).to(index_dtype)

    with pytest.raises(RuntimeError, match="found unexpected index type"):
        unsafe_index(x, [idx])


def test_unsafe_index_error_non_broadcastable_indices():
    x = torch.randn(32, 64, dtype=torch.float32)
    idx1 = torch.randint(0, 32, (4,), dtype=torch.int64)
    idx2 = torch.randint(0, 64, (3,), dtype=torch.int64)

    with pytest.raises(IndexError, match="could not be broadcast"):
        unsafe_index(x, [idx1, idx2])


def test_unsafe_index_error_empty_indices():
    x = torch.randn(32, 64, dtype=torch.float32)

    with pytest.raises(ValueError, match="at least one index must be provided"):
        unsafe_index(x, [])


def test_unsafe_index_error_too_many_indices():
    x = torch.randn(32, 64, dtype=torch.float32)
    indices = [
        torch.randint(0, 32, (8,), dtype=torch.int64),
        torch.randint(0, 64, (8,), dtype=torch.int64),
        torch.randint(0, 32, (8,), dtype=torch.int64),
    ]

    with pytest.raises(IndexError, match="too many indices"):
        unsafe_index(x, indices)


def test_unsafe_index_input_not_modified():
    torch.manual_seed(0)
    x = torch.randn(512, dtype=torch.float32)
    snap = x.clone()
    idx = torch.randint(0, 512, (64,), dtype=torch.int64)

    unsafe_index(x, [idx])

    assert torch.equal(x, snap), "unsafe_index is out-of-place"


def test_unsafe_index_triton_path(monkeypatch):
    # Smoke: the main path must reach the code-generated Triton kernel.
    # unsafe_index is code-generated per (rank, idx_dims, index_rank,
    # bcast_pos) key: the spy asserts the host dispatches through the
    # UnsafeIndexFunction launcher and that a generated kernel module was
    # executed (its wrapper only exists after exec_module ran the
    # @triton.jit decoration). The bit-exact result proves the kernel ran
    # (only the Triton kernel writes `out`).
    mod = _spy_module()

    calls = []
    real_func = mod._unsafe_index_func

    def spy_dispatch(inp, tensor_indices, out, idx_dims, bcast_pos):
        calls.append((idx_dims, bcast_pos))
        return real_func(inp, tensor_indices, out, idx_dims, bcast_pos)

    monkeypatch.setattr(mod, "_unsafe_index_func", spy_dispatch)
    x = torch.randn(64, 64, dtype=torch.float32)
    idx = torch.randint(0, 64, (8,), dtype=torch.int64)
    out = mod.unsafe_index(x, [idx])

    assert len(calls) == 1, "unsafe_index must go through the codegen launcher"
    assert calls[0] == ((0,), 0)
    assert len(real_func.overloads) >= 1, "a generated kernel must exist"
    torch.testing.assert_close(out, _ref(x, [idx]), rtol=0, atol=0)
