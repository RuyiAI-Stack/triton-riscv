import sys

import pytest
import torch

from ._unsafe_masked_index import _unsafe_masked_index

# Semantics (upstream flag_gems/ops/_unsafe_masked_index.py + aten, probed
# on this host): _unsafe_masked_index(self, mask, indices, fill) implements
# the inductor flat-lane contract of aten::_unsafe_masked_index:
#     out[j] = mask.flat[j] ? self.flat[indices[0].flat[j]] : fill
# with the output shaped like mask and typed like self.  indices must be a
# list holding exactly one used tensor (indices[0]); mask is bool.
# Legal domain (the "unsafe" hazard pinned here): indices[0] must align with
# mask elementwise and every lane's index value must lie in
# [0, self.numel()) -- INCLUDING lanes where mask is False, because the
# upstream kernel dereferences the gathered address at masked-off lanes too
# (the mask only picks the result, it does not guard the load).  Tests
# therefore always generate fully in-range indices, exactly like the
# upstream suite.  The eager aten composite (self._unsafe_index(...).masked_
# fill(~mask, fill)) imposes a different multi-dim contract (indices pair
# with self's dims); on 1-D inputs the two coincide, which the cross-check
# test below pins against torch._unsafe_masked_index directly (probed equal
# on this host).  The general reference is the upstream formula's
# safe-composition torch.where(mask, self.flat[idx].reshape(mask.shape),
# fill).  fill is a Python scalar (a 0-d tensor is also accepted, probed);
# for integer self the fill truncates to the integer nearest zero, like
# aten's masked_fill.  Pure data movement -> bit-exact comparisons
# (rtol=0, atol=0).

# The op module object, captured at collection time. pytest 9.2 evicts the
# flaggens package from sys.modules after collection, and this environment's
# FileFinder cache for python/examples nondeterministically forgets the
# package (probed: 'flaggens' absent from FileFinder._path_cache while
# present on disk), which breaks re-imports by name at test runtime.
_UNSAFE_MASKED_INDEX_MOD = sys.modules[__name__.rsplit(".", 1)[0] + "._unsafe_masked_index"]


def _spy_module():
    """Fetch the op module via importlib (S1 rule: package __init__ exports
    shadow submodule attributes). Re-seed sys.modules from the object the
    collection-time import kept alive before the mandated import_module call
    (a direct sys.modules hit needs no parent lookup)."""
    import importlib

    name = "flaggens._unsafe_masked_index"
    if name not in sys.modules:
        sys.modules[name] = _UNSAFE_MASKED_INDEX_MOD
    return importlib.import_module(name)


def _ref(self_t, mask, indices, fill):
    f = fill.item() if isinstance(fill, torch.Tensor) else fill
    if self_t.dtype.is_floating_point:
        fill_val = f
    else:
        fill_val = int(f)
    flat = self_t.reshape(-1)[indices[0].reshape(-1)].reshape(mask.shape)
    return torch.where(mask, flat, torch.tensor(fill_val, dtype=self_t.dtype))


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_unsafe_masked_index_1d(size):
    torch.manual_seed(0)
    x = torch.randn(size, dtype=torch.float32)
    mask = torch.rand(size) < 0.3
    idx = torch.randint(0, size, (size,))

    out = _unsafe_masked_index(x, mask, [idx], 0.0)

    assert out.dtype == torch.float32
    assert out.shape == mask.shape
    torch.testing.assert_close(out, _ref(x, mask, [idx], 0.0), rtol=0, atol=0)


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.int32, torch.int64]
)
def test_unsafe_masked_index_dtype_matrix(dtype):
    torch.manual_seed(0)
    if dtype.is_floating_point:
        x = torch.randn(512, dtype=dtype)
        fill = 0.0
    else:
        x = torch.randint(-1000, 1000, (512,), dtype=dtype)
        fill = -1
    mask = torch.rand(512) < 0.5
    idx = torch.randint(0, 512, (512,))

    out = _unsafe_masked_index(x, mask, [idx], fill)

    assert out.dtype == dtype
    torch.testing.assert_close(out, _ref(x, mask, [idx], fill), rtol=0, atol=0)


@pytest.mark.parametrize("fill", [0.0, -1.0, 2.5])
def test_unsafe_masked_index_fill_matrix(fill):
    torch.manual_seed(0)
    x = torch.randn(256, dtype=torch.float32)
    mask = torch.rand(256) < 0.5
    idx = torch.randint(0, 256, (256,))

    out = _unsafe_masked_index(x, mask, [idx], fill)

    torch.testing.assert_close(out, _ref(x, mask, [idx], fill), rtol=0, atol=0)


def test_unsafe_masked_index_int_fill_on_float_data():
    torch.manual_seed(0)
    x = torch.randn(128, dtype=torch.float32)
    mask = torch.rand(128) < 0.5
    idx = torch.randint(0, 128, (128,))

    out = _unsafe_masked_index(x, mask, [idx], 7)

    torch.testing.assert_close(out, _ref(x, mask, [idx], 7), rtol=0, atol=0)


def test_unsafe_masked_index_tensor_fill():
    torch.manual_seed(0)
    x = torch.randn(128, dtype=torch.float32)
    mask = torch.rand(128) < 0.5
    idx = torch.randint(0, 128, (128,))
    fill = torch.tensor(-3.0)

    out = _unsafe_masked_index(x, mask, [idx], fill)

    torch.testing.assert_close(out, _ref(x, mask, [idx], fill), rtol=0, atol=0)


def test_unsafe_masked_index_2d():
    torch.manual_seed(0)
    x = torch.randn(8, 64, dtype=torch.float32)
    mask = torch.rand(8, 64) < 0.5
    idx = torch.randint(0, 512, (8, 64))

    out = _unsafe_masked_index(x, mask, [idx], 0.0)

    assert out.shape == (8, 64)
    torch.testing.assert_close(out, _ref(x, mask, [idx], 0.0), rtol=0, atol=0)


def test_unsafe_masked_index_flat_self_2d_mask():
    # self is addressed flat: the index values range over self.numel(),
    # regardless of self's own shape.
    torch.manual_seed(0)
    x = torch.randn(512, dtype=torch.float32)
    mask = torch.rand(8, 64) < 0.5
    idx = torch.randint(0, 512, (8, 64))

    out = _unsafe_masked_index(x, mask, [idx], 1.5)

    assert out.shape == (8, 64)
    torch.testing.assert_close(out, _ref(x, mask, [idx], 1.5), rtol=0, atol=0)


def test_unsafe_masked_index_all_false_mask():
    # masked-off lanes still dereference their index (the unsafe hazard), so
    # the indices must be in range even where the mask is False; the output
    # collapses to fill everywhere.
    torch.manual_seed(0)
    x = torch.randn(256, dtype=torch.float32)
    mask = torch.zeros(256, dtype=torch.bool)
    idx = torch.randint(0, 256, (256,))

    out = _unsafe_masked_index(x, mask, [idx], -2.5)

    assert torch.equal(out, torch.full((256,), -2.5))
    torch.testing.assert_close(out, _ref(x, mask, [idx], -2.5), rtol=0, atol=0)


def test_unsafe_masked_index_all_true_mask():
    torch.manual_seed(0)
    x = torch.randn(256, dtype=torch.float32)
    mask = torch.ones(256, dtype=torch.bool)
    idx = torch.randint(0, 256, (256,))

    out = _unsafe_masked_index(x, mask, [idx], 0.0)

    torch.testing.assert_close(out, x[idx], rtol=0, atol=0)


def test_unsafe_masked_index_empty():
    x = torch.randn(8, dtype=torch.float32)
    mask = torch.zeros(0, dtype=torch.bool)
    idx = torch.zeros(0, dtype=torch.int64)

    out = _unsafe_masked_index(x, mask, [idx], 0.0)

    assert out.shape == (0,)
    assert out.dtype == torch.float32


def test_unsafe_masked_index_zero_dim():
    torch.manual_seed(0)
    x = torch.randn(6, dtype=torch.float32)
    mask = torch.tensor(True)
    idx = torch.tensor(3)

    out = _unsafe_masked_index(x, mask, [idx], 0.0)

    assert out.shape == ()
    torch.testing.assert_close(out, _ref(x, mask, [idx], 0.0), rtol=0, atol=0)


def test_unsafe_masked_index_non_contiguous():
    # The port flattens reads via reshape (copies for non-contiguous views)
    # and allocates a contiguous output, so transposed mask/indices/self all
    # work; upstream's empty_like(mask) allocation would leave a transposed
    # output unwritten (latent bug never exercised upstream).
    torch.manual_seed(0)
    base = torch.randn(8, 64, dtype=torch.float32)
    self_t = base.t()  # (64, 8) view
    mask = torch.rand(64, 8) < 0.5
    idx = torch.randint(0, 512, (64, 8))
    mask_t = mask.t()  # (8, 64) non-contiguous view
    idx_t = idx.t()
    assert not self_t.is_contiguous() and not mask_t.is_contiguous()

    out = _unsafe_masked_index(self_t, mask_t, [idx_t], 0.5)

    assert out.shape == mask_t.shape
    torch.testing.assert_close(
        out, _ref(self_t, mask_t, [idx_t], 0.5), rtol=0, atol=0
    )


def test_unsafe_masked_index_matches_aten_reference():
    # On 1-D inputs the flat-lane contract coincides with the eager aten
    # composite (probed); pin the equivalence against the aten oracle.
    torch.manual_seed(0)
    x = torch.randn(512, dtype=torch.float32)
    mask = torch.rand(512) < 0.3
    idx = torch.randint(0, 512, (512,))

    aten_out = torch._unsafe_masked_index(x, mask, [idx], -1.5)

    torch.testing.assert_close(
        _unsafe_masked_index(x, mask, [idx], -1.5), aten_out, rtol=0, atol=0
    )


def test_unsafe_masked_index_error_empty_indices():
    x = torch.randn(8, dtype=torch.float32)
    mask = torch.rand(8) < 0.5

    with pytest.raises(ValueError, match="indices cannot be empty or None"):
        _unsafe_masked_index(x, mask, [], 0.0)


def test_unsafe_masked_index_error_none_indices():
    x = torch.randn(8, dtype=torch.float32)
    mask = torch.rand(8) < 0.5

    with pytest.raises(ValueError, match="indices cannot be empty or None"):
        _unsafe_masked_index(x, mask, [None], 0.0)


def test_unsafe_masked_index_triton_path(monkeypatch):
    # Smoke: the main path must launch the Triton kernel.
    mod = _spy_module()

    launches = []
    real_run = mod._unsafe_masked_index_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._unsafe_masked_index_kernel, "run", spy_run)
    x = torch.randn(1024, dtype=torch.float32)
    mask = torch.rand(1024) < 0.5
    idx = torch.randint(0, 1024, (1024,))
    out = mod._unsafe_masked_index(x, mask, [idx], 0.0)

    assert len(launches) == 1, "_unsafe_masked_index must use the Triton kernel"
    assert launches[0] == (2,)  # cdiv(1024, BLOCK_SIZE=512)
    torch.testing.assert_close(out, _ref(x, mask, [idx], 0.0), rtol=0, atol=0)
