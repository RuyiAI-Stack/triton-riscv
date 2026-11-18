import sys

import pytest
import torch
import torch.nn.functional as F

from ._masked_softmax import _masked_softmax

# Semantics (upstream flag_gems/ops/_masked_softmax.py + aten, probed on this
# host with torch 2.15.0a0+cpu): _masked_softmax(input, mask, dim=None,
# mask_type=None) is softmax over `dim` with bool-masked positions removed.
# mask is a bool tensor where True marks a MASKED-OUT position (probed:
# aten == F.softmax(x.masked_fill(mask, -inf), dim)); masked positions come
# out exactly 0.0.  mask_type only selects how the mask broadcasts to the
# input shape:
#   0 -- source mask: right-aligned, [1]*(ndim-mask.ndim)+mask.shape
#        broadcast over the leading dims (attention (B,H,L,L) scores with an
#        (L,L) mask; aten additionally rejects non-(L,L) shapes, the upstream
#        right-align generalization is a superset, probed on the overlap);
#   1 -- padding mask: 2-D [B,K] reshaped to
#        [B]+[1]*(dim-1)+[K]+[1]*(ndim-dim-1), broadcast across the middle
#        dims ((B,H,L,L) with a (B,L) mask); dim must be >= 1 for the
#        reshape to stay inside the input rank;
#   2 -- elementwise mask with the same shape as the input.
# dim defaults to the last dim; negative dims are normalized.  Fully-masked
# reduction rows are NOT special-cased: the kernel's masked lanes contribute
# 0 to the softmax sum, so the row divides 0/0 and yields NaN -- matching
# aten (probed) and F.softmax(masked_fill(...)) (probed: NaN pattern equal).
# Overflow: exp runs on (x - row_max) with the row max taken over the masked
# row (upstream formulation), so 1e30-scale fp32 inputs stay finite.  The
# wrapper asserts mask.dtype == bool ("Mask should be a boolean tensor",
# same text as aten) and mask_type in (0,1,2) (aten silently accepts other
# mask_type values; the port follows upstream and rejects them).
#
# Tolerances: the kernel accumulates in fp32 like torch's CPU softmax, but
# in a different summation order -> a few-ulp drift (probed: fp64 aten vs
# formula within 1e-12, fp32 within 8e-9).  fp32 uses rtol=atol=1e-4 (the
# repo softmax-family convention); fp16 stores round once from the fp32
# tile, torch fp16 softmax also accumulates fp32 -> rtol=atol=1e-3.

# The op module object, captured at collection time. pytest 9.2 evicts the
# flaggems package from sys.modules after collection, and this environment's
# FileFinder cache for python/examples nondeterministically forgets the
# package (probed: package absent from FileFinder._path_cache while present
# on disk), which breaks re-imports by name at test runtime.
_MASKED_SOFTMAX_MOD = sys.modules[__name__.rsplit(".", 1)[0] + "._masked_softmax"]


def _spy_module():
    """Fetch the op module via importlib (S1 rule: package __init__ exports
    shadow submodule attributes). Re-seed sys.modules from the object the
    collection-time import kept alive before the mandated import_module call
    (a direct sys.modules hit needs no parent lookup)."""
    import importlib

    name = "flaggems._masked_softmax"
    if name not in sys.modules:
        sys.modules[name] = _MASKED_SOFTMAX_MOD
    return importlib.import_module(name)


def _broadcast_mask(x, mask, dim, mask_type):
    # Materialize the mask to x's shape exactly like the wrapper does, for
    # the masked_fill-based reference.
    ndim = x.ndim
    if mask_type == 2:
        return mask
    if mask_type == 0:
        new_shape = [1] * (ndim - mask.ndim) + list(mask.shape)
    else:  # mask_type == 1
        new_shape = (
            [mask.shape[0]]
            + [1] * (dim - 1)
            + [mask.shape[1]]
            + [1] * (ndim - dim - 1)
        )
    return mask.reshape(new_shape)


def _ref(x, mask, dim, mask_type):
    return F.softmax(x.masked_fill(_broadcast_mask(x, mask, dim, mask_type), float("-inf")), dim)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_masked_softmax_1d(size):
    torch.manual_seed(0)
    x = torch.randn(size, dtype=torch.float32)
    mask = torch.rand(size) < 0.3

    out = _masked_softmax(x, mask, -1, 2)

    assert out.dtype == torch.float32
    torch.testing.assert_close(out, _ref(x, mask, -1, 2), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_masked_softmax_2d(size):
    torch.manual_seed(0)
    x = torch.randn(size, size, dtype=torch.float32)
    mask = torch.rand(size, size) < 0.5

    out = _masked_softmax(x, mask, -1, 2)

    assert out.dtype == torch.float32
    torch.testing.assert_close(out, _ref(x, mask, -1, 2), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("dim", [-1, 0, 1, 2, -2])
def test_masked_softmax_dim(dim):
    # Elementwise mask along every reduction axis (including negatives);
    # non-last dims exercise the strided (inner > 1) kernel path.
    torch.manual_seed(0)
    x = torch.randn(8, 16, 32, dtype=torch.float32)
    mask = torch.rand(8, 16, 32) < 0.4

    out = _masked_softmax(x, mask, dim, 2)

    torch.testing.assert_close(out, _ref(x, mask, dim, 2), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape,mask_shape", [((4, 8, 32, 32), (32, 32)), ((2, 3, 16, 16), (16, 16))])
def test_masked_softmax_type0_source_mask(shape, mask_shape):
    # mask_type 0: (B,H,L,L) attention scores with an (L,L) source mask.
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32)
    mask = torch.rand(mask_shape) < 0.5

    out = _masked_softmax(x, mask, 3, 0)

    torch.testing.assert_close(out, _ref(x, mask, 3, 0), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape,mask_shape", [((4, 8, 32, 32), (4, 32)), ((2, 3, 16, 16), (2, 16))])
def test_masked_softmax_type1_padding_mask(shape, mask_shape):
    # mask_type 1: (B,H,L,L) attention scores with a (B,L) padding mask.
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32)
    mask = torch.rand(mask_shape) < 0.5

    out = _masked_softmax(x, mask, 3, 1)

    torch.testing.assert_close(out, _ref(x, mask, 3, 1), rtol=1e-4, atol=1e-4)


def test_masked_softmax_type0_right_align_general():
    # The upstream right-align broadcast is more general than aten's strict
    # (L,L) check (aten rejects a 1-D mask with mask_type 0, probed);
    # pin the generalization against the composed reference.
    torch.manual_seed(0)
    x = torch.randn(2, 3, 6, 4, dtype=torch.float32)
    mask = torch.rand(4) < 0.5  # right-aligns to (1, 1, 1, 4)

    out = _masked_softmax(x, mask, 3, 0)

    torch.testing.assert_close(out, _ref(x, mask, 3, 0), rtol=1e-4, atol=1e-4)


def test_masked_softmax_type1_non_last_dim():
    torch.manual_seed(0)
    x = torch.randn(8, 16, 32, dtype=torch.float32)
    mask = torch.rand(8, 16) < 0.5  # (outer, K) for dim=1

    out = _masked_softmax(x, mask, 1, 1)

    torch.testing.assert_close(out, _ref(x, mask, 1, 1), rtol=1e-4, atol=1e-4)


def test_masked_softmax_dim_default_is_last():
    torch.manual_seed(0)
    x = torch.randn(16, 64, dtype=torch.float32)
    mask = torch.rand(16, 64) < 0.5

    out = _masked_softmax(x, mask, None, 2)

    torch.testing.assert_close(out, _ref(x, mask, -1, 2), rtol=1e-4, atol=1e-4)


def test_masked_softmax_fully_masked_row_is_nan():
    # Contract (probed on aten and the masked_fill reference): a fully
    # masked reduction row is 0/0 = NaN, not special-cased to zeros.
    torch.manual_seed(0)
    x = torch.randn(8, 64, dtype=torch.float32)
    mask = torch.rand(8, 64) < 0.5
    mask[3] = True

    out = _masked_softmax(x, mask, -1, 2)

    assert torch.isnan(out[3]).all(), "fully-masked row must be NaN"
    ref = _ref(x, mask, -1, 2)
    torch.testing.assert_close(out[:3], ref[:3], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(out[4:], ref[4:], rtol=1e-4, atol=1e-4)


def test_masked_softmax_all_masked_is_nan():
    torch.manual_seed(0)
    x = torch.randn(8, 64, dtype=torch.float32)
    mask = torch.ones(8, 64, dtype=torch.bool)

    out = _masked_softmax(x, mask, -1, 2)

    assert torch.isnan(out).all()


def test_masked_softmax_no_mask_equals_softmax():
    # An all-False mask is the identity: plain softmax.
    torch.manual_seed(0)
    x = torch.randn(8, 64, dtype=torch.float32)
    mask = torch.zeros(8, 64, dtype=torch.bool)

    out = _masked_softmax(x, mask, -1, 2)

    torch.testing.assert_close(out, F.softmax(x, -1), rtol=1e-4, atol=1e-4)


def test_masked_softmax_masked_positions_exactly_zero():
    torch.manual_seed(0)
    x = torch.randn(8, 64, dtype=torch.float32)
    mask = torch.rand(8, 64) < 0.5

    out = _masked_softmax(x, mask, -1, 2)

    assert (out[mask] == 0).all(), "masked positions must be exactly 0.0"


def test_masked_softmax_large_values_no_overflow():
    # Overflow guard: exp runs on x - row_max, so 1e30-scale inputs stay
    # finite (without the shift, exp(1e30) overflows to inf and the row
    # collapses to NaN).
    torch.manual_seed(0)
    x = torch.randn(8, 64, dtype=torch.float32) * 1e30
    mask = torch.rand(8, 64) < 0.5

    out = _masked_softmax(x, mask, -1, 2)

    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, _ref(x, mask, -1, 2), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_masked_softmax_dtype_matrix(dtype):
    torch.manual_seed(0)
    x = torch.randn(8, 512, dtype=torch.float32).to(dtype)
    mask = torch.rand(8, 512) < 0.5

    out = _masked_softmax(x, mask, -1, 2)

    assert out.dtype == dtype
    tol = 1e-3 if dtype == torch.float16 else 1e-4
    torch.testing.assert_close(
        out, _ref(x, mask, -1, 2), equal_nan=True, rtol=tol, atol=tol
    )


def test_masked_softmax_fp16_large_values():
    # fp16 tops out at ~65504; the max-shift keeps exp() in range.
    torch.manual_seed(0)
    # stay inside fp16's finite range (max ~65504) so only the softmax
    # itself, not the input cast, can overflow
    x = (torch.randn(4, 128) * 2e4).clamp(-6e4, 6e4).to(torch.float16)
    mask = torch.rand(4, 128) < 0.5

    out = _masked_softmax(x, mask, -1, 2)

    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, _ref(x, mask, -1, 2), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("shape", [(0, 512), (512, 0), (0,)])
def test_masked_softmax_empty(shape):
    x = torch.randn(shape, dtype=torch.float32)
    mask = torch.rand(shape) < 0.5

    out = _masked_softmax(x, mask, -1 if len(shape) > 1 else 0, 2)

    assert out.shape == x.shape
    assert out.dtype == x.dtype
    assert out.numel() == 0


def test_masked_softmax_non_contiguous():
    torch.manual_seed(0)
    x_base = torch.randn(64, 32, dtype=torch.float32)
    m_base = torch.rand(64, 32) < 0.5
    x = x_base.t()  # (32, 64) view
    mask = m_base.t()
    assert not x.is_contiguous() and not mask.is_contiguous()

    out = _masked_softmax(x, mask, -1, 2)

    assert out.shape == x.shape
    torch.testing.assert_close(out, _ref(x.contiguous(), mask.contiguous(), -1, 2), rtol=1e-4, atol=1e-4)


def test_masked_softmax_non_contiguous_mask_only():
    torch.manual_seed(0)
    x = torch.randn(32, 64, dtype=torch.float32)
    mask = (torch.rand(64, 32) < 0.5).t()  # (32, 64) view
    assert not mask.is_contiguous()

    out = _masked_softmax(x, mask, -1, 2)

    torch.testing.assert_close(out, _ref(x, mask.contiguous(), -1, 2), rtol=1e-4, atol=1e-4)


def test_masked_softmax_matches_aten():
    # Oracle cross-check on the probed-legal aten domain: elementwise mask,
    # any dim, plus the two attention broadcast patterns on the last dim.
    torch.manual_seed(0)
    # small-K reductions randomly hit fully-masked rows, which are NaN on
    # both sides by contract -> equal_nan
    x = torch.randn(4, 8, 16, dtype=torch.float32)
    mask = torch.rand(4, 8, 16) < 0.5
    for dim in (0, 1, 2):
        aten_out = torch.ops.aten._masked_softmax(x, mask, dim, 2)
        torch.testing.assert_close(
            _masked_softmax(x, mask, dim, 2), aten_out, equal_nan=True, rtol=1e-4, atol=1e-4
        )

    x4 = torch.randn(2, 3, 16, 16, dtype=torch.float32)
    m_src = torch.rand(16, 16) < 0.5
    aten_out = torch.ops.aten._masked_softmax(x4, m_src, 3, 0)
    torch.testing.assert_close(
        _masked_softmax(x4, m_src, 3, 0), aten_out, equal_nan=True, rtol=1e-4, atol=1e-4
    )

    m_pad = torch.rand(2, 16) < 0.5
    aten_out = torch.ops.aten._masked_softmax(x4, m_pad, 3, 1)
    torch.testing.assert_close(
        _masked_softmax(x4, m_pad, 3, 1), aten_out, equal_nan=True, rtol=1e-4, atol=1e-4
    )


def test_masked_softmax_input_not_modified():
    torch.manual_seed(0)
    x = torch.randn(16, 64, dtype=torch.float32)
    mask = torch.rand(16, 64) < 0.5
    x_snap, m_snap = x.clone(), mask.clone()

    _masked_softmax(x, mask, -1, 2)

    assert torch.equal(x, x_snap), "out-of-place op must not touch the input"
    assert torch.equal(mask, m_snap)


def test_masked_softmax_error_non_bool_mask():
    x = torch.randn(8, dtype=torch.float32)
    mask = torch.zeros(8)
    with pytest.raises(AssertionError, match="Mask should be a boolean tensor"):
        _masked_softmax(x, mask, -1, 2)


@pytest.mark.parametrize("mask_type", [None, 3, -1])
def test_masked_softmax_error_bad_mask_type(mask_type):
    x = torch.randn(8, dtype=torch.float32)
    mask = torch.rand(8) < 0.5
    with pytest.raises(AssertionError, match="mask_type should be 0, 1 or 2"):
        _masked_softmax(x, mask, -1, mask_type)


def test_masked_softmax_error_zero_dim():
    x = torch.tensor(3.5)
    mask = torch.tensor(True)
    with pytest.raises(AssertionError, match="at least 1 dimension"):
        _masked_softmax(x, mask, -1, 2)


def test_masked_softmax_triton_path(monkeypatch):
    # Smoke: the main path must launch the Triton kernel.
    mod = _spy_module()

    launches = []
    real_run = mod._masked_softmax_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._masked_softmax_kernel, "run", spy_run)
    x = torch.randn(8, 64, dtype=torch.float32)
    mask = torch.rand(8, 64) < 0.5
    out = mod._masked_softmax(x, mask, 1, 2)

    assert len(launches) == 1, "_masked_softmax must use the Triton kernel"
    assert launches[0] == (8,)  # one program per reduction row (outer*inner)
    torch.testing.assert_close(out, _ref(x, mask, 1, 2), rtol=1e-4, atol=1e-4)
