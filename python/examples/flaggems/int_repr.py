"""Triton implementation of ``aten::int_repr`` / ``aten::int_repr.out``.

``int_repr`` returns the raw integer codes stored by an affine-quantized
tensor: ``torch.quint8 -> torch.uint8``, ``torch.qint8 -> torch.int8``
and ``torch.qint32 -> torch.int32``. The scale / zero-point metadata
never enters the result, so the operation is a bit-exact typed memory
copy of the quantized storage (rtol = 0, atol = 0).

Layout strategy (mirrors the native operator, which preserves memory
format): the freshly allocated output is given the same strides as the
input alias whenever that alias is storage-dense (contiguous,
``channels_last``, transposed, ``offset_slice``, per-channel), so the
copy degenerates to one linear span. A rank-decomposing kernel handles
genuinely non-dense layouts (zero-stride ``expand``, stepped slices) and
caller-supplied ``out`` buffers with different strides; it pads the
logical shape to rank 5, so higher ranks are rejected.

Port notes (triton-riscv, vs the upstream flag_gems file):
- The integer alias of the quantized storage is taken through
  ``Tensor.set_(untyped_storage(), ...)`` directly. The upstream file
  tries ``tensor.view(dtype)`` first, but on this torch build reading a
  ``quantized.view(integer_dtype)`` tensor crashes natively (probed on
  offset-0 and offset slices alike), while the ``set_`` alias reads the
  exact ``aten::int_repr`` codes.
- The upstream unaligned head/bulk/tail split (16-byte realignment via
  shifted rank-1 aliases) is dropped: unaligned base pointers are legal
  on the CPU backend and the split is a bandwidth optimization only.
- Guards: a non-tensor argument raises TypeError; a non-quantized or
  unsupported-quantized input raises RuntimeError; an ``out`` buffer
  with the wrong dtype raises RuntimeError (aten message style), and a
  wrong-shape buffer is resized, like the native out variant.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

__all__ = ["int_repr"]

# Quantized storage dtype -> plain integer dtype of the result.
_Q_TO_INT = {
    torch.quint8: torch.uint8,
    torch.qint8: torch.int8,
    torch.qint32: torch.int32,
}

# Number of elements each program copies along a dense/linear span.
_BLOCK = 1024
# Longest rank the generic (strided) kernel decomposes in one pass.
_MAX_RANK = 5


@triton.jit
def _int_repr_flat_kernel(in_ptr, out_ptr, numel, BLOCK: tl.constexpr):
    """Linear copy of a dense span: one integer element per lane."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    val = tl.load(in_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, val, mask=mask)


@triton.jit
def _int_repr_strided_kernel(
    in_ptr,
    out_ptr,
    numel,
    n0: tl.constexpr,
    n1: tl.constexpr,
    n2: tl.constexpr,
    n3: tl.constexpr,
    n4: tl.constexpr,
    ia0: tl.constexpr,
    ia1: tl.constexpr,
    ia2: tl.constexpr,
    ia3: tl.constexpr,
    ia4: tl.constexpr,
    ob0: tl.constexpr,
    ob1: tl.constexpr,
    ob2: tl.constexpr,
    ob3: tl.constexpr,
    ob4: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Copy through arbitrary (possibly zero / non-dense) strides.

    The logical shape is right-padded with size-1 / stride-0 slots so a
    single unrolled decomposition covers every rank. Indices decompose
    from the innermost dimension outwards, exactly like a row-major
    multi-index walk. Sizes and strides are ``constexpr`` so the divisors
    fold into multiply-shift sequences.
    """
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel

    rem = offs
    iidx = offs * 0
    oidx = offs * 0

    j = rem % n4
    iidx += j * ia4
    oidx += j * ob4
    rem = rem // n4

    j = rem % n3
    iidx += j * ia3
    oidx += j * ob3
    rem = rem // n3

    j = rem % n2
    iidx += j * ia2
    oidx += j * ob2
    rem = rem // n2

    j = rem % n1
    iidx += j * ia1
    oidx += j * ob1
    rem = rem // n1

    j = rem % n0
    iidx += j * ia0
    oidx += j * ob0

    val = tl.load(in_ptr + iidx, mask=mask)
    tl.store(out_ptr + oidx, val, mask=mask)


def _int_view(tensor, dtype):
    """Zero-copy integer alias of a quantized tensor's storage.

    Aliased through ``set_`` with identical size, stride and storage
    offset: a metadata-only reinterpretation (no device work, no payload
    allocation). See the module docstring for why ``view(dtype)`` is not
    used on this build.
    """
    alias = torch.empty(0, dtype=dtype, device=tensor.device)
    alias.set_(
        tensor.untyped_storage(),
        tensor.storage_offset(),
        tensor.size(),
        tensor.stride(),
    )
    return alias


def _is_storage_dense(numel, shape, stride):
    """True when the strides enumerate one dense span of ``numel`` elements.

    Contiguous layouts short-circuit; otherwise the dimensions are walked
    from the smallest stride upwards and each stride must equal the
    product of the sizes already covered. Zero strides (``expand``) and
    stepped slices fail, so they keep the generic kernel.
    """
    if numel <= 1:
        return True
    if len(shape) <= 1:
        return stride[-1] == 1
    expect = 1
    for n, s in sorted(zip(shape, stride), key=lambda p: p[1]):
        if n <= 1:
            continue
        if s != expect:
            return False
        expect *= n
    return True


def _padded(n, seq, fill):
    return (fill,) * (n - len(seq)) + tuple(int(v) for v in seq)


def int_repr(self, out=None):
    """Return the raw integer codes of the affine-quantized tensor ``self``.

    Implements both ``aten::int_repr(Tensor self) -> Tensor`` and
    ``aten::int_repr.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)``.
    """
    logger.debug("GEMS INT_REPR")
    if not isinstance(self, torch.Tensor):
        raise TypeError(
            "int_repr(): argument 'self' must be a Tensor, not "
            f"{type(self).__name__}"
        )
    if not self.is_quantized:
        raise RuntimeError(
            f"int_repr(): expected a quantized tensor, got dtype {self.dtype}"
        )
    out_dtype = _Q_TO_INT.get(self.dtype)
    if out_dtype is None:
        raise RuntimeError(f"int_repr(): unsupported quantized dtype {self.dtype}")
    if self.ndim > _MAX_RANK:
        raise RuntimeError(
            f"int_repr(): input ranks above {_MAX_RANK} are unsupported"
        )

    shape = self.shape
    numel = self.numel()
    stride = self.stride()
    dense = _is_storage_dense(numel, shape, stride)
    in_view = _int_view(self, out_dtype)

    if out is None:
        if dense:
            # Same strides as the input: element offsets from each
            # data_ptr() are the same linear span, so the copy is flat.
            out = torch.empty_strided(
                shape, stride, dtype=out_dtype, device=self.device
            )
        else:
            out = torch.empty(shape, dtype=out_dtype, device=self.device)
    else:
        if not isinstance(out, torch.Tensor) or out.dtype != out_dtype:
            raise RuntimeError(
                "int_repr(): expected out tensor to have dtype "
                f"{out_dtype}, but got "
                f"{getattr(out, 'dtype', type(out).__name__)} instead"
            )
        if out.shape != shape:
            out.resize_(shape)

    if numel == 0:
        # Never launch a zero grid: one fully masked program is legal and
        # the empty result is already correct.
        _int_repr_flat_kernel[(1,)](in_view, out, 0, BLOCK=_BLOCK)
        return out

    if stride == out.stride():
        _int_repr_flat_kernel[(triton.cdiv(numel, _BLOCK),)](
            in_view, out, numel, BLOCK=_BLOCK
        )
        return out

    sizes = _padded(_MAX_RANK, shape, 1)
    istrides = _padded(_MAX_RANK, stride, 0)
    ostrides = _padded(_MAX_RANK, out.stride(), 0)
    grid = (triton.cdiv(numel, _BLOCK),)
    _int_repr_strided_kernel[grid](
        in_view, out, numel, *sizes, *istrides, *ostrides, BLOCK=_BLOCK
    )
    return out
