"""quantize_per_tensor - per-tensor affine quantization.

Port notes (triton-riscv, vs the upstream flag_gems file):
- flag_gems.utils (tl_extra_shim / shape_utils / ops.copy) are inlined: the
  backend lowers no libdevice externs, so ``rint`` is a floor-based
  round-half-to-even helper and the overlap check uses
  ``torch._debug_has_internal_overlap`` (0/1/2 = No/Yes/TooHard).
- The kernel arithmetic follows the CPU aten oracle, which was probed to
  compute ``rint(x * (1.f/scale)) + zero_point`` in fp32 (the fp32
  reciprocal-multiply): a true fp32 division flips crafted tie values, and the
  upstream fp64 division matches CUDA aten but NOT the CPU oracle the tests
  pin.  The reciprocal is computed on the host with an exact fp32 division.
- The integer copy on the input-aliasing ``out=`` path uses aten ``copy_``
  over the integer views instead of the flag_gems Triton ``copy_`` op.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# Mapping from a quantized dtype to the underlying integer storage dtype and
# the representable integer range (qmin, qmax) used for clamping.
_QUANT_INFO = {
    torch.quint8: (torch.uint8, 0, 255),
    torch.qint8: (torch.int8, -128, 127),
    torch.qint32: (torch.int32, -2147483648, 2147483647),
}


@triton.jit
def _round_half_to_even(x):
    floor_x = tl.floor(x)
    fraction = x - floor_x
    floor_is_even = (floor_x % 2.0) == 0.0
    rounded = tl.where(
        fraction > 0.5,
        floor_x + 1.0,
        tl.where((fraction < 0.5) | floor_is_even, floor_x, floor_x + 1.0),
    )
    # floor() saturates non-finite inputs to huge finite values on some
    # backends (riscv64 CPU), which would round NaN/inf to a clamp bound.
    # Detect them from the fp32 exponent bits rather than by float comparison
    # so the passthrough does not depend on comparison semantics.
    abs_bits = x.to(tl.float32).to(tl.int32, bitcast=True) & 0x7FFFFFFF
    return tl.where(abs_bits >= 0x7F800000, x, rounded)


@triton.jit
def quantize_per_tensor_kernel(
    x_ptr,
    out_ptr,
    inv_scale,
    zero_point,
    qmin: tl.constexpr,
    qmax: tl.constexpr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # CPU aten: int = clamp(rint(x * (1.f/scale)) + zero_point, qmin, qmax);
    # the rounding happens before the zero_point add.
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    q = _round_half_to_even(x * inv_scale) + zero_point
    q = tl.minimum(tl.maximum(q, qmin), qmax)
    tl.store(out_ptr + offsets, q.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def quantize_per_tensor_strided_kernel(
    x_ptr,
    out_ptr,
    inv_scale,
    zero_point,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    x0,
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    x7,
    o0,
    o1,
    o2,
    o3,
    o4,
    o5,
    o6,
    o7,
    qmin: tl.constexpr,
    qmax: tl.constexpr,
    n_elements,
    ndim: tl.constexpr,
    IDX32: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Same arithmetic as ``quantize_per_tensor_kernel``, but strided on both sides.

    Used whenever the input and the destination do not share a linear memory
    order -- an ``out=`` in a non-contiguous layout (a slice, a transpose, a
    channels-last buffer) or a non-contiguous input. The flat kernel walks
    memory as a linear array, which only addresses the right elements when the
    logical order is the physical order; here the linear task index is instead
    decomposed into a multi-index against the *logical* shape and each side is
    re-flattened with its own strides.

    The sizes and strides arrive as scalar arguments (``s0..s7``, ``x0..x7``,
    ``o0..o7``, padded with 1 up to ``MAX_NDIM`` = 8) rather than through
    pointer tensors: the wrapper then needs no per-call metadata allocations.

    ``IDX32`` selects 32-bit index arithmetic; the launcher only sets it when
    the *whole span* of both sides fits int32, so the multiplication cannot
    overflow (see ``_int32_safe``).

    Decomposition runs from the last dimension inward, the order a C-contiguous
    linear index is built in. Any order would be a valid bijection as long as
    both sides use the same one; this one is chosen so that for a contiguous
    destination the traversal matches the flat kernel's element for element.
    The shape is shared because both sides always have the input's shape (the
    ``out=`` path rejects a mismatch).
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    if IDX32:
        src = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
        dst = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
        rem = offsets.to(tl.int32)
    else:
        src = tl.zeros([BLOCK_SIZE], dtype=tl.int64)
        dst = tl.zeros([BLOCK_SIZE], dtype=tl.int64)
        rem = offsets.to(tl.int64)
    for j in tl.static_range(ndim):
        k = ndim - 1 - j
        if k == 0:
            size, xst, ost = s0, x0, o0
        elif k == 1:
            size, xst, ost = s1, x1, o1
        elif k == 2:
            size, xst, ost = s2, x2, o2
        elif k == 3:
            size, xst, ost = s3, x3, o3
        elif k == 4:
            size, xst, ost = s4, x4, o4
        elif k == 5:
            size, xst, ost = s5, x5, o5
        elif k == 6:
            size, xst, ost = s6, x6, o6
        else:
            size, xst, ost = s7, x7, o7
        rem_col = rem % size
        src += rem_col * xst
        dst += rem_col * ost
        rem = rem // size
    x = tl.load(x_ptr + src, mask=mask, other=0.0).to(tl.float32)
    q = _round_half_to_even(x * inv_scale) + zero_point
    q = tl.minimum(tl.maximum(q, qmin), qmax)
    tl.store(out_ptr + dst, q.to(out_ptr.dtype.element_ty), mask=mask)


# The stride arguments are padded to this many dimensions; a quantized
# tensor beyond 8 dimensions has no linear path anyway (``is_nonoverlapping_
# and_dense`` rejects it), so the strided kernel is the last resort and its
# argument list is bounded here.
MAX_NDIM = 8

# 2**31 - 1, the largest value an int32 index can hold before the address
# arithmetic overflows.
_INT32_MAX = 2**31 - 1


def _int32_safe(a, out_int):
    """True when every address either side touches fits in an int32 index.

    The largest element offset a side can produce is the span of its view
    (``(size - 1) * stride`` summed over the dimensions); both spans plus the
    task count must stay under 2**31, otherwise the strided kernel runs in
    int64 and pays the wider arithmetic.
    """
    if a.numel() > _INT32_MAX or out_int.numel() > _INT32_MAX:
        return False

    def span(shape, strides):
        return sum((size - 1) * stride for size, stride in zip(shape, strides))

    return (
        span(a.shape, a.stride()) <= _INT32_MAX
        and span(out_int.shape, out_int.stride()) <= _INT32_MAX
    )


def _padded(values, ndim):
    """Pad a shape/stride tuple with 1s up to ``MAX_NDIM`` dimensions."""
    return tuple(values) + (1,) * (MAX_NDIM - ndim)


# Mapping from a quantized dtype to its integer storage dtype, used to alias a
# quantized tensor's storage as an integer tensor a Triton kernel can write.
_QINT_DTYPES = {
    torch.quint8: torch.uint8,
    torch.qint8: torch.int8,
    torch.qint32: torch.int32,
}

# Inverse of ``_QINT_DTYPES``: the quantized dtype owning an integer storage
# dtype. Needed because ``_empty_affine_quantized`` requires a quantized dtype
# (it is registered only for the quantized backends).
_QDTYPE_FROM_INT = {v: k for k, v in _QINT_DTYPES.items()}


class MemOverlap:
    No = 0
    TooHard = 1
    Yes = 2


def has_internal_overlapping(x):
    """Inlined flag_gems shape_utils helper (torch._debug_has_internal_overlap
    reports 0 = No, 1 = Yes, 2 = TooHard)."""
    if x.is_contiguous():
        return MemOverlap.No
    if torch._debug_has_internal_overlap(x) == 0:
        return MemOverlap.No
    if torch._debug_has_internal_overlap(x) == 1:
        return MemOverlap.Yes
    return MemOverlap.TooHard


def _validate_quant_args(a, dtype):
    """ATen's input validation for the quantized dtype and the float input.

    Shared by the functional and the ``out=`` entry point so both reject the
    same inputs with the same messages.
    """
    if dtype not in _QUANT_INFO:
        raise RuntimeError(
            f"quantize_per_tensor: unsupported quantized dtype {dtype}, expected "
            "one of torch.quint8, torch.qint8, torch.qint32"
        )
    # ATen only accepts float32 input and raises for anything else, rather than
    # silently upcasting/downcasting.
    if a.dtype != torch.float32:
        raise RuntimeError(
            f"quantize_per_tensor: expected input of dtype torch.float32, got {a.dtype}"
        )
    return _QUANT_INFO[dtype]


def _check_zero_point_range(zero_point, dtype):
    """ATen's zero-point range check (``quantize_tensor_per_tensor_affine``).

    Native rejects an out-of-range zero-point with a synchronous RuntimeError
    naming the value instead of silently clamping it into a valid bin.
    """
    _, qmin, qmax = _QUANT_INFO[dtype]
    zero_point = int(zero_point)
    if zero_point < qmin:
        raise RuntimeError(
            f"quantize_tensor_per_tensor_affine zero_point {zero_point} is below "
            "lower bound."
        )
    if zero_point > qmax:
        raise RuntimeError(
            f"quantize_tensor_per_tensor_affine zero_point {zero_point} is above "
            "upper bound."
        )


def _int_view(qtensor):
    """Alias a quantized tensor's storage as a writable integer tensor.

    ``int_repr()`` returns a *copy*, so it cannot be used as a kernel
    destination, and for a strided tensor it would materialise a compacted
    copy. Aliasing the storage with the tensor's own sizes and strides gives a
    view the kernel can store into, whatever layout it has; the storage offset
    is part of ``data_ptr()``, so an offset view writes into its own slice.
    """
    return torch.empty(
        0, dtype=_QINT_DTYPES[qtensor.dtype], device=qtensor.device
    ).set_(
        qtensor.untyped_storage(),
        qtensor.storage_offset(),
        tuple(qtensor.shape),
        qtensor.stride(),
    )


def _aliases(a, out):
    """True when ``out`` shares storage with the float input ``a``."""
    return out.untyped_storage().data_ptr() == a.untyped_storage().data_ptr()


def _same_quantizer(out, scale, zero_point):
    """True when ``out`` already carries this per-tensor quantizer."""
    return (
        out.qscheme() == torch.per_tensor_affine
        and out.q_scale() == float(scale)
        and out.q_zero_point() == int(zero_point)
    )


def _adopt_quantizer(out, scale, zero_point):
    """Re-point ``out`` at ``scale`` / ``zero_point``, in place, without copying.

    ``out.copy_(result)`` is the only ATen operation that moves a quantizer
    between quantized tensors, but it moves the values too -- and on a
    quantized tensor it redispatches to ATen, adding a native O(n) pass after
    the Triton kernel has already written the integers. Since the values are
    written by the kernel, only the quantizer needs to move, and a ``.data``
    assignment of an alias that shares ``out``'s storage, offset, sizes and
    strides swaps it while leaving the tensor's storage, layout and identity
    untouched. The caller's ``out`` object is mutated in place, and a strided
    or offset view keeps pointing at its own slice of the buffer.

    The alias is built with ``_empty_affine_quantized`` + ``set_`` because
    ``_make_per_tensor_quantized_tensor`` (and ``Tensor.to``) silently
    normalises the strides of a quantized tensor on CUDA, which would lose a
    channels-last layout.
    """
    alias = torch._empty_affine_quantized(
        0,
        scale=float(scale),
        zero_point=int(zero_point),
        dtype=out.dtype,
        device=out.device,
    )
    alias.set_(
        out.untyped_storage(),
        out.storage_offset(),
        tuple(out.shape),
        out.stride(),
    )
    out.data = alias
    return out


def _copy_int(dst_int, src_int):
    """Copy integers honouring both sides' strides (aten copy on the views)."""
    dst_int.copy_(src_int)
    return dst_int


def _linear_layout(a, out_int):
    """True when a flat element index addresses the same physical slot on both sides.

    The flat kernel walks memory as a linear array (``ptr + offsets``), which is
    only meaningful when the tensor's logical order *is* its physical order.
    ``is_nonoverlapping_and_dense`` covers the C-contiguous case and the
    channels-last one (a dense permutation of the same buffer), and is False for
    a gapped view such as ``t[:, ::2]`` or ``t.T``. With the two sides sharing a
    stride tuple, the flat index then lands on the same logical element in both.
    """
    return a.stride() == out_int.stride() and torch._debug_has_internal_overlap(a) == 0


def _fp32_reciprocal(scale):
    """aten's ``1.0f / (float)scale``, computed with an exact fp32 division."""
    return (
        torch.tensor(1.0, dtype=torch.float32)
        / torch.tensor(float(scale), dtype=torch.float32)
    ).item()


def _launch_quantize_kernel(a, out_int, scale, zero_point, qmin, qmax):
    """Run the quantization kernel, writing ``out_int`` through its own layout.

    ``a`` is read and ``out_int`` is written; both may be strided and both have
    the same shape. The flat kernel is used when the two share a linear memory
    order (the functional path materialises both in one format, so it always
    does); otherwise the strided kernel decomposes the task index against the
    logical shape and re-flattens each side with its own strides.
    """
    n_elements = out_int.numel()
    if n_elements == 0:
        return out_int
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    inv_scale = _fp32_reciprocal(scale)
    if _linear_layout(a, out_int):
        quantize_per_tensor_kernel[grid](
            a,
            out_int,
            inv_scale,
            int(zero_point),
            qmin=qmin,
            qmax=qmax,
            n_elements=n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        ndim = out_int.dim()
        if ndim > MAX_NDIM:
            raise RuntimeError(
                f"quantize_per_tensor: inputs with more than {MAX_NDIM} dimensions "
                f"are not supported, got {ndim}"
            )
        quantize_per_tensor_strided_kernel[grid](
            a,
            out_int,
            inv_scale,
            int(zero_point),
            *_padded(out_int.shape, ndim),
            *_padded(a.stride(), ndim),
            *_padded(out_int.stride(), ndim),
            qmin=qmin,
            qmax=qmax,
            n_elements=n_elements,
            ndim=ndim,
            IDX32=_int32_safe(a, out_int),
            BLOCK_SIZE=BLOCK_SIZE,
        )
    return out_int


def _suggest_memory_format(t):
    """Mirror of ``TensorImpl::suggest_memory_format`` for the dense cases.

    ATen keeps a channels-last (4-D) or channels-last-3d (5-D) input in that
    format. The check is: the dimension is present, the channel dimension is not
    degenerate, and the tensor is already dense in that format. Everything else
    -- including 1-D/2-D tensors, which reject an explicit ``memory_format=``
    -- falls back to the default contiguous format. Verified against the native op
    on contiguous, channels-last, channels-last-3d, C=1, N=1, transposed and
    sliced-channel-last inputs.
    """
    if (
        t.dim() == 4
        and t.shape[1] != 1
        and t.is_contiguous(memory_format=torch.channels_last)
    ):
        return torch.channels_last
    if (
        t.dim() == 5
        and t.shape[1] != 1
        and t.is_contiguous(memory_format=torch.channels_last_3d)
    ):
        return torch.channels_last_3d
    return torch.contiguous_format


def _quantize_per_tensor_impl(a, scale, zero_point, dtype):
    """Shared core: compute the quantized integer representation of ``a``.

    Returns the integer storage tensor (e.g. uint8 for quint8) holding
    ``clamp(rint(x * inv_scale) + zero_point, qmin, qmax)``.
    """
    storage_dtype, qmin, qmax = _validate_quant_args(a, dtype)

    # ATen materializes the input with ``rtensor.suggest_memory_format()`` and
    # allocates the quantized output in that same format, so a channels-last
    # (or channels-last-3d) input yields a channels-last int_repr rather than
    # being silently re-laid-out to contiguous. ``suggest_memory_format`` has no
    # Python binding, so mirror its 4-D/5-D rules here (see
    # ``TensorImpl::suggest_memory_format``): contiguous when the batch stride
    # does not fit the channels-last pattern or the channel count is too small,
    # channels-last otherwise.
    memory_format = _suggest_memory_format(a)
    if memory_format is torch.contiguous_format or a.dim() < 4:
        out = torch.empty(a.shape, dtype=storage_dtype, device=a.device)
        a = a.contiguous()
    else:
        out = torch.empty(
            a.shape, dtype=storage_dtype, device=a.device, memory_format=memory_format
        )
        a = a.contiguous(memory_format=memory_format)
    # One thread per element; 1024 is a standard block size that keeps the
    # launch grid small for the typical quantization buffer sizes.
    return _launch_quantize_kernel(a, out, scale, zero_point, qmin, qmax)


def _make_per_tensor_from_int(int_repr, scale, zero_point):
    """Wrap ``int_repr`` as a per-tensor quantized tensor, preserving its layout.

    ``torch._make_per_tensor_quantized_tensor`` silently normalises the
    strides of its argument to contiguous on CUDA, which would lose the
    channels-last (or channels-last-3d) layout the kernel produced from a
    channels-last input -- native keeps it. Building the tensor through
    ``_empty_affine_quantized`` + ``set_`` (the same trick
    :func:`_adopt_quantizer` uses) shares ``int_repr``'s storage and keeps
    its sizes, strides and layout untouched.
    """
    result = torch._empty_affine_quantized(
        0,
        scale=float(scale),
        zero_point=int(zero_point),
        dtype=_QDTYPE_FROM_INT[int_repr.dtype],
        device=int_repr.device,
    )
    result.set_(
        int_repr.untyped_storage(),
        int_repr.storage_offset(),
        tuple(int_repr.shape),
        int_repr.stride(),
    )
    return result


def quantize_per_tensor(a, scale, zero_point, dtype):
    logger.debug("GEMS QUANTIZE_PER_TENSOR")
    # The dtype is validated before the zero-point range (the upstream order
    # raises a bare KeyError for an unsupported dtype; the pinned oracle is
    # _validate_quant_args' RuntimeError).
    _validate_quant_args(a, dtype)
    _check_zero_point_range(zero_point, dtype)
    int_repr = _quantize_per_tensor_impl(a, scale, zero_point, dtype)
    return _make_per_tensor_from_int(int_repr, float(scale), int(zero_point))


def quantize_per_tensor_out(a, scale, zero_point, dtype, *, out=None):
    logger.debug("GEMS QUANTIZE_PER_TENSOR_OUT")
    _validate_quant_args(a, dtype)
    _check_zero_point_range(zero_point, dtype)
    if out is None:
        raise RuntimeError("quantize_per_tensor.out: argument 'out' is required")

    # ATen validates the out tensor's dtype and device before computing, with the
    # requested quantized dtype -- not the input dtype -- deciding what is legal.
    if out.dtype != dtype:
        raise RuntimeError(
            f"Expected out tensor to have dtype {dtype}, but got {out.dtype} instead"
        )
    if out.device != a.device:
        raise RuntimeError(
            f"Expected out tensor to have device {a.device}, but got {out.device} "
            "instead"
        )
    # A shape mismatch is a resize in ATen, but ``aten::resize_`` has no
    # QuantizedCUDA kernel, so the native op itself fails on this path. Reject it
    # with a clear message rather than silently producing a wrong-shaped result.
    if tuple(out.shape) != tuple(a.shape):
        raise RuntimeError(
            f"quantize_per_tensor.out: out has shape {tuple(out.shape)} but the "
            f"result requires {tuple(a.shape)}; resizing a quantized CUDA output "
            "is not supported"
        )
    # Validated before anything is written, so a rejected input leaves ``out``
    # untouched -- the same point in the sequence at which ATen's quantizer
    # raises.
    storage_dtype, qmin, qmax = _validate_quant_args(a, dtype)

    if out.qscheme() != torch.per_tensor_affine:
        # ATen's quantized ``copy_`` rejects a per-channel destination with this
        # message. The kernel writes integers and has no way to install a
        # per-tensor quantizer over a per-channel tensor, so the same rejection
        # is raised here instead of being supplied by the copy.
        raise RuntimeError("Quantized Copy only works with same qscheme")

    if has_internal_overlapping(out) == MemOverlap.Yes:
        # An ``out`` whose elements alias each other in memory (a stride-0
        # broadcast view) has no single correct value to hold; ATen rejects it
        # in the copy it would otherwise perform. Checking first keeps this from
        # silently producing whichever write happened last.
        raise RuntimeError(
            "unsupported operation: more than one element of the written-to "
            "tensor refers to a single memory location. Please clone() the "
            "tensor before performing the operation."
        )

    if _aliases(a, out):
        # ``out`` shares storage with the float input, so the kernel cannot
        # write into it while reading ``a``: the result is computed first, then
        # the integers are moved over the integer views (a quantized
        # ``out.copy_(result)`` would redispatch to ATen and run a second
        # native O(n) pass).
        result = _quantize_per_tensor_impl(a, scale, zero_point, dtype)
        if out.numel() != 0:
            _copy_int(_int_view(out), result)
        return _adopt_quantizer(out, scale, zero_point)

    # ``out`` cannot overlap the input, so the quantized integers can be stored
    # straight through ``out``'s own strides and storage offset -- the strided
    # kernel places exactly the elements ``out.copy_(result)`` would have, in
    # any layout, including a non-contiguous view, without disturbing the
    # neighbours in the surrounding buffer. ``a`` is read through its own
    # strides, so no host-side materialisation is needed either.
    _launch_quantize_kernel(a, _int_view(out), scale, zero_point, qmin, qmax)
    if _same_quantizer(out, scale, zero_point):
        return out
    return _adopt_quantizer(out, scale, zero_point)
