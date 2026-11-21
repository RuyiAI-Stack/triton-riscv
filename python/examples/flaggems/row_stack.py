import torch
import triton
import triton.language as tl

# row_stack: aten::row_stack (vstack semantics) -- atleast_2d every input
# and concatenate along dim 0, preserving ATen's output memory-format
# derivation (contiguous or channels_last). Upstream copied through a single
# four-pointer kernel that selected between pointer arguments with
# tl.where(tensor_idx == k, ptr_k, ...), which aborts codegen on this
# backend; this port launches one plain copy kernel per tensor (rule 2b)
# with identical results, for any number of inputs.
#
# The overlap diagnostics in the out= path use torch's own
# _debug_has_internal_overlap / _C._overlaps instead of the flag_gems
# shape_utils helpers. Bool tensors are copied through uint8 views
# (bit-identical). Wrapper validation (messages, order), dtype promotion,
# memory-format handling and out= resize semantics are upstream's, which
# match aten's texts exactly (probed on torch 2.15.0+cpu). One deviation:
# upstream accepted a bare Tensor by iterating its rows after atleast_2d;
# aten rejects it with TypeError("row_stack(): argument 'tensors'
# (position 1) must be tuple of Tensors, not Tensor").

_BLOCK_SIZE = 1024


@triton.jit(do_not_specialize=["out_row_offset", "num_elems", "out_row_stride"])
def _row_stack_copy_kernel(
    src_ptr,
    out_ptr,
    out_row_offset,
    num_elems,
    out_row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    """Copy one input's dense run of rows into ``out`` at ``out_row_offset``.

    Valid for contiguous and channels-last layouts alike: each row is a
    dense run of ``row_width`` elements and rows tile back to back, so the
    destination is ``out_row_offset * out_row_stride + idx``.
    """
    pid = tl.program_id(0)
    idx = (pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)).to(tl.int64)
    mask = idx < num_elems
    row_stride_offset = out_row_offset.to(tl.int64) * out_row_stride
    val = tl.load(src_ptr + idx, mask=mask)
    tl.store(out_ptr + row_stride_offset + idx, val, mask=mask)


def _row_width(shape):
    """Number of elements in one row = product of the trailing dimensions."""
    width = 1
    for size in shape[1:]:
        width *= size
    return width


def _row_strides(shape):
    """Row-major contiguous strides, matching ATen for zero-sized dimensions.

    ATen skips the stride multiply for a zero-sized dimension, so ``(5, 0)``
    gets stride ``(1, 1)`` rather than ``(0, 1)``.  Reproducing that keeps the
    output metadata identical to the reference for empty inputs.
    """
    strides = [0] * len(shape)
    step = 1
    for dim in range(len(shape) - 1, -1, -1):
        strides[dim] = step
        if shape[dim] != 0:
            step *= shape[dim]
    return strides


def _channels_last_strides(shape):
    """channels_last (rank 4) / channels_last_3d (rank 5) strides."""
    if len(shape) == 4:
        _, c, h, w = shape
        return [c * h * w, 1, w * c, c]
    _, c, d, h, w = shape
    return [c * d * h * w, 1, h * w * c, w * c, c]


def _strides_for(shape, memory_format):
    """Strides ATen's allocator uses for ``shape`` in ``memory_format``.

    For channels-last the allocated strides are what the kernel needs: dim 0 is
    stepped by ``row_width`` (the product of the trailing sizes) and each row is
    a dense run of ``row_width`` elements, tiled back to back.
    """
    if memory_format == torch.channels_last and len(shape) == 4:
        return _channels_last_strides(shape)
    if memory_format == torch.channels_last_3d and len(shape) == 5:
        return _channels_last_strides(shape)
    return _row_strides(shape)


def _promote_dtype(tensors):
    dtype = tensors[0].dtype
    for tensor in tensors[1:]:
        if tensor.dtype != dtype:
            dtype = torch.promote_types(dtype, tensor.dtype)
    return dtype


def _suggest_memory_format(tensor):
    """Mirror ATen's ``TensorImpl::suggest_memory_format``."""
    from torch._prims_common import suggest_memory_format

    if tensor.layout != torch.strided:
        return torch.contiguous_format
    return suggest_memory_format(tensor)


def _derive_memory_format(tensors):
    """ATen's ``cat_compute_output_memory_format``: all inputs must agree."""
    fmt = None
    for tensor in tensors:
        current = _suggest_memory_format(tensor)
        if current == torch.contiguous_format:
            return torch.contiguous_format
        if fmt is not None and fmt != current:
            return torch.contiguous_format
        fmt = current
    return fmt if fmt is not None else torch.contiguous_format


def _allocate(shape, strides, dtype, device):
    return torch.empty_strided(tuple(shape), tuple(strides), dtype=dtype, device=device)


def _normalize_tensors(tensors):
    """``atleast_2d`` plus ATen-compatible validation."""
    if isinstance(tensors, torch.Tensor):
        raise TypeError(
            "row_stack(): argument 'tensors' (position 1) must be tuple of "
            "Tensors, not Tensor"
        )
    tensors = list(torch.atleast_2d(tensors))
    if len(tensors) == 0:
        raise RuntimeError("vstack expects a non-empty TensorList")

    device = tensors[0].device
    ndim = tensors[0].dim()
    for i, tensor in enumerate(tensors[1:], start=1):
        if tensor.device != device:
            raise RuntimeError(
                "Expected all tensors to be on the same device, but got "
                f"{tensor.device} and {device}"
            )
        if tensor.dim() != ndim:
            raise RuntimeError(
                "Tensors must have same number of dimensions: got "
                f"{ndim} and {tensor.dim()}"
            )
        for dim, (expected, got) in enumerate(zip(tensors[0].shape, tensor.shape)):
            if dim != 0 and expected != got:
                raise RuntimeError(
                    "Sizes of tensors must match except in dimension 0. "
                    f"Expected size {expected} but got size {got} for tensor "
                    f"number {i} in the list."
                )
    return tensors


def _cast_tensors(tensors, dtype, memory_format):
    """Bring every input to ``dtype`` with a row block dense in layout order.

    The kernel copies row blocks positionally, so each input must be dense in
    the *same* trailing layout as the output; a channels-last input that was
    flattened to contiguous order would scatter elements to wrong positions.
    """
    cast = []
    for tensor in tensors:
        if tensor.dtype != dtype:
            tensor = tensor.to(dtype)
        if not tensor.is_contiguous(memory_format=memory_format):
            tensor = tensor.contiguous(memory_format=memory_format)
        if tensor.dtype == torch.bool:
            tensor = tensor.view(torch.uint8)
        cast.append(tensor)
    return cast


def _run_kernel(tensors, output, row_width, out_row_stride):
    """Copy every input row block into ``output`` through the Triton kernel."""
    dst = output.view(torch.uint8) if output.dtype == torch.bool else output
    total_row_offset = 0
    for tensor in tensors:
        num_elems = tensor.shape[0] * row_width
        if num_elems > 0:
            grid = (triton.cdiv(num_elems, _BLOCK_SIZE),)
            _row_stack_copy_kernel[grid](
                tensor,
                dst,
                total_row_offset,
                num_elems,
                out_row_stride,
                BLOCK_SIZE=_BLOCK_SIZE,
            )
        total_row_offset += tensor.shape[0]
    return output


def _output_shape(tensors):
    total_rows = sum(tensor.shape[0] for tensor in tensors)
    shape = list(tensors[0].shape)
    shape[0] = total_rows
    return shape


def _prepare(tensors):
    """Shared validation/shape work for both entry points."""
    tensors = _normalize_tensors(tensors)
    memory_format = _derive_memory_format(tensors)
    dtype = _promote_dtype(tensors)
    # A zero-sized trailing dimension means zero work: ``stride(0)`` can still
    # be non-zero for such inputs (e.g. ``(3, 0)`` has stride ``(1, 1)``), so a
    # launch based on ``rows * stride(0)`` would run out of bounds.
    row_width = _row_width(tensors[0].shape)
    output_shape = _output_shape(tensors)
    return tensors, memory_format, dtype, row_width, output_shape


def row_stack(tensors: list):
    tensors, memory_format, dtype, row_width, output_shape = _prepare(tensors)
    device = tensors[0].device
    if row_width == 0 or output_shape[0] == 0:
        # Nothing to copy; still hand back a correctly shaped, format-correct
        # tensor rather than launching a kernel that would read out of bounds.
        return _allocate(
            output_shape, _strides_for(output_shape, memory_format), dtype, device
        )

    strides = _strides_for(output_shape, memory_format)
    output = _allocate(output_shape, strides, dtype, device)
    c_tensors = _cast_tensors(tensors, dtype, memory_format)
    return _run_kernel(c_tensors, output, row_width, strides[0])


def _check_out(out, output_shape, dtype, memory_format, tensors):
    """Validate/prepare ``out`` the way ATen's ``cat_out`` meta function does."""
    if not torch.can_cast(dtype, out.dtype):
        raise TypeError(
            "torch.cat(): input types can't be cast to the desired output type "
            f"{out.dtype}"
        )

    if list(out.shape) != list(output_shape):
        out.resize_(output_shape)
        # ``resize_`` always installs contiguous strides, whereas ATen's
        # ``set_output_raw_strided`` re-imposes the derived memory format; for
        # channels-last that changes the strides, so restore them here.
        strides = _strides_for(list(out.shape), memory_format)
        if tuple(out.stride()) != tuple(strides):
            out.as_strided_(tuple(out.shape), tuple(strides))

    if torch._debug_has_internal_overlap(out) == 2:
        raise RuntimeError(
            "unsupported operation: more than one element of the written-to "
            "tensor refers to a single memory location. Please clone() the "
            "tensor before performing the operation."
        )
    for tensor in tensors:
        if torch._C._overlaps(out, tensor):
            raise RuntimeError(
                "unsupported operation: some elements of the input tensor and "
                "the written-to tensor refer to a single memory location. "
                "Please clone() the tensor before performing the operation."
            )
    return out


def row_stack_out(tensors: list, *, out: torch.Tensor):
    tensors, memory_format, dtype, row_width, output_shape = _prepare(tensors)
    _check_out(out, output_shape, dtype, memory_format, tensors)

    if row_width == 0 or output_shape[0] == 0:
        return out

    c_tensors = _cast_tensors(tensors, dtype, memory_format)

    # Fast path: ``out`` stores rows as dense runs of ``row_width`` elements and
    # its trailing strides match the layout the kernel assumes. This holds for
    # contiguous and for channels-last outputs, so only an arbitrary strided
    # view falls through to the staging copy below.
    fast = out.dtype == dtype and tuple(out.stride()) == tuple(
        _strides_for(list(out.shape), memory_format)
    )
    if fast:
        return _run_kernel(c_tensors, out, row_width, out.stride(0))

    # Otherwise scatter through a dense staging buffer; ``copy_`` is
    # stride-aware and also honours ``out``'s dtype.
    strides = _strides_for(output_shape, memory_format)
    staging = _allocate(output_shape, strides, dtype, out.device)
    _run_kernel(c_tensors, staging, row_width, strides[0])
    out.copy_(staging)
    return out
