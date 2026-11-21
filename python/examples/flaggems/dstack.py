import logging

import torch
import triton
import triton.language as tl

# dstack: aten::dstack -- atleast_3d every input and concatenate along the
# depth axis (dim 2). Each input is one contiguous block of the output, so a
# single flat copy kernel per input covers the op (one launch per pointer;
# the upstream four-pointer token kernel selected between pointers inside
# the kernel, which this backend cannot compile).
#
# Deviations from upstream, pinned to the aten oracle (torch 2.15.0+cpu):
# - a bare Tensor argument now raises aten's TypeError; a non-Tensor list
#   element raises aten's "expected Tensor as element i in argument 0".
# - shape validation now walks the promoted atleast_3d shapes dim by dim
#   (dim 0, dim 1, then each trailing dim) raising aten's exact
#   "Sizes of tensors must match except in dimension 2 ..." text, and a
#   rank mismatch raises "Tensors must have same number of dimensions";
#   upstream only compared a packed (d0, d1, prod-trailing) triple.
# - the token interleaving fast path is dropped (pure fast path; values
#   identical through the copy kernel).

logger = logging.getLogger(__name__)

_BLOCK_WIDE = 1024
_BLOCK_NARROW = 2048

_REAL_OF_COMPLEX = {
    torch.complex32: torch.float16,
    torch.complex64: torch.float32,
    torch.complex128: torch.float64,
}


@triton.jit(do_not_specialize=["cnt", "seg", "out_row", "out_off"])
def _dstack_copy_kernel(
    out_ptr, in_ptr, cnt, seg, out_row, out_off, BLOCK: tl.constexpr
):
    """Copy ``cnt`` contiguous copy units of one input into its output slice.

    ``g`` is the flat copy-unit index within the input; ``seg = K_i * inner``
    units form one outer row, so ``o = g // seg`` and ``t = g - o * seg`` give
    the destination row and slot inside the output slice.
    """
    g = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = g < cnt
    o = g // seg
    t = g - o * seg
    tl.store(
        out_ptr + o * out_row + out_off + t,
        tl.load(in_ptr + g, mask=mask),
        mask=mask,
    )


def _prod(values) -> int:
    out = 1
    for value in values:
        out *= int(value)
    return int(out)


def _atleast_3d(shape):
    """atleast_3d promotion: 0-D -> (1,1,1), 1-D -> (1,N,1), 2-D -> (M,N,1)."""
    dims = [int(x) for x in shape]
    if len(dims) == 0:
        return [1, 1, 1]
    if len(dims) == 1:
        return [1, dims[0], 1]
    if len(dims) == 2:
        return [dims[0], dims[1], 1]
    return dims


def _real_dtype(dtype):
    return _REAL_OF_COMPLEX.get(dtype)


def _as_copy_tensor(tensor, real_dtype):
    """Zero-copy reinterpretation of ``tensor`` in the copy element dtype."""
    if real_dtype is None:
        return tensor
    if tensor.dim() == 0:
        # A 0-D complex tensor cannot be reinterpreted in place (the element
        # size changes); materialise the single element as a 1-D view first.
        return tensor.reshape(1).view(real_dtype)
    return tensor.view(real_dtype)


def _validate_shapes(first, shape, idx):
    if len(shape) != len(first):
        raise RuntimeError(
            f"Tensors must have same number of dimensions: got "
            f"{len(first)} and {len(shape)}"
        )
    for d in list(range(2)) + list(range(3, len(first))):
        if shape[d] != first[d]:
            raise RuntimeError(
                f"Sizes of tensors must match except in dimension 2. "
                f"Expected size {first[d]} but got size {shape[d]} for tensor "
                f"number {idx} in the list."
            )


def dstack(tensors, out=None, **kwargs):
    """``aten::dstack(Tensor[] tensors) -> Tensor``, plus the ``out=`` overload."""
    del kwargs
    if isinstance(tensors, torch.Tensor):
        raise TypeError(
            "dstack(): argument 'tensors' (position 1) must be tuple of "
            "Tensors, not Tensor"
        )
    items = list(tensors)
    if len(items) == 0:
        raise RuntimeError("dstack expects a non-empty TensorList")
    for i, item in enumerate(items):
        if not isinstance(item, torch.Tensor):
            raise TypeError(
                f"expected Tensor as element {i} in argument 0, "
                f"but got {type(item).__name__}"
            )

    # The kernel addresses inputs with a flat linear index, which only
    # matches logical element order when the tensor is contiguous.
    items = [item if item.is_contiguous() else item.contiguous() for item in items]

    promoted = [_atleast_3d(item.shape) for item in items]
    first = promoted[0]
    outer = first[0] * first[1]
    inner = _prod(first[3:])
    for idx, shape in enumerate(promoted[1:], start=1):
        _validate_shapes(first, shape, idx)

    depth = [shape[2] for shape in promoted]
    total = sum(depth)
    result_shape = [first[0], first[1], total, *first[3:]]
    dtype = items[0].dtype
    for item in items[1:]:
        dtype = torch.promote_types(dtype, item.dtype)
    result = (
        torch.empty(result_shape, dtype=dtype, device=items[0].device)
        if out is None
        else out
    )
    # Cast every input to the promoted dtype so the copy-element width used by
    # the kernels matches the one implied by the result (a real input promoted
    # to complex must be read as two copy units, not one).
    items = [item.to(dtype=dtype).contiguous() for item in items]

    real_dtype = _real_dtype(result.dtype)
    unit = 1 if real_dtype is None else 2
    if result.numel() == 0:
        return result

    out_view = _as_copy_tensor(result, real_dtype)

    out_row = total * inner * unit
    offset = 0
    for item, depth_i in zip(items, depth):
        if item.numel() > 0:
            source = _as_copy_tensor(item, _real_dtype(item.dtype))
            seg = depth_i * inner * unit
            count = outer * seg
            if count > 0:
                # Short rows (one slice per outer row) are bound by the number
                # of live programs, so they take the wider block; longer runs
                # were measured to prefer the smaller one.
                block = _BLOCK_WIDE if seg <= 4 else _BLOCK_NARROW
                grid = (triton.cdiv(count, block),)
                _dstack_copy_kernel[grid](
                    out_view,
                    source,
                    count,
                    seg,
                    out_row,
                    offset * inner * unit,
                    BLOCK=block,
                )
        offset += depth_i
    return result
