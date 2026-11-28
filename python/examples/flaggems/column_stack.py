from typing import List, Optional, Tuple, Union

import torch
import triton
import triton.language as tl

# column_stack: aten::column_stack -- 1-D inputs become columns, everything
# is concatenated along dim 1. Four upstream copy kernels are kept verbatim
# (per-part row/tile/big-tile copies for dense trailing layouts, and a
# rank<=4 strided kernel for arbitrary ``out``); two fast paths are dropped
# because this backend cannot compile them:
# - _copy_slots_kernel branched on a runtime scalar ``slot`` chain to pick
#   between four pointer arguments (rule 2a + 2b);
# - _copy_span_kernel iterated a runtime-bounded ``for`` loop.
# Both were pure fast paths; the per-part kernels produce identical values.
# The autotuner on the span kernel went with it.
#
# Wrapper behavior (validation order, error texts, dtype promotion via
# promote_types, out= handling with can_cast/resize) is upstream's, which
# already matches aten's messages for the empty list / size / rank errors.
# Bool tensors are copied through uint8 views (bit-identical), the repo
# convention for 1-byte Triton storage.

_BLOCK = 1024
_BLOCK_MIN = 64
_BLOCK_MAX = 4096
_TILE_MAX_ELEMS = 1 << 16
_TILE_LANES = 4096
# Big-part path: one program copies _BIG_TILE_ROWS contiguous rows.
_BIG_TILE_ROWS = 16
_BIG_TILE_MIN = 1 << 16
_I32_MAX = 1 << 31


def _pick_block(row_len):
    """Per-part program granularity: cover a whole row when it is small."""
    block = _BLOCK_MIN
    while block < row_len and block < _BLOCK_MAX:
        block <<= 1
    return block


@triton.jit(
    do_not_specialize=[
        "in_row_stride",
        "out_row_stride",
        "col_base",
        "row_len",
        "n_col_blocks",
    ]
)
def _copy_rows_kernel(
    in_ptr,
    out_ptr,
    in_row_stride,
    out_row_stride,
    col_base,
    row_len,
    n_col_blocks,
    EVEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Copy one part, viewed as a logical (rows, row_len) tile, into ``out``.

    Each program owns one contiguous run of ``BLOCK`` columns of a single
    logical row, so no per-element index division is needed: the program
    identifier is split into (row, column block) once.
    """
    pid = tl.program_id(0).to(tl.int64)
    row = pid // n_col_blocks
    col = (pid - row * n_col_blocks) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    src = in_ptr + row * in_row_stride + col
    dst = out_ptr + row * out_row_stride + col_base + col
    if EVEN:
        val = tl.load(src)
        tl.store(dst, val.to(out_ptr.dtype.element_ty))
    else:
        mask = col < row_len
        val = tl.load(src, mask=mask)
        tl.store(dst, val.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit(do_not_specialize=["in_row_stride", "out_row_stride", "col_base"])
def _copy_big_tile_kernel(
    in_ptr,
    out_ptr,
    in_row_stride,
    out_row_stride,
    col_base,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Tall unmasked tile (BLOCK_R rows x BLOCK_C columns) with 32-bit offsets.

    Large parts are pure bandwidth; addresses stay in int32, which is why
    this kernel is only used when the host proved every offset < 2**31.
    """
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    val = tl.load(in_ptr + r[:, None] * in_row_stride + c[None, :])
    tl.store(
        out_ptr + r[:, None] * out_row_stride + col_base + c[None, :],
        val.to(out_ptr.dtype.element_ty),
    )


@triton.jit(do_not_specialize=["in_row_stride", "out_row_stride", "col_base"])
def _copy_tile_kernel(
    in_ptr,
    out_ptr,
    in_row_stride,
    out_row_stride,
    col_base,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Unmasked 2-D tile copy: ``BLOCK_R`` whole rows of ``BLOCK_C`` columns.

    Used for small parts, where one program per row would spend all its time
    in per-program overhead. The grid exactly covers the tile space, so no
    masks are needed.
    """
    pid_r = tl.program_id(0).to(tl.int64)
    pid_c = tl.program_id(1).to(tl.int64)
    r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R).to(tl.int64)
    c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C).to(tl.int64)
    src = in_ptr + r[:, None] * in_row_stride + c[None, :]
    dst = out_ptr + r[:, None] * out_row_stride + col_base + c[None, :]
    val = tl.load(src)
    tl.store(dst, val.to(out_ptr.dtype.element_ty))


@triton.jit(
    do_not_specialize=[
        "n_elem",
        "base_out",
        "s1",
        "s2",
        "s3",
        "i0",
        "i1",
        "i2",
        "i3",
        "o0",
        "o1",
        "o2",
        "o3",
    ]
)
def _copy_generic_kernel(
    in_ptr,
    out_ptr,
    n_elem,
    base_out,
    s0,
    s1,
    s2,
    s3,
    i0,
    i1,
    i2,
    i3,
    o0,
    o1,
    o2,
    o3,
    BLOCK: tl.constexpr,
):
    """Strided copy of one part (rank <= 4) into a possibly strided ``out``."""
    pid = tl.program_id(0).to(tl.int64)
    idx = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = idx < n_elem
    rem = idx
    off_in = tl.zeros((BLOCK,), dtype=tl.int64)
    off_out = tl.zeros((BLOCK,), dtype=tl.int64) + base_out

    c3 = rem % s3
    rem = rem // s3
    off_in += c3 * i3
    off_out += c3 * o3

    c2 = rem % s2
    rem = rem // s2
    off_in += c2 * i2
    off_out += c2 * o2

    c1 = rem % s1
    rem = rem // s1
    off_in += c1 * i1
    off_out += c1 * o1

    off_in += rem * i0
    off_out += rem * o0

    val = tl.load(in_ptr + off_in, mask=mask)
    tl.store(out_ptr + off_out, val.to(out_ptr.dtype.element_ty), mask=mask)


def _rows_fast_path(t, ndim):
    """True when the trailing dims of ``t`` are exactly contiguous."""
    st = t.stride()
    acc = 1
    for d in range(ndim - 1, 0, -1):
        if st[d] != acc:
            return False
        acc *= t.shape[d]
    return True


def column_stack(
    tensors: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]],
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    parts = []
    for t in tensors:
        if t.dim() < 2:
            t = t.view(t.numel(), 1)
        parts.append(t)
    if not parts:
        raise RuntimeError("column_stack expects a non-empty TensorList")

    first = parts[0]
    ndim = first.dim()
    shape0 = first.shape
    out_shape = list(shape0)
    result_dtype = first.dtype
    width = first.size(1)
    for idx in range(1, len(parts)):
        t = parts[idx]
        tshape = t.shape
        if t.dim() != ndim:
            raise RuntimeError(
                f"Tensors must have same number of dimensions: got {ndim} and {t.dim()}"
            )
        for dim in range(ndim):
            if dim != 1 and tshape[dim] != shape0[dim]:
                raise RuntimeError(
                    f"Sizes of tensors must match except in dimension 1. "
                    f"Expected size {shape0[dim]} but got size {tshape[dim]} "
                    f"for tensor number {idx} in the list."
                )
        if t.dtype != result_dtype:
            result_dtype = torch.promote_types(result_dtype, t.dtype)
        width += t.size(1)
    out_shape[1] = width

    if out is None:
        out = torch.empty(out_shape, dtype=result_dtype, device=first.device)

    if not torch.can_cast(result_dtype, out.dtype):
        raise TypeError(
            "torch.cat(): input types can't be cast to the desired "
            f"output type {out.dtype}"
        )

    if list(out.shape) != out_shape:
        out.resize_(out_shape)
    if out.numel() == 0:
        return out

    bytewise = out.dtype == torch.bool
    out_k = out.view(torch.uint8) if bytewise else out

    rows = shape0[0]
    trailing = 1
    for dim in range(2, ndim):
        trailing *= shape0[dim]
    out_contig = out.is_contiguous()
    out_row_stride = out.numel() // rows if out_contig else 0

    col = 0
    for t in parts:
        n_elem = t.numel()
        width_t = t.size(1)
        if n_elem:
            t_k = t.view(torch.uint8) if t.dtype == torch.bool else t
            row_len = width_t * trailing
            if out_contig and _rows_fast_path(t, ndim):
                block = _pick_block(row_len)
                col_start = col * trailing
                if (
                    row_len % block == 0
                    and n_elem >= _BIG_TILE_MIN
                    and rows >= _BIG_TILE_ROWS
                    and out_row_stride * rows + col_start + row_len <= _I32_MAX
                    and n_elem <= _I32_MAX
                ):
                    br = _BIG_TILE_ROWS
                    while rows % br:
                        br //= 2
                    if br > 1:
                        _copy_big_tile_kernel[(rows // br, row_len // block)](
                            t_k,
                            out_k,
                            t.stride(0),
                            out_row_stride,
                            col_start,
                            BLOCK_R=br,
                            BLOCK_C=block,
                        )
                        col += width_t
                        continue
                if n_elem <= _TILE_MAX_ELEMS and row_len % block == 0:
                    max_br = _TILE_LANES // block
                    br = 1
                    while rows // br > 8 and br * 2 <= max_br:
                        br *= 2
                    while rows % br:
                        br //= 2
                    _copy_tile_kernel[(rows // br, row_len // block)](
                        t_k,
                        out_k,
                        t.stride(0),
                        out_row_stride,
                        col_start,
                        BLOCK_R=br,
                        BLOCK_C=block,
                    )
                else:
                    n_col_blocks = (row_len + block - 1) // block
                    _copy_rows_kernel[(rows * n_col_blocks,)](
                        t_k,
                        out_k,
                        t.stride(0),
                        out_row_stride,
                        col_start,
                        row_len,
                        n_col_blocks,
                        EVEN=(row_len % block == 0),
                        BLOCK=block,
                    )
            else:
                if ndim > 4:
                    raise NotImplementedError(
                        "column_stack_out supports at most 4 dimensions"
                    )
                ss = [1, 1, 1, 1]
                ii = [0, 0, 0, 0]
                oo = [0, 0, 0, 0]
                for dim in range(ndim):
                    ss[dim] = t.shape[dim]
                    ii[dim] = t.stride(dim)
                    oo[dim] = out.stride(dim)
                _copy_generic_kernel[(triton.cdiv(n_elem, _BLOCK),)](
                    t_k,
                    out_k,
                    n_elem,
                    col * out.stride(1),
                    ss[0],
                    ss[1],
                    ss[2],
                    ss[3],
                    ii[0],
                    ii[1],
                    ii[2],
                    ii[3],
                    oo[0],
                    oo[1],
                    oo[2],
                    oo[3],
                    BLOCK=_BLOCK,
                )
        col += width_t

    return out


def column_stack_out(
    tensors: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]],
    *,
    out: torch.Tensor,
) -> torch.Tensor:
    return column_stack(tensors, out=out)
