"""Triton implementation of torch.linalg.multi_dot.

Ported from upstream FlagGems ops/linalg_multi_dot.py.

Multiplies a chain of 1-D/2-D tensors with the optimal (minimum-flop)
parenthesization, mirroring aten::linalg_multi_dot semantics.

Porting notes (repo conventions):
  - @libentry stripped (plain @triton.jit); no torch device context
    manager; ``ext.program_id`` -> ``tl.program_id``.
  - Upstream's ``_matrix_multiply`` redispatches to aten::mm on CPU
    devices; that short-circuit is stripped so the Triton kernel launches
    unconditionally (repo convention for CPU-backend ports).
  - fp64: the backend's fp64 ``tl.dot`` accumulates at ~fp32 precision, so
    the IS_FP64 branch of ``_multi_dot_mm_kernel`` accumulates with
    explicit per-k outer products in fp64 instead of ``tl.dot`` (probe:
    fp64 tl.dot carries ~2.4e-7 error; exact fp64 FMA here).
"""

import logging
import warnings

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _multi_dot_mm_kernel(
    left,
    right,
    output,
    rows,
    columns,
    inner,
    stride_lm,
    stride_lk,
    stride_rk,
    stride_rn,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    IS_FP64: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_n = tl.cdiv(columns, BLOCK_N)
    pid_m = pid // grid_n
    pid_n = pid % grid_n
    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)

    accumulator = tl.zeros(
        (BLOCK_M, BLOCK_N), dtype=tl.float64 if IS_FP64 else tl.float32
    )
    for start_k in range(0, tl.cdiv(inner, BLOCK_K)):
        current_k = start_k * BLOCK_K + offsets_k
        left_values = tl.load(
            left + offsets_m[:, None] * stride_lm + current_k[None, :] * stride_lk,
            mask=(offsets_m[:, None] < rows) & (current_k[None, :] < inner),
            other=0.0,
        )
        right_values = tl.load(
            right + current_k[:, None] * stride_rk + offsets_n[None, :] * stride_rn,
            mask=(current_k[:, None] < inner) & (offsets_n[None, :] < columns),
            other=0.0,
        )
        if IS_FP64:
            # fp64 tl.dot accumulates at ~fp32 precision on this backend;
            # fold one k at a time with exact fp64 products instead.
            for kk in tl.static_range(BLOCK_K):
                l_col = tl.sum(
                    tl.where(offsets_k[None, :] == kk, left_values, 0.0), axis=1
                )
                r_row = tl.sum(
                    tl.where(offsets_k[:, None] == kk, right_values, 0.0), axis=0
                )
                accumulator += l_col[:, None] * r_row[None, :]
        else:
            accumulator += tl.dot(left_values, right_values, allow_tf32=False)

    tl.store(
        output + offsets_m[:, None] * stride_om + offsets_n[None, :] * stride_on,
        accumulator.to(output.dtype.element_ty),
        mask=(offsets_m[:, None] < rows) & (offsets_n[None, :] < columns),
    )


def _matrix_multiply(left, right, out=None):
    """Multiply without TF32 so results match ATen's multi_dot precision."""
    rows, inner = left.shape
    columns = right.shape[1]
    if out is None:
        out = torch.empty((rows, columns), dtype=left.dtype, device=left.device)
    grid = (triton.cdiv(rows, 32) * triton.cdiv(columns, 32),)
    _multi_dot_mm_kernel[grid](
        left,
        right,
        out,
        rows,
        columns,
        inner,
        left.stride(0),
        left.stride(1),
        right.stride(0),
        right.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=32,
        BLOCK_N=32,
        BLOCK_K=32,
        IS_FP64=left.dtype == torch.float64,
        num_warps=4,
    )
    return out


def _validate_and_prepare(tensors):
    num_tensors = len(tensors)
    if num_tensors < 2:
        raise RuntimeError(
            f"multi_dot(): expected at least 2 tensors but got {num_tensors}"
        )

    arrays = list(tensors)
    first = arrays[0]
    last = arrays[-1]
    if first.ndim not in (1, 2):
        raise RuntimeError(
            f"multi_dot(): the first tensor must be 1D or 2D but got {first.ndim}D"
        )
    if last.ndim not in (1, 2):
        raise RuntimeError(
            f"multi_dot(): the last tensor must be 1D or 2D but got {last.ndim}D"
        )

    for index, tensor in enumerate(arrays[1:-1], 1):
        if tensor.ndim != 2:
            raise RuntimeError(
                f"multi_dot(): tensor {index} must be 2D but got {tensor.ndim}D"
            )

    for index, tensor in enumerate(arrays[1:], 1):
        if tensor.dtype != first.dtype:
            raise RuntimeError(
                "multi_dot(): all tensors must have be the same dtype but "
                f"tensor 0 is {first.dtype} and tensor {index} {tensor.dtype}"
            )
        if tensor.device != first.device:
            raise RuntimeError(
                "multi_dot(): all tensors must be on the same device but "
                f"tensor 0 is on {first.device} and tensor {index} is on "
                f"{tensor.device}"
            )

    first_was_1d = first.ndim == 1
    last_was_1d = last.ndim == 1
    if first_was_1d:
        arrays[0] = first.unsqueeze(0)
    if last_was_1d:
        arrays[-1] = last.unsqueeze(1)

    for index in range(num_tensors - 1):
        if arrays[index].shape[1] != arrays[index + 1].shape[0]:
            raise RuntimeError(
                f"multi_dot(): tensors {index} and {index + 1} with shapes "
                f"{list(tensors[index].shape)} and {list(tensors[index + 1].shape)} "
                "cannot be multiplied"
            )

    output_shape = [arrays[0].shape[0], arrays[-1].shape[1]]
    if first_was_1d:
        output_shape.pop(0)
    if last_was_1d:
        output_shape.pop(-1)
    return arrays, tuple(output_shape)


def _matrix_chain_order(arrays):
    num_tensors = len(arrays)
    dimensions = [arrays[0].shape[0]] + [array.shape[1] for array in arrays]
    costs = [[0] * num_tensors for _ in range(num_tensors)]
    splits = [[0] * num_tensors for _ in range(num_tensors)]

    for chain_length in range(2, num_tensors + 1):
        for start in range(num_tensors - chain_length + 1):
            end = start + chain_length - 1
            best_cost = None
            for split in range(start, end):
                cost = (
                    costs[start][split]
                    + costs[split + 1][end]
                    + dimensions[start] * dimensions[split + 1] * dimensions[end + 1]
                )
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    splits[start][end] = split
            costs[start][end] = best_cost
    return splits


def _multiply_chain(arrays, splits, start, end, out=None):
    if start == end:
        return arrays[start]

    split = splits[start][end]
    left = _multiply_chain(arrays, splits, start, split)
    right = _multiply_chain(arrays, splits, split + 1, end)
    if out is not None:
        return _matrix_multiply(left, right, out=out)
    return _matrix_multiply(left, right)


def _multiply_three(arrays, out=None):
    a, b, c = arrays
    rows, inner_ab = a.shape
    inner_bc, columns = c.shape
    left_cost = rows * inner_bc * (inner_ab + columns)
    right_cost = inner_ab * columns * (rows + inner_bc)

    if left_cost > right_cost:
        right = _matrix_multiply(b, c)
        return _matrix_multiply(a, right, out=out)

    left = _matrix_multiply(a, b)
    return _matrix_multiply(left, c, out=out)


def _multi_dot_impl(arrays, out=None):
    num_tensors = len(arrays)
    if num_tensors == 2:
        return _matrix_multiply(arrays[0], arrays[1], out=out)
    if num_tensors == 3:
        return _multiply_three(arrays, out=out)

    splits = _matrix_chain_order(arrays)
    return _multiply_chain(arrays, splits, 0, num_tensors - 1, out=out)


def linalg_multi_dot(tensors):
    logger.debug("GEMS LINALG_MULTI_DOT")
    arrays, output_shape = _validate_and_prepare(tensors)
    result = _multi_dot_impl(arrays)
    return result.view(output_shape)


def linalg_multi_dot_out(tensors, *, out):
    logger.debug("GEMS LINALG_MULTI_DOT_OUT")
    arrays, output_shape = _validate_and_prepare(tensors)
    first = arrays[0]
    if out.dtype != first.dtype:
        raise RuntimeError(
            f"multi_dot(): expected out tensor to have dtype {first.dtype} "
            f"but got {out.dtype}"
        )
    if out.device != first.device:
        raise RuntimeError(
            f"multi_dot(): expected out tensor to be on device {first.device} "
            f"but got {out.device}"
        )
    if tuple(out.shape) != output_shape:
        if out.numel() != 0:
            warnings.warn(
                "An output with one or more elements was resized since it had "
                f"shape {list(out.shape)}, which does not match the required "
                f"output shape {list(output_shape)}. This behavior is deprecated, "
                "and in a future PyTorch release outputs will not be resized "
                "unless they have zero elements. You can explicitly reuse an out "
                "tensor t by resizing it, inplace, to zero elements with "
                "t.resize_(0).",
                UserWarning,
                stacklevel=2,
            )
        out.resize_(output_shape)

    matrix_out = out.view(arrays[0].shape[0], arrays[-1].shape[1])
    _multi_dot_impl(arrays, out=matrix_out)
    return out
