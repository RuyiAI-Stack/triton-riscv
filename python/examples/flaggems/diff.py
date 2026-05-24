from functools import reduce

import torch
import triton
import triton.language as tl


@triton.jit
def diff_kernel_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)

    row_start = pid_m * BLOCK_M
    row_offsets = row_start + tl.arange(0, BLOCK_M)
    row_mask = row_offsets < M

    output_N = N - 1

    for n_start in range(0, output_N, BLOCK_N):
        col_offsets = n_start + tl.arange(0, BLOCK_N)
        col_mask = col_offsets < output_N

        mask = row_mask[:, None] & col_mask[None, :]

        input_offsets_next = row_offsets[:, None] * N + (
            col_offsets[None, :] + 1
        )
        input_offsets_curr = row_offsets[:, None] * N + col_offsets[None, :]

        inp_next = tl.load(
            input_ptr + input_offsets_next, mask=mask, other=0.0
        )
        inp_curr = tl.load(
            input_ptr + input_offsets_curr, mask=mask, other=0.0
        )

        diff_val = inp_next - inp_curr

        output_offsets = row_offsets[:, None] * output_N + col_offsets[None, :]
        tl.store(output_ptr + output_offsets, diff_val, mask=mask)


@triton.jit
def diff_kernel_non_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    k_start = pid_k * BLOCK_K
    k_offsets = k_start + tl.arange(0, BLOCK_K)
    k_mask = k_offsets < K

    output_N = N - 1

    for n in range(output_N):
        input_offset_next = pid_m * N * K + (n + 1) * K + k_offsets
        input_offset_curr = pid_m * N * K + n * K + k_offsets

        inp_next = tl.load(
            input_ptr + input_offset_next, mask=k_mask, other=0.0
        )
        inp_curr = tl.load(
            input_ptr + input_offset_curr, mask=k_mask, other=0.0
        )

        diff_val = inp_next - inp_curr

        output_offset = pid_m * output_N * K + n * K + k_offsets
        tl.store(output_ptr + output_offset, diff_val, mask=k_mask)


def _diff_once(inp, dim):
    shape = list(inp.shape)
    ndim = inp.ndim
    dim = dim % ndim

    N = shape[dim]
    if N < 2:
        raise RuntimeError(
            f"diff requires at least 2 elements along dim {dim}, got {N}"
        )

    M = reduce(lambda x, y: x * y, shape[:dim], 1)
    K = reduce(lambda x, y: x * y, shape[dim + 1 :], 1)

    out_shape = list(shape)
    out_shape[dim] = N - 1
    out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)

    if K == 1:
        BLOCK_M = triton.next_power_of_2(min(32, M))
        BLOCK_N = triton.next_power_of_2(min(256, N - 1))
        grid = (triton.cdiv(M, BLOCK_M),)
        diff_kernel_inner[grid](
            out,
            inp,
            M,
            N,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )
    else:
        BLOCK_K = triton.next_power_of_2(min(256, K))
        grid = (M, triton.cdiv(K, BLOCK_K))
        diff_kernel_non_inner[grid](
            out,
            inp,
            M,
            N,
            K,
            BLOCK_M=1,
            BLOCK_K=BLOCK_K,
        )

    return out


def diff(
    inp,
    n: int = 1,
    dim: int = -1,
    prepend: torch.Tensor | None = None,
    append: torch.Tensor | None = None,
):
    if n == 0:
        return inp.clone()

    if n < 0:
        raise RuntimeError(f"diff expects n >= 0, got {n}")

    ndim = inp.ndim
    if ndim == 0:
        raise RuntimeError(
            "diff requires input to be at least one-dimensional"
        )

    dim = dim % ndim

    tensors_to_cat = []
    if prepend is not None:
        tensors_to_cat.append(prepend)
    tensors_to_cat.append(inp)
    if append is not None:
        tensors_to_cat.append(append)

    if len(tensors_to_cat) > 1:
        inp = torch.cat(tensors_to_cat, dim=dim)

    inp = inp.contiguous()

    result = inp
    for _ in range(n):
        if result.shape[dim] < 2:
            raise RuntimeError("diff requires at least 2 elements along dim")
        result = _diff_once(result, dim)

    return result
