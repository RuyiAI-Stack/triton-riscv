import torch
import triton
import triton.language as tl

from .embedding import embedding_backward


@triton.jit
def _embedding_dense_backward_kernel(
    grad_output_ptr,
    indices_ptr,
    grad_weight_ptr,
    num_weights,
    padding_idx,
    BLOCK_D: tl.constexpr,
    EMBED_DIM: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < EMBED_DIM

    idx = tl.load(indices_ptr + pid_n)
    valid = (idx != padding_idx) & (idx >= 0) & (idx < num_weights)

    go_ptrs = grad_output_ptr + pid_n * EMBED_DIM + offs_d
    go = tl.load(go_ptrs, mask=mask_d, other=0).to(tl.float32)

    gw_ptrs = grad_weight_ptr + idx * EMBED_DIM + offs_d
    mask = mask_d & valid
    tl.atomic_add(gw_ptrs, go, mask=mask)


@triton.jit
def _embedding_dense_backward_count_kernel(
    indices_ptr,
    counts_ptr,
    N,
    num_weights,
    padding_idx,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N
    idx = tl.load(indices_ptr + offs, mask=mask, other=0).to(tl.int32)
    valid = mask & (idx != padding_idx) & (idx >= 0) & (idx < num_weights)
    tl.atomic_add(counts_ptr + idx, 1, mask=valid)


@triton.jit
def _embedding_dense_backward_kernel_scale_by_freq(
    grad_output_ptr,
    indices_ptr,
    counts_ptr,
    grad_weight_ptr,
    num_weights,
    padding_idx,
    BLOCK_D: tl.constexpr,
    EMBED_DIM: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < EMBED_DIM

    idx = tl.load(indices_ptr + pid_n).to(tl.int32)
    valid = (idx != padding_idx) & (idx >= 0) & (idx < num_weights)

    go_ptrs = grad_output_ptr + pid_n * EMBED_DIM + offs_d
    go = tl.load(go_ptrs, mask=mask_d, other=0.0)

    cnt = tl.load(counts_ptr + idx, mask=valid, other=1)
    go = go / cnt

    gw_ptrs = grad_weight_ptr + idx * EMBED_DIM + offs_d
    mask = mask_d & valid
    tl.atomic_add(gw_ptrs, go, mask=mask)


def embedding_dense_backward(
    grad_output: torch.Tensor,
    indices: torch.Tensor,
    num_weights: int,
    padding_idx: int,
    scale_grad_by_freq: bool,
):
    assert indices.dtype in (
        torch.int32,
        torch.int64,
    ), "Indices must be int32 or int64."

    assert grad_output.dim() >= 2, (
        "grad_output must have embedding dimension as the last dim."
    )

    return embedding_backward(
        grad_output,
        indices,
        num_weights,
        padding_idx=padding_idx,
        scale_grad_by_freq=scale_grad_by_freq,
    )
