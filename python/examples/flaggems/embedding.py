import torch
import triton
import triton.language as tl


@triton.jit
def embedding_kernel(
    out_ptr,
    in_ptr,
    weight_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    out_ptr += pid * N
    in_ptr += pid

    mask = tl.arange(0, BLOCK_SIZE) < N
    cols = tl.arange(0, BLOCK_SIZE)

    row_idx = tl.load(in_ptr)
    weight_ptr += row_idx * N
    embedding_weight = tl.load(weight_ptr + cols, mask, other=0.0)
    tl.store(out_ptr + cols, embedding_weight, mask)


@triton.jit
def indice_freq_kernel(
    indices_freq,
    indices,
    elem_cnt: tl.constexpr,
    INDICE_BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * INDICE_BLOCK_SIZE

    offsets = block_start + tl.arange(0, INDICE_BLOCK_SIZE)
    mask = offsets < elem_cnt

    index_element = tl.load(indices + offsets, mask=mask)
    tl.atomic_add(indices_freq + index_element, 1, mask=mask)


@triton.jit(do_not_specialize=["padding_idx"])
def embedding_backward_kernel(
    grad_in,
    grad_out,
    indices,
    padding_idx,
    HAS_PADDING_IDX: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    grad_out += pid * N
    indices += pid

    mask = tl.arange(0, BLOCK_SIZE) < N
    cols = tl.arange(0, BLOCK_SIZE)

    row_idx = tl.load(indices).to(tl.int32)
    if not HAS_PADDING_IDX:
        grad_in += row_idx * N
        embedding_grad = tl.load(grad_out + cols, mask, other=0.0)
        if tl.constexpr(embedding_grad.dtype.is_bf16()):
            embedding_grad = embedding_grad.to(tl.float32)
        tl.atomic_add(grad_in + cols, embedding_grad, mask=mask)
    else:
        if row_idx != padding_idx:
            grad_in += row_idx * N
            embedding_grad = tl.load(grad_out + cols, mask, other=0.0)
            if tl.constexpr(embedding_grad.dtype.is_bf16()):
                embedding_grad = embedding_grad.to(tl.float32)
            tl.atomic_add(grad_in + cols, embedding_grad, mask=mask)


@triton.jit(do_not_specialize=["n_rows"])
def embedding_grad_scale_kernel(
    grad_out,
    indice_freq,
    n_rows,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)

    for row_idx in range(row_start, n_rows, row_step):
        embedding_scale = 1.0
        indice_freq_val = tl.load(indice_freq + row_idx)
        if indice_freq_val > 1:
            embedding_scale = 1.0 / indice_freq_val

        cols = tl.arange(0, BLOCK_SIZE)
        mask = tl.arange(0, BLOCK_SIZE) < N
        embedding_grad = tl.load(grad_out + row_idx * N + cols, mask=mask)
        scaled_embedding_grad = embedding_grad * embedding_scale
        tl.store(
            grad_out + row_idx * N + cols, scaled_embedding_grad, mask=mask
        )


@triton.jit
def embedding_backward_gather_kernel(
    grad_in,
    grad_out,
    indices,
    M,
    padding_idx,
    HAS_PADDING_IDX: tl.constexpr,
    SCALE_GRAD_BY_FREQ: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    col_mask = cols < N
    grad = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    frequency = tl.full((), 0, dtype=tl.int32)

    for index_offset in range(0, M):
        index_value = tl.load(indices + index_offset).to(tl.int32)
        matches = index_value == row_idx
        output_grad = tl.load(
            grad_out + index_offset * N + cols,
            mask=col_mask,
            other=0.0,
        ).to(tl.float32)
        grad += tl.where(matches, output_grad, 0.0)
        frequency += matches.to(tl.int32)

    if SCALE_GRAD_BY_FREQ:
        divisor = tl.where(frequency > 0, frequency, 1).to(tl.float32)
        grad /= divisor
    if HAS_PADDING_IDX:
        grad = tl.where(row_idx == padding_idx, 0.0, grad)

    tl.store(grad_in + row_idx * N + cols, grad, mask=col_mask)


def embedding(
    weight, indices, padding_idx=-1, scale_grad_by_freq=False, sparse=False
):
    assert not sparse, "Currently do not support sparse format"

    M = indices.numel()
    N = weight.shape[-1]

    BLOCK_SIZE = triton.next_power_of_2(N)
    indices = indices.contiguous()
    weight = weight.contiguous()
    output = torch.empty(
        (*indices.shape, N), device=indices.device, dtype=weight.dtype
    )

    grid = (M,)
    embedding_kernel[grid](output, indices, weight, N, BLOCK_SIZE)

    return output


def embedding_backward(
    grad_outputs,
    indices,
    num_weights,
    padding_idx=-1,
    scale_grad_by_freq=False,
    sparse=False,
):
    assert not sparse, "Currently do not support sparse format"

    grad_outputs = grad_outputs.contiguous()
    indices = indices.contiguous()
    M = indices.numel()
    N = grad_outputs.shape[-1]

    grad_inputs = torch.zeros(
        (num_weights, grad_outputs.shape[-1]),
        device=grad_outputs.device,
        dtype=(
            torch.float32
            if grad_outputs.dtype is torch.bfloat16
            else grad_outputs.dtype
        ),
    )

    BLOCK_SIZE = triton.next_power_of_2(N)
    HAS_PADDING_IDX = padding_idx is not None and padding_idx >= 0
    embedding_backward_gather_kernel[(num_weights,)](
        grad_inputs,
        grad_outputs,
        indices,
        M,
        padding_idx,
        HAS_PADDING_IDX,
        SCALE_GRAD_BY_FREQ=scale_grad_by_freq,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return (
        grad_inputs.to(torch.bfloat16)
        if grad_outputs.dtype is torch.bfloat16
        else grad_inputs
    )
