import torch
import triton
import triton.language as tl


@triton.jit
def triu_kernel(
    X,
    Y,
    M,
    N,
    diagonal,
    M_BLOCK_SIZE: tl.constexpr,
    N_BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid * M_BLOCK_SIZE + tl.arange(0, M_BLOCK_SIZE)[:, None]
    m_mask = row < M
    X += row * N
    Y += row * N

    for n_offset in range(0, N, N_BLOCK_SIZE):
        cols = n_offset + tl.arange(0, N_BLOCK_SIZE)[None, :]
        n_mask = cols < N
        mask = m_mask & n_mask

        x = tl.load(X + cols, mask, other=0.0)
        y = tl.where(row + diagonal <= cols, x, 0.0)
        tl.store(Y + cols, y, mask=mask)


@triton.jit
def triu_batch_kernel(
    X,
    Y,
    batch,
    MN,
    N,
    diagonal,
    BATCH_BLOCK_SIZE: tl.constexpr,
    MN_BLOCK_SIZE: tl.constexpr,
):
    batch_id = tl.program_id(0)
    mn_id = tl.program_id(1)
    row = batch_id * BATCH_BLOCK_SIZE + tl.arange(0, BATCH_BLOCK_SIZE)[:, None]
    batch_mask = row < batch
    X += row * MN
    Y += row * MN

    cols = mn_id * MN_BLOCK_SIZE + tl.arange(0, MN_BLOCK_SIZE)[None, :]
    mn_mask = cols < MN
    mask = batch_mask & mn_mask
    x = tl.load(X + cols, mask, other=0.0)
    m = cols // N
    n = cols % N
    y = tl.where(m + diagonal <= n, x, 0.0)
    tl.store(Y + cols, y, mask=mask)


def _check_batch_contiguous(tensor, allow_zero_stride=True):
    if tensor.is_contiguous():
        return True, tensor

    dims = tensor.dim()

    if dims >= 2:
        n = tensor.size(-1)
        stride_row, stride_col = tensor.stride(-2), tensor.stride(-1)

        if not (stride_col == 1 and stride_row == n):
            return False, tensor.contiguous()

    if allow_zero_stride and dims <= 3:
        return True, tensor

    expected_stride = tensor.size(-1) * tensor.size(-2)
    for i in range(dims - 3, -1, -1):
        if (
            allow_zero_stride
            and i == 0
            and (tensor.stride(i) == 0 or tensor.size(i) == 1)
        ):
            continue

        if tensor.stride(i) != expected_stride:
            return False, tensor.contiguous()

        expected_stride *= tensor.size(i)

    return True, tensor


def triu(A, diagonal=0):
    assert len(A.shape) > 1, "Input tensor must have at least 2 dimensions"

    _, A_input = _check_batch_contiguous(A, allow_zero_stride=False)

    out = torch.empty(
        A.shape,
        dtype=A.dtype,
        device=A.device,
        memory_format=torch.contiguous_format,
    )

    M, N = A_input.shape[-2:]

    if len(A_input.shape) == 2:
        M_BLOCK_SIZE = 32
        N_BLOCK_SIZE = 128
        grid = (triton.cdiv(M, M_BLOCK_SIZE),)
        triu_kernel[grid](
            A_input,
            out,
            M,
            N,
            diagonal,
            M_BLOCK_SIZE=M_BLOCK_SIZE,
            N_BLOCK_SIZE=N_BLOCK_SIZE,
        )
    else:
        batch = int(torch.numel(A_input) / M / N)
        B = A_input.view(batch, -1)
        BATCH_BLOCK_SIZE = 4
        MN_BLOCK_SIZE = 1024
        grid = (
            triton.cdiv(batch, BATCH_BLOCK_SIZE),
            triton.cdiv(M * N, MN_BLOCK_SIZE),
        )
        triu_batch_kernel[grid](
            B,
            out,
            batch,
            M * N,
            N,
            diagonal,
            BATCH_BLOCK_SIZE=BATCH_BLOCK_SIZE,
            MN_BLOCK_SIZE=MN_BLOCK_SIZE,
        )
        out = out.view(A.shape)

    return out


def triu_out(
    input: torch.Tensor,
    diagonal: int = 0,
    *,
    out: torch.Tensor = None,
):
    if out is None:
        return triu(input, diagonal)
    result = triu(input, diagonal)
    out.copy_(result)
    return out


def triu_(A, diagonal=0):
    assert len(A.shape) > 1, "Input tensor must have at least 2 dimensions"
    diagonal = int(diagonal)
    M, N = A.shape[-2:]

    can_use_directly, A_to_use = _check_batch_contiguous(A, allow_zero_stride=True)

    if not can_use_directly:
        result_temp = torch.empty_like(A_to_use, memory_format=torch.contiguous_format)

        if len(A.shape) == 2:
            M_BLOCK_SIZE = 32
            N_BLOCK_SIZE = 128
            grid = (triton.cdiv(M, M_BLOCK_SIZE),)
            triu_kernel[grid](
                A_to_use,
                result_temp,
                M,
                N,
                diagonal,
                M_BLOCK_SIZE=M_BLOCK_SIZE,
                N_BLOCK_SIZE=N_BLOCK_SIZE,
            )
        else:
            batch = int(torch.numel(A) / M / N)
            B = A_to_use.view(batch, -1)
            result_temp_flat = result_temp.view(batch, -1)
            BATCH_BLOCK_SIZE = 4
            MN_BLOCK_SIZE = 1024
            grid = (
                triton.cdiv(batch, BATCH_BLOCK_SIZE),
                triton.cdiv(M * N, MN_BLOCK_SIZE),
            )
            triu_batch_kernel[grid](
                B,
                result_temp_flat,
                batch,
                M * N,
                N,
                diagonal,
                BATCH_BLOCK_SIZE=BATCH_BLOCK_SIZE,
                MN_BLOCK_SIZE=MN_BLOCK_SIZE,
            )

        A.copy_(result_temp)
    else:
        if len(A.shape) == 2:
            M_BLOCK_SIZE = 32
            N_BLOCK_SIZE = 128
            grid = (triton.cdiv(M, M_BLOCK_SIZE),)
            triu_kernel[grid](
                A,
                A,
                M,
                N,
                diagonal,
                M_BLOCK_SIZE=M_BLOCK_SIZE,
                N_BLOCK_SIZE=N_BLOCK_SIZE,
            )
        else:
            batch = int(torch.numel(A) / M / N)
            B = A.view(batch, -1)
            BATCH_BLOCK_SIZE = 4
            MN_BLOCK_SIZE = 1024
            grid = (
                triton.cdiv(batch, BATCH_BLOCK_SIZE),
                triton.cdiv(M * N, MN_BLOCK_SIZE),
            )
            triu_batch_kernel[grid](
                B,
                B,
                batch,
                M * N,
                N,
                diagonal,
                BATCH_BLOCK_SIZE=BATCH_BLOCK_SIZE,
                MN_BLOCK_SIZE=MN_BLOCK_SIZE,
            )

    return A
