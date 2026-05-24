import torch
import triton
import triton.language as tl


@triton.jit
def t_copy_2d_kernel(
    in_ptr,
    out_ptr,
    in_stride_0,
    in_stride_1,
    out_stride_0,
    out_stride_1,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    i = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    j = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    i64 = i.to(tl.int64)[None, :]
    j64 = j.to(tl.int64)[:, None]

    mask = (i64 < N) & (j64 < M)

    in_offsets = j64 * in_stride_0 + i64 * in_stride_1
    out_offsets = i64 * out_stride_0 + j64 * out_stride_1

    x = tl.load(in_ptr + in_offsets, mask=mask)
    tl.store(out_ptr + out_offsets, x, mask=mask)


@triton.jit
def copy_1d_strided_kernel(
    in_ptr,
    out_ptr,
    in_stride,
    out_stride,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    offs64 = offs.to(tl.int64)
    in_idx = offs64 * in_stride
    out_idx = offs64 * out_stride
    x = tl.load(in_ptr + in_idx, mask=mask)
    tl.store(out_ptr + out_idx, x, mask=mask)


def _launch_t_copy_kernel(inp: torch.Tensor, out: torch.Tensor):
    assert inp.dtype == out.dtype, "dtype mismatch between input and output"

    dim = inp.dim()
    if dim == 0:
        n = 1
        grid = (triton.cdiv(n, 1),)
        copy_1d_strided_kernel[grid](
            inp,
            out,
            0,
            0,
            n,
            BLOCK_SIZE=1,
        )
    elif dim == 1:
        n = inp.numel()
        in_stride = inp.stride(0)
        out_stride = out.stride(0)
        assert out.numel() == n, "Output size mismatch for 1D t_copy"
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n, BLOCK_SIZE),)
        copy_1d_strided_kernel[grid](
            inp,
            out,
            in_stride,
            out_stride,
            n,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    elif dim == 2:
        M, N = inp.shape
        assert out.dim() == 2 and out.shape[0] == N and out.shape[1] == M, (
            "Output shape must be (input.size(1), input.size(0)) for t_copy"
        )
        in_s0, in_s1 = inp.stride()
        out_s0, out_s1 = out.stride()
        BLOCK_M = 32
        BLOCK_N = 32
        grid = (triton.cdiv(N, BLOCK_M), triton.cdiv(M, BLOCK_N))
        t_copy_2d_kernel[grid](
            inp,
            out,
            in_s0,
            in_s1,
            out_s0,
            out_s1,
            M,
            N,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )
    else:
        raise RuntimeError("t_copy expects a tensor with <= 2 dims")


def t_copy_out(
    input: torch.Tensor,
    out: torch.Tensor,
    memory_format: torch.memory_format | None = None,
):
    _launch_t_copy_kernel(input, out)
    return out


def t_copy(
    input: torch.Tensor, memory_format: torch.memory_format | None = None
):
    dim = input.dim()
    if dim == 0:
        out = torch.empty((), dtype=input.dtype, device=input.device)
    elif dim == 1:
        out = torch.empty_like(input, memory_format=torch.contiguous_format)
    elif dim == 2:
        M, N = input.shape
        out = torch.empty(
            (N, M),
            dtype=input.dtype,
            device=input.device,
            memory_format=torch.contiguous_format,
        )
    else:
        raise RuntimeError("t_copy expects a tensor with <= 2 dims")
    _launch_t_copy_kernel(input, out)
    return out
