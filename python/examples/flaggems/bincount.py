import torch
import triton
import triton.language as tl


@triton.jit
def bincount_kernel(
    inp_ptr,
    out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    indices = tl.load(inp_ptr + offsets, mask=mask, other=0)

    ones = tl.full((BLOCK_SIZE,), 1, dtype=tl.int64)
    tl.atomic_add(out_ptr + indices, ones, mask=mask, sem="relaxed")


@triton.jit
def bincount_weights_kernel(
    inp_ptr,
    weights_ptr,
    out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    indices = tl.load(inp_ptr + offsets, mask=mask, other=0)
    weights = tl.load(weights_ptr + offsets, mask=mask, other=0.0)

    tl.atomic_add(out_ptr + indices, weights, mask=mask, sem="relaxed")


def bincount(inp, weights=None, minlength=0):
    assert inp.ndim == 1, "bincount only supports 1-d tensors"
    assert inp.dtype in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ), "bincount only supports integer tensors"

    N = inp.numel()

    if N == 0:
        if weights is not None:
            return torch.zeros(minlength, dtype=weights.dtype, device=inp.device)
        return torch.zeros(minlength, dtype=torch.int64, device=inp.device)

    max_val = int(inp.max().item())
    output_size = max(max_val + 1, minlength)

    inp = inp.contiguous()

    if weights is not None:
        assert weights.shape == inp.shape, "weights must have same shape as input"
        weights = weights.contiguous()

        weights_dtype = weights.dtype
        if weights_dtype in (torch.float16, torch.bfloat16):
            weights = weights.to(torch.float32)
            out = torch.zeros(output_size, dtype=torch.float32, device=inp.device)
        else:
            out = torch.zeros(output_size, dtype=weights.dtype, device=inp.device)

        BLOCK_SIZE = 1024
        grid = (triton.cdiv(N, BLOCK_SIZE),)

        bincount_weights_kernel[grid](
            inp,
            weights,
            out,
            N,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        if weights_dtype in (torch.float16, torch.bfloat16):
            out = out.to(weights_dtype)

        return out
    else:
        out = torch.zeros(output_size, dtype=torch.int64, device=inp.device)
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(N, BLOCK_SIZE),)
        bincount_kernel[grid](
            inp,
            out,
            N,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return out
