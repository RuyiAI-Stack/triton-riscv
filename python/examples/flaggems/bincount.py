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
    bin_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    count = tl.zeros((1,), dtype=tl.int32)
    for block_start in range(0, N, BLOCK_SIZE):
        block_offsets = block_start + offsets
        mask = block_offsets < N
        indices = tl.load(inp_ptr + block_offsets, mask=mask, other=-1)
        matches = mask & (indices == bin_idx)
        count += tl.sum(matches.to(tl.int32))
    tl.store(out_ptr + bin_idx, tl.sum(count).to(tl.int64))


@triton.jit
def bincount_weights_kernel(
    inp_ptr,
    weights_ptr,
    out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    bin_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    count = tl.zeros((1,), dtype=tl.float64)
    for block_start in range(0, N, BLOCK_SIZE):
        block_offsets = block_start + offsets
        mask = block_offsets < N
        indices = tl.load(inp_ptr + block_offsets, mask=mask, other=-1)
        weights = tl.load(weights_ptr + block_offsets, mask=mask, other=0.0)
        count += tl.sum(tl.where(mask & (indices == bin_idx), weights, 0.0))
    tl.store(out_ptr + bin_idx, tl.sum(count))


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
            return torch.zeros(
                minlength, dtype=weights.dtype, device=inp.device
            )
        return torch.zeros(minlength, dtype=torch.int64, device=inp.device)

    if int(inp.min().item()) < 0:
        raise RuntimeError("bincount only supports non-negative inputs")

    max_val = int(inp.max().item())
    output_size = max(max_val + 1, minlength)

    inp = inp.contiguous()

    if weights is not None:
        assert weights.shape == inp.shape, (
            "weights must have same shape as input"
        )
        weights = weights.contiguous()

        weights_dtype = weights.dtype
        if weights_dtype in (torch.float16, torch.bfloat16):
            weights = weights.to(torch.float32)
            out = torch.zeros(
                output_size, dtype=torch.float32, device=inp.device
            )
        else:
            out = torch.zeros(
                output_size, dtype=weights.dtype, device=inp.device
            )

        BLOCK_SIZE = 1024
        grid = (output_size,)

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
        grid = (output_size,)
        bincount_kernel[grid](
            inp,
            out,
            N,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return out
