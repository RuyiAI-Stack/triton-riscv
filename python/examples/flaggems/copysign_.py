import torch
import triton
import triton.language as tl


@triton.jit
def _copysign_core(x, other):
    # Magnitude of x, sign of other; libdevice.copysign is not lowered on
    # this backend, so read the sign bit from the unsigned bitcast of other.
    abs_val = tl.abs(x)
    num_bits: tl.constexpr = x.dtype.primitive_bitwidth
    if num_bits == 16:
        is_neg = (other.to(tl.uint16, bitcast=True) & 0x8000) != 0
    else:
        is_neg = (other.to(tl.uint32, bitcast=True) & 0x80000000) != 0
    return tl.where(is_neg, -abs_val, abs_val)


@triton.jit
def copysign_kernel__tt(
    x_ptr,
    other_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    other = tl.load(other_ptr + offsets, mask=mask, other=1.0)

    tl.store(out_ptr + offsets, _copysign_core(x, other), mask=mask)


@triton.jit
def copysign_kernel__ts(
    x_ptr,
    other,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    tl.store(out_ptr + offsets, _copysign_core(x, other), mask=mask)


def copysign_(input: torch.Tensor, other):
    if input.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        # fp64 included: unsupported dtypes go to the aten op.
        return torch.ops.aten.copysign_(input, other)

    BLOCK_SIZE = 1024

    if isinstance(other, torch.Tensor):
        other = torch.broadcast_to(other, input.shape)
        input_c = input.contiguous()
        other_c = other.contiguous()
        out = torch.empty_like(input_c)
        n_elements = input_c.numel()
        if n_elements == 0:
            return input
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        copysign_kernel__tt[grid](
            input_c, other_c, out, n_elements, BLOCK_SIZE=BLOCK_SIZE
        )
    else:
        input_c = input.contiguous()
        out = torch.empty_like(input_c)
        n_elements = input_c.numel()
        if n_elements == 0:
            return input
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        copysign_kernel__ts[grid](
            input_c, other, out, n_elements, BLOCK_SIZE=BLOCK_SIZE
        )

    input.copy_(out)
    return input
