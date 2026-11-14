import torch
import triton
import triton.language as tl


@triton.jit
def nextafter_kernel__tt(
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

    # IEEE 754 nextafter via the sign-magnitude bit pattern; see nextafter.py.
    x_int = x.to(tl.uint32, bitcast=True)
    other_int = other.to(tl.uint32, bitcast=True)

    x_is_nan = ((x_int & 0x7F800000) == 0x7F800000) & ((x_int & 0x007FFFFF) != 0)
    other_is_nan = other != other
    is_nan = x_is_nan | other_is_nan
    is_equal = x == other

    is_positive = (x_int & 0x80000000) == 0
    is_going_up = x < other

    uint_zero = x_int & 0
    uint_one = uint_zero | 1
    uint_neg_one = ~uint_zero
    cross_const = uint_zero | 0x80000001
    nan_bits = uint_zero | 0x7FC00000

    normal_inc = tl.where(
        is_positive,
        tl.where(is_going_up, uint_one, uint_neg_one),
        tl.where(is_going_up, uint_neg_one, uint_one),
    )
    is_zero_cross = ((x_int == uint_zero) & (~is_going_up)) | (
        (x_int == uint_zero | 0x80000000) & is_going_up
    )

    result = tl.where(
        is_nan,
        nan_bits,
        tl.where(
            is_equal,
            # torch returns `other` when x == other; only the signed
            # zero bit patterns differ between the two.
            other_int,
            tl.where(is_zero_cross, x_int + cross_const, x_int + normal_inc),
        ),
    )

    tl.store(out_ptr + offsets, result.to(tl.float32, bitcast=True), mask=mask)


def nextafter_(input: torch.Tensor, other: torch.Tensor):
    if input.dtype != torch.float32 or (
        isinstance(other, torch.Tensor) and other.dtype != input.dtype
    ):
        return torch.ops.aten.nextafter_(input, other)

    other = torch.broadcast_to(other, input.shape)
    input_c = input.contiguous()
    other_c = other.contiguous()
    out = torch.empty_like(input_c)

    n_elements = input_c.numel()
    if n_elements == 0:
        return input

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    nextafter_kernel__tt[grid](input_c, other_c, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    input.copy_(out)
    return input
