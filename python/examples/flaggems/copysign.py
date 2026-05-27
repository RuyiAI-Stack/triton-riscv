import torch
import triton
import triton.language as tl


@triton.jit
def copysign_kernel_tt(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    abs_x = tl.abs(x)
    num_bits: tl.constexpr = x.dtype.primitive_bitwidth
    uint_dtype = tl.core.get_int_dtype(num_bits, False)
    sign_bit_mask: tl.constexpr = 1 << (num_bits - 1)
    y_u = y.to(uint_dtype, bitcast=True)
    result = tl.where((y_u & sign_bit_mask) != 0, -abs_x, abs_x)
    tl.store(out_ptr + offsets, result, mask=mask)


@triton.jit
def copysign_kernel_ts(
    x_ptr,
    y_val,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    abs_x = tl.abs(x)
    num_bits: tl.constexpr = x.dtype.primitive_bitwidth
    uint_dtype = tl.core.get_int_dtype(num_bits, False)
    sign_bit_mask: tl.constexpr = 1 << (num_bits - 1)
    y_u = y_val.to(uint_dtype, bitcast=True)
    result = tl.where((y_u & sign_bit_mask) != 0, -abs_x, abs_x)
    tl.store(out_ptr + offsets, result, mask=mask)


def copysign(input, other, *, out=None):
    BLOCK_SIZE = 1024
    if isinstance(other, torch.Tensor):
        A, B = torch.broadcast_tensors(input, other)
        A_c = A.contiguous()
        B_c = B.contiguous()
        if out is None:
            out_t = torch.empty_like(A_c)
            out_c = out_t
        else:
            out_c = out if out.is_contiguous() else out.contiguous()
        n_elements = A_c.numel()
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        copysign_kernel_tt[grid](
            A_c, B_c, out_c, n_elements, BLOCK_SIZE=BLOCK_SIZE
        )
        if out is not None and out_c.data_ptr() != out.data_ptr():
            out.copy_(out_c)
        return out_t if out is None else out
    A_c = input.contiguous()
    if out is None:
        out_t = torch.empty_like(A_c)
        out_c = out_t
    else:
        out_c = out if out.is_contiguous() else out.contiguous()
    n_elements = A_c.numel()
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    copysign_kernel_ts[grid](
        A_c, other, out_c, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    if out is not None and out_c.data_ptr() != out.data_ptr():
        out.copy_(out_c)
    return out_t if out is None else out


def copysign_out(input, other, *, out):
    return copysign(input, other, out=out)
