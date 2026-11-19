import torch
import triton
import triton.language as tl


@triton.jit
def and_tensor_kernel_tt(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    res = x & y
    tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def and_tensor_kernel_ts(
    x_ptr,
    y_val,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    res = x & y_val
    tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


def and_tensor(A, B):
    """Bitwise AND (aten ``__and__.Tensor``); for bool operands it is the
    logical AND. ``B`` may be a tensor or a Python scalar."""
    BLOCK_SIZE = 1024
    if isinstance(B, bool):
        # torch compares bool scalars as 0/1 in the tensor's domain.
        B = int(B)
    if isinstance(B, torch.Tensor):
        if A.dtype != B.dtype:
            # torch promotes mixed operand dtypes before the bitwise op.
            dt = torch.promote_types(A.dtype, B.dtype)
            A = A.to(dt)
            B = B.to(dt)
        A, B = torch.broadcast_tensors(A, B)
        A_c = A.contiguous()
        B_c = B.contiguous()
        out = torch.empty_like(A_c)
        n_elements = A_c.numel()
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        and_tensor_kernel_tt[grid](A_c, B_c, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        return out.view_as(A)
    else:
        A_c = A.contiguous()
        out = torch.empty_like(A_c)
        n_elements = A_c.numel()
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        and_tensor_kernel_ts[grid](A_c, B, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        return out.view_as(A)
