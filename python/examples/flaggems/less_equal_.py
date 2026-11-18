import torch
import triton
import triton.language as tl


@triton.jit
def less_equal_kernel__tt(
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
    res = x.to(tl.float32) <= y.to(tl.float32)
    tl.store(out_ptr + offsets, res, mask=mask)


@triton.jit
def less_equal_kernel__ts(
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
    res = x.to(tl.float32) <= y_val
    tl.store(out_ptr + offsets, res, mask=mask)


def less_equal_(A, B):
    if A.dtype != torch.bool:
        raise RuntimeError(
            "less_equal_ result type Bool can't be cast to the desired "
            "output type"
        )
    # Compute into a contiguous buffer, then copy back so the original
    # (possibly non-contiguous) tensor is updated in place.
    if isinstance(B, torch.Tensor):
        if torch.broadcast_shapes(A.shape, B.shape) != A.shape:
            raise RuntimeError(
                "the shape of the self tensor must equal the broadcast "
                "result shape for the in-place comparison"
            )
        A_c = A.contiguous()
        B_c = B.expand(A.shape).contiguous()
        out = torch.empty_like(A_c, dtype=torch.bool)
        n_elements = A_c.numel()
        if n_elements == 0:
            return A
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        less_equal_kernel__tt[grid](A_c, B_c, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    else:
        A_c = A.contiguous()
        out = torch.empty_like(A_c, dtype=torch.bool)
        n_elements = A_c.numel()
        if n_elements == 0:
            return A
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        less_equal_kernel__ts[grid](A_c, B, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    A.copy_(out)
    return A


def less_equal_scalar_(A, B):
    return less_equal_(A, B)
