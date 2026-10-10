import torch
import triton
import triton.language as tl


@triton.jit
def logical_xor_kernel(
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
    res = x.to(tl.int1) ^ y.to(tl.int1)
    tl.store(out_ptr + offsets, res, mask=mask)


def logical_xor(A, B):
    assert isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor)
    broadcast_A, broadcast_B = torch.broadcast_tensors(A, B)
    n_elements = broadcast_A.numel()
    out = torch.empty_like(broadcast_A, dtype=torch.bool)
    if n_elements == 0:
        return out

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    A_c = broadcast_A.contiguous()
    B_c = broadcast_B.contiguous()
    logical_xor_kernel[grid](A_c, B_c, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out
