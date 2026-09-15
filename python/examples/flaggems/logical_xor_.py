import triton
import triton.language as tl

from .logical_xor import logical_xor


@triton.jit
def logical_xor_inplace_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    out = x.to(tl.int1) ^ y.to(tl.int1)
    tl.store(x_ptr + offsets, out, mask=mask)


def logical_xor_(A, B):
    if A.is_contiguous() and B.is_contiguous() and A.shape == B.shape:
        n_elements = A.numel()
        if n_elements == 0:
            return A
        grid = (triton.cdiv(n_elements, 1024),)
        logical_xor_inplace_kernel[grid](A, B, n_elements, BLOCK_SIZE=1024)
        return A
    A.copy_(logical_xor(A, B))
    return A
