import triton
import triton.language as tl


@triton.jit
def threshold_kernel_(x_ptr, n_elements, threshold, value, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    out = tl.where(x <= threshold, value, x)
    tl.store(x_ptr + offsets, out, mask=mask)


def threshold_(A, threshold, value):
    n_elements = A.numel()
    A_c = A if A.is_contiguous() else A.contiguous()
    grid = (triton.cdiv(n_elements, 1024),)
    threshold_kernel_[grid](A_c, n_elements, threshold, value, BLOCK_SIZE=1024)
    if not A.is_contiguous():
        A.copy_(A_c)
    return A
