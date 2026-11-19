import torch
import triton
import triton.language as tl

# expit(x) = 1 / (1 + exp(-x)); the closed form saturates to 1 / 0 on
# both infinities instead of producing inf or nan.


@triton.jit
def special_expit_forward(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)

    res = 1.0 / (1.0 + tl.exp(-x))

    tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


def special_expit(A):
    A_c = A.contiguous()
    out = torch.empty_like(A_c)
    n_elements = A_c.numel()
    if n_elements == 0:
        return out
    grid = (triton.cdiv(n_elements, 1024),)
    special_expit_forward[grid](A_c, out, n_elements, BLOCK_SIZE=1024)
    return out
