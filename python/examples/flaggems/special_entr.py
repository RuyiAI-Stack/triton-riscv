import torch
import triton
import triton.language as tl

# entr semantics (probed on torch CPU fp32): entr(x) = -x log(x) for
# x > 0, entr(+-0) = 0, entr(x < 0) = -inf (not nan), entr(inf) = -inf
# and entr(nan) = nan.  The -x log(x) form reproduces the -0.0 at x = 1.


@triton.jit
def special_entr_func(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)

    res = -x * tl.log(x)
    res = tl.where(x == 0.0, 0.0, res)
    res = tl.where(x < 0.0, float("-inf"), res)

    tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


def special_entr(A):
    A_c = A.contiguous()
    out = torch.empty_like(A_c)
    n_elements = A_c.numel()
    if n_elements == 0:
        return out
    grid = (triton.cdiv(n_elements, 1024),)
    special_entr_func[grid](A_c, out, n_elements, BLOCK_SIZE=1024)
    return out
