import torch
import triton
import triton.language as tl

# sinc semantics: sinc(+-0) = 1, sinc(nan) = nan and both infinities
# give nan on torch CPU.  The kernel evaluates sin(pi x) / (pi x) with
# the pole special-cased; the fp32-rounded pi reproduces torch's small
# nonzero residue at the zeros of sin (probed: sinc(1) = -2.78e-8).


@triton.jit
def special_sinc_func(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)

    pi = 3.141592653589793
    xp = x * pi
    res = tl.where(x == 0.0, 1.0, tl.sin(xp) / xp)

    tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


def special_sinc(A):
    A_c = A.contiguous()
    out = torch.empty_like(A_c)
    n_elements = A_c.numel()
    if n_elements == 0:
        return out
    grid = (triton.cdiv(n_elements, 1024),)
    special_sinc_func[grid](A_c, out, n_elements, BLOCK_SIZE=1024)
    return out
