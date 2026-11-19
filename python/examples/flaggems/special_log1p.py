import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# log1p semantics: log1p(-1) = -inf, x < -1 = nan, -inf = nan and +-0
# maps to +-0.  libdevice.log1p lowers here but behaves as plain
# log(1 + x) for tiny x (probed: returns ~0 at 1e-8), losing every
# significant digit to cancellation, so |x| < 0.01 takes a 3-term
# Taylor branch (error ~x^4 / 4 <= 2.5e-10).


@triton.jit
def special_log1p_func(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)

    taylor = x - x * x * 0.5 + x * x * x * (1.0 / 3.0)
    res = tl.where(tl.abs(x) < 0.01, taylor, libdevice.log1p(x))

    tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


def _run(A, out=None):
    A_c = A.contiguous()
    if out is None:
        out = torch.empty_like(A_c)
        out_c = out
    else:
        out_c = out if out.is_contiguous() else out.contiguous()
    n_elements = A_c.numel()
    if n_elements == 0:
        if out_c.data_ptr() != out.data_ptr():
            out.copy_(out_c)
        return out
    grid = (triton.cdiv(n_elements, 1024),)
    special_log1p_func[grid](A_c, out_c, n_elements, BLOCK_SIZE=1024)
    if out_c.data_ptr() != out.data_ptr():
        out.copy_(out_c)
    return out


def special_log1p(A):
    return _run(A)


def special_log1p_out(A, out):
    return _run(A, out=out)
