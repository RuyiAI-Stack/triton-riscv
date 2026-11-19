import torch
import triton
import triton.language as tl

# gammaln via the Lanczos approximation (g = 7, n = 9) with the
# reflection formula below 0.5 -- the same construction measured in the
# foreach-unary lgamma port (worst case 6.4e-4 including the reflection
# band).  Boundary contract (probed on torch CPU): poles at 0 and the
# negative integers give +inf, both infinities give +inf, nan stays nan.


@triton.jit
def _lgamma_pos(x):
    x = x - 1.0
    a = 0.99999999999980993
    a = a + 676.5203681218851 / (x + 1.0)
    a = a + -1259.1392167224028 / (x + 2.0)
    a = a + 771.32342877765313 / (x + 3.0)
    a = a + -176.61502916214059 / (x + 4.0)
    a = a + 12.507343278686905 / (x + 5.0)
    a = a + -0.13857109526572012 / (x + 6.0)
    a = a + 9.9843695780195716e-06 / (x + 7.0)
    a = a + 1.5056327351493116e-07 / (x + 8.0)
    t = x + 7.5
    return 0.9189385332046727 + (x + 0.5) * tl.log(t) - t + tl.log(a)


@triton.jit
def special_gammaln_func(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    pi = 3.141592653589793
    y = tl.where(
        x < 0.5,
        tl.log(pi / tl.abs(tl.sin(pi * x))) - _lgamma_pos(1.0 - x),
        _lgamma_pos(x),
    )
    y = tl.where(x == 0.0, float("inf"), y)
    y = tl.where(x < 0.0, tl.where(x == tl.floor(x), float("inf"), y), y)
    y = tl.where(x == float("inf"), float("inf"), y)
    y = tl.where(x == float("-inf"), float("inf"), y)
    y = tl.where(x != x, x, y)

    tl.store(out_ptr + offsets, y.to(out_ptr.dtype.element_ty), mask=mask)


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
    special_gammaln_func[grid](A_c, out_c, n_elements, BLOCK_SIZE=1024)
    if out_c.data_ptr() != out.data_ptr():
        out.copy_(out_c)
    return out


def special_gammaln(A):
    return _run(A)


def special_gammaln_out(A, *, out=None):
    return _run(A, out=out)
