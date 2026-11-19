import torch
import triton
import triton.language as tl

# logit semantics (probed on torch CPU fp32): logit(0) = -inf,
# logit(1) = +inf, values outside [0, 1] = nan and nan stays nan -- all
# reproduced by log(x / (1 - x)) in plain IEEE arithmetic.  eps clamps
# the input into [eps, 1 - eps] before the transform (tl.where keeps
# NaN unclamped, unlike a fmax-based clamp).


@triton.jit
def special_logit_kernel(x_ptr, out_ptr, n_elements, lo, hi, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)

    x = tl.where(x < lo, lo, x)
    x = tl.where(x > hi, hi, x)
    res = tl.log(x / (1.0 - x))

    tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


def _run(A, eps=None, out=None):
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
    if eps is None:
        lo, hi = float("-inf"), float("inf")
    else:
        lo, hi = eps, 1.0 - eps
    grid = (triton.cdiv(n_elements, 1024),)
    special_logit_kernel[grid](A_c, out_c, n_elements, lo, hi, BLOCK_SIZE=1024)
    if out_c.data_ptr() != out.data_ptr():
        out.copy_(out_c)
    return out


def special_logit(input, eps=None, out=None):
    return _run(input, eps=eps, out=out)


def special_logit_out(input, eps=None, out=None):
    return _run(input, eps=eps, out=out)
