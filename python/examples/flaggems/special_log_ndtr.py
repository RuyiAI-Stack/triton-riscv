import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# log_ndtr, following ATen's calc_log_ndtr: for x >= -1 it is
# log1p(-erfc(t)/2) on t = x/sqrt(2); the deep tail uses the erfcx
# rearrangement log(erfcx(-t)/2) - t^2 so the log keeps relative accuracy
# where ndtr itself underflows.  erfc / erfcx have no lowering here, so
# both use the ported hybrids (1 - erf below the crossover, asymptotic
# series above; erfcx(t) for t >= 2.5 is 1/(sqrt(pi) t) times its own
# series, erfcx(-t) = 2 e^{t^2} - erfcx(t)).
# Boundary contract (probed on torch CPU): log_ndtr(-inf) = -inf,
# log_ndtr(0) = -log 2, log_ndtr(+inf) = -0.0, nan stays nan.


@triton.jit
def special_log_ndtr_func(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)

    t = x * 0.70710678118654752440

    # erfc(t) hybrid for the near branch.
    s0 = tl.zeros((BLOCK_SIZE,), tl.float32) + 1.0
    tt = s0
    for k in range(1, 12):
        tt = tt * (-(2.0 * k - 1.0) / (2.0 * t * t))
        s0 = s0 + tt
    asym = tl.exp(-t * t) / (t * 1.7724538509055159) * s0
    erfc_t = tl.where(t > 2.85, asym, 1.0 - libdevice.erf(t))
    near = libdevice.log1p(-erfc_t * 0.5)

    # tail branch (x < -1, t > 0): log_ndtr = -t^2 + log(erfcx(t)/2).
    # Working through erfcx(t) itself keeps the whole range representable:
    # erfc(t) underflows for t > ~10.2 and 2 e^{t^2} overflows, but
    # erfcx(t) stays O(1/t) -- so the log never sees 0 or inf until the
    # fp32 floor at ndtr ~ e^-104.
    at = tl.abs(t)
    u = 1.0 / (2.0 * at * at)
    sx = 1.0 - u * (
        1.0 - u * (3.0 - u * (15.0 - u * (105.0 - u * (945.0 - u * 10395.0))))
    )
    a64 = at.to(tl.float64)
    erfcx_mid = tl.exp(a64 * a64) * (1.0 - libdevice.erf(a64))
    erfcx_t = tl.where(
        at < 1.5,
        tl.exp(at * at) * (1.0 - libdevice.erf(at)),
        tl.where(at < 4.2, erfcx_mid.to(tl.float32), 0.5641895835477563 / at * sx),
    )
    tail = -t * t + tl.log(erfcx_t * 0.5)

    res = tl.where(x < -1.0, tail, near)
    res = tl.where(x != x, x, res)

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
    special_log_ndtr_func[grid](A_c, out_c, n_elements, BLOCK_SIZE=1024)
    if out_c.data_ptr() != out.data_ptr():
        out.copy_(out_c)
    return out


def special_log_ndtr(A):
    return _run(A)


def special_log_ndtr_(A):
    res = _run(A)
    A.copy_(res)
    return A
