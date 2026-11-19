import torch
import triton
import triton.language as tl

# round semantics: round-half-to-even (rint/nearbyint do not lower on
# this toolchain, so the tie detection is explicit floor arithmetic --
# the same construction proven in the fused fake-quant kernel), with a
# decimal scale for the decimals overloads.  inf and nan propagate
# through the same arithmetic (floor(inf) = inf, floor(nan) = nan).


@triton.jit
def _special_round_kernel(x_ptr, out_ptr, n_elements, scale, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)

    xs = x * scale
    f = tl.floor(xs)
    d = xs - f
    odd = (f - 2.0 * tl.floor(f * 0.5)) > 0.5
    r = tl.where(d > 0.5, f + 1.0, tl.where(d < 0.5, f, tl.where(odd, f + 1.0, f)))
    res = r / scale

    tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


def _run(A, out=None, *, decimals=0):
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
    _special_round_kernel[grid](
        A_c, out_c, n_elements, float(10.0**decimals), BLOCK_SIZE=1024
    )
    if out_c.data_ptr() != out.data_ptr():
        out.copy_(out_c)
    return out


def special_round(A, *, decimals=0):
    return _run(A, decimals=decimals)


def special_round_out(A, out, *, decimals=0):
    return _run(A, out=out, decimals=decimals)
