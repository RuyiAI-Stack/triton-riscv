import torch
import triton
import triton.language as tl

# Modified Bessel I0 (Numerical Recipes form): the even polynomial in
# x/3.75 below 3.75 and exp(|x|)/sqrt(|x|) times the asymptotic
# polynomial in 3.75/|x| above.  I0(+-0) = 1 exactly, and non-finite
# input lands in the large branch where inf/sqrt(inf) yields nan,
# matching torch on this build (probed).


@triton.jit
def _special_i0_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    ax = tl.abs(x)

    t = x / 3.75
    y = t * t
    p_small = 1.0 + y * (
        3.5156229
        + y
        * (
            3.0899424
            + y * (1.2067492 + y * (0.2659732 + y * (0.0360768 + y * 0.0045813)))
        )
    )

    yb = 3.75 / ax
    p_big = 0.39894228 + yb * (
        0.01328592
        + yb
        * (
            0.00225319
            + yb
            * (
                -0.00157565
                + yb
                * (
                    0.00916281
                    + yb
                    * (
                        -0.02057706
                        + yb * (0.02635537 + yb * (-0.01647633 + yb * 0.00392377))
                    )
                )
            )
        )
    )
    res_big = tl.exp(ax) * p_big / tl.sqrt(ax)

    res = tl.where(ax <= 3.75, p_small, res_big)

    tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


def _run(x, out=None):
    x_c = x.contiguous()
    if out is None:
        out = torch.empty_like(x_c)
        out_c = out
    else:
        out_c = out if out.is_contiguous() else out.contiguous()
    n_elements = x_c.numel()
    if n_elements == 0:
        if out_c.data_ptr() != out.data_ptr():
            out.copy_(out_c)
        return out
    grid = (triton.cdiv(n_elements, 1024),)
    _special_i0_kernel[grid](x_c, out_c, n_elements, BLOCK_SIZE=1024)
    if out_c.data_ptr() != out.data_ptr():
        out.copy_(out_c)
    return out


def special_i0(x):
    return _run(x)


def special_i0_out(x, out):
    return _run(x, out=out)
