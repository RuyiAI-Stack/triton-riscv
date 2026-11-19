import torch
import triton
import triton.language as tl

# Modified Bessel I1 (Cephes / Numerical Recipes form, from upstream
# FlagGems): the odd polynomial x * P((x/3.75)^2) below 3.75 and
# exp(|x|)/sqrt(|x|) times the asymptotic polynomial in 3.75/|x| above,
# with the odd sign restored.  Non-finite input lands in the large
# branch where inf/sqrt(inf) yields nan (probed to match torch).


@triton.jit
def special_modified_bessel_i1_kernel(
    x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    ax = tl.abs(x)

    y = x / 3.75
    y2 = y * y
    p = 0.00032411
    p = 0.00301532 + y2 * p
    p = 0.02658733 + y2 * p
    p = 0.15084934 + y2 * p
    p = 0.51498869 + y2 * p
    p = 0.87890594 + y2 * p
    p = 0.5 + y2 * p
    ans_small = x * p

    t = 3.75 / tl.maximum(ax, 1e-20)
    q = -0.00420059
    q = 0.01787654 + t * q
    q = -0.02895312 + t * q
    q = 0.02282967 + t * q
    q = -0.01031555 + t * q
    q = 0.00163801 + t * q
    q = -0.00362018 + t * q
    q = -0.03988024 + t * q
    q = 0.39894228 + t * q
    pref = tl.exp(ax) / tl.sqrt(tl.maximum(ax, 1e-20))
    ans_large = tl.where(x < 0.0, -pref * q, pref * q)

    res = tl.where(ax <= 3.75, ans_small, ans_large)

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
    special_modified_bessel_i1_kernel[grid](x_c, out_c, n_elements, BLOCK_SIZE=1024)
    if out_c.data_ptr() != out.data_ptr():
        out.copy_(out_c)
    return out


def special_modified_bessel_i1(x):
    return _run(x)


def special_modified_bessel_i1_out(x, out):
    return _run(x, out=out)
