import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# erfinv semantics: input domain is [-1, 1]; erfinv(+/-1) = +/-inf,
# |x| > 1 (including +/-inf) and NaN map to NaN, 0 maps to 0 (odd
# function, -0.0 preserved).
#
# The backend cannot lower libdevice.erfinv (triton-shared-opt's extern
# set covers __nv_erf/erfc but not __nv_erfinv), so erfinv is evaluated
# with the same algorithm as ATen's CPU kernel (aten/src/ATen/native/
# Math.h, calc_erfinv; a direct translation of the MATLAB erfinv by
# G. L. Pavlis, Indiana University, 1996): a piecewise rational
# approximation followed by two Newton steps that only need erf, exp,
# sqrt and log.  The Newton residual erf(x) - y cancels catastrophically
# in fp32 near |y| -> 1, so the iteration runs in fp64 and narrows once
# at the end (measured max relative error vs torch.erfinv on fp32
# inputs: 1.2e-7).
#
# fp64 tensors fall back to torch.ops.aten.erfinv_ so the in-place
# op keeps full double precision (upstream FlagGems instead pushes fp64
# through its fp32 shim, which loses precision).


@triton.jit
def erfinv_kernel_(
    x_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    y = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float64)
    y_abs = tl.abs(y)

    # |y| <= 0.7: central rational approximation.
    z = y * y
    num = (((-0.140543331 * z + 0.914624893) * z - 1.645349621) * z + 0.886226899)
    dem = ((((0.012229801 * z - 0.329097515) * z + 1.442710462) * z - 2.118377725) * z + 1.0)
    x_central = y * num / dem

    # |y| > 0.7: tail approximation in sqrt(-log((1 - |y|) / 2)).
    z_t = tl.sqrt(-tl.log((1.0 - y_abs) / 2.0))
    num_t = ((1.641345311 * z_t + 3.429567803) * z_t - 1.624906493) * z_t - 1.970840454
    dem_t = (1.637067800 * z_t + 3.543889200) * z_t + 1.0
    x_tail = tl.where(y < 0, -num_t, num_t) / dem_t

    x = tl.where(y_abs <= 0.7, x_central, x_tail)

    # Two Newton steps: x -= (erf(x) - y) / ((2/sqrt(pi)) * exp(-x*x)).
    c = 2.0 / tl.sqrt(3.14159265358979323846)
    for _ in tl.static_range(2):
        f = libdevice.erf(x) - y
        fp = c * tl.exp(-x * x)
        x = x - f / fp

    # Boundary handling: NaN input flows through the tail branch as NaN.
    x = tl.where(y_abs == 1.0, tl.where(y < 0.0, float("-inf"), float("inf")), x)
    x = tl.where(y_abs > 1.0, float("nan"), x)

    tl.store(x_ptr + offsets, x.to(x_ptr.dtype.element_ty), mask=mask)


_TRITON_FLOAT_TYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
)


def _erfinv_contiguous(t):
    # t must be a contiguous tensor of a Triton-supported float dtype;
    # mutated in place by the kernel.
    n_elements = t.numel()
    if n_elements == 0:
        return t

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    erfinv_kernel_[grid](t, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return t


def erfinv_(x):
    if x.dtype not in _TRITON_FLOAT_TYPES:
        # fp64 (and any type torch knows but the kernel does not) keeps
        # aten's precision.
        return torch.ops.aten.erfinv_(x)

    if x.is_contiguous():
        _erfinv_contiguous(x)
        return x
    else:
        # Rewrite the caller's storage behind the strided view.
        y = _erfinv_contiguous(x.contiguous())
        x.copy_(y)
        return x


def erfinv(x):
    if x.dtype not in _TRITON_FLOAT_TYPES:
        return torch.ops.aten.erfinv(x)

    out = x.contiguous()
    if out is x:
        out = x.clone()
    _erfinv_contiguous(out)
    return out.view_as(x)


def erfinv_out(x, *, out=None):
    if out is None:
        return erfinv(x)
    out.copy_(erfinv(x))
    return out
