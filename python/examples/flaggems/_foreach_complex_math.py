import triton
import triton.language as tl
import triton.language.extra.libdevice as _ld


@triton.jit
def _hypot_fn(re, im):
    return tl.sqrt(re * re + im * im)


@triton.jit
def _sinh(x):
    return 0.5 * (tl.exp(x) - tl.exp(-x))


@triton.jit
def _cosh(x):
    return 0.5 * (tl.exp(x) + tl.exp(-x))


@triton.jit
def c_neg(re, im):
    return -re, -im


@triton.jit
def c_exp(re, im):
    r = tl.exp(re)
    return r * tl.cos(im), r * tl.sin(im)


@triton.jit
def c_expm1(re, im):
    r = tl.exp(re)
    return r * tl.cos(im) - 1.0, r * tl.sin(im)


@triton.jit
def c_log(re, im):
    return tl.log(_hypot_fn(re, im)), _ld.atan2(im, re)


@triton.jit
def c_log1p(re, im):
    return c_log(re + 1.0, im)


@triton.jit
def c_log2(re, im):
    lr, li = c_log(re, im)
    inv = 1.4426950408889634
    return lr * inv, li * inv


@triton.jit
def c_log10(re, im):
    lr, li = c_log(re, im)
    inv = 0.4342944819032518
    return lr * inv, li * inv


@triton.jit
def c_sqrt(re, im):
    m = _hypot_fn(re, im)
    a = tl.sqrt(0.5 * (m + re))
    b = tl.sqrt(0.5 * (m - re))
    return a, tl.where(im < 0, -b, b)


@triton.jit
def c_mul(ar, ai, br, bi):
    return ar * br - ai * bi, ar * bi + ai * br


@triton.jit
def c_div(ar, ai, br, bi):
    d = br * br + bi * bi
    return (ar * br + ai * bi) / d, (ai * br - ar * bi) / d


@triton.jit
def c_reciprocal(re, im):
    return c_div(1.0, 0.0, re, im)


@triton.jit
def c_rsqrt(re, im):
    sr, si = c_sqrt(re, im)
    return c_reciprocal(sr, si)


@triton.jit
def c_sin(re, im):
    return tl.sin(re) * _cosh(im), tl.cos(re) * _sinh(im)


@triton.jit
def c_cos(re, im):
    return tl.cos(re) * _cosh(im), -tl.sin(re) * _sinh(im)


@triton.jit
def c_sinh(re, im):
    return _sinh(re) * tl.cos(im), _cosh(re) * tl.sin(im)


@triton.jit
def c_cosh(re, im):
    return _cosh(re) * tl.cos(im), _sinh(re) * tl.sin(im)


@triton.jit
def c_tan(re, im):
    sr, si = c_sin(re, im)
    cr, ci = c_cos(re, im)
    return c_div(sr, si, cr, ci)


@triton.jit
def c_tanh(re, im):
    sr, si = c_sinh(re, im)
    cr, ci = c_cosh(re, im)
    return c_div(sr, si, cr, ci)


@triton.jit
def c_sigmoid(re, im):
    er, ei = c_exp(-re, -im)
    return c_div(1.0, 0.0, 1.0 + er, ei)


@triton.jit
def c_asin(re, im):
    zr, zi = c_mul(re, im, re, im)
    sr, si = c_sqrt(1.0 - zr, -zi)
    lr, li = c_log(-im + sr, re + si)
    return li, -lr


@triton.jit
def c_acos(re, im):
    ar, ai = c_asin(re, im)
    return 1.5707963267948966 - ar, -ai


@triton.jit
def c_atan(re, im):
    qr, qi = c_div(re, 1.0 + im, -re, 1.0 - im)
    lr, li = c_log(qr, qi)
    return -0.5 * li, 0.5 * lr
