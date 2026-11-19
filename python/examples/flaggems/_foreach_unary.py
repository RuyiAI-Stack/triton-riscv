"""``aten::_foreach_*`` unary family: 30 operators x {functional, in-place}.

One kernel with an ``OP`` selector carries the elementwise math; every public
entry is a table row over ``_apply_unary``.  Where this Triton build cannot
lower a libdevice intrinsic, the equivalent closed form is used instead:

* ``round``  -- round-half-to-even by explicit tie detection (``rint``/``nearbyint`` do not lower)
* ``tan``    -- ``sin / cos``
* ``sinh``/``cosh``/``tanh`` -- ``exp`` forms
* ``expm1``  -- 4-term Taylor inside |x| < 0.25, ``exp(x) - 1`` outside
* ``log2``   -- ``log(x) / ln 2``
* ``erfc``   -- ``1 - erf(x)`` below 2.85, 12-term asymptotic series above
  (plain ``1 - erf`` returns 0 for ``x >= 3.5`` in fp32)
* ``lgamma`` -- Lanczos (g = 7, n = 9) with the reflection formula below 0.5
"""

import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as _ld

_BLOCK = 1024

_OPS = {
    "abs": 0,
    "neg": 1,
    "ceil": 2,
    "floor": 3,
    "trunc": 4,
    "round": 5,
    "frac": 6,
    "sign": 7,
    "reciprocal": 8,
    "sqrt": 9,
    "rsqrt": 10,
    "exp": 11,
    "expm1": 12,
    "log": 13,
    "log10": 14,
    "log1p": 15,
    "log2": 16,
    "sin": 17,
    "cos": 18,
    "tan": 19,
    "sinh": 20,
    "cosh": 21,
    "tanh": 22,
    "sigmoid": 23,
    "erf": 24,
    "erfc": 25,
    "acos": 26,
    "asin": 27,
    "atan": 28,
    "lgamma": 29,
}


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
def _unary_kernel(x_ptr, y_ptr, n, OP: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    m = idx < n
    x = tl.load(x_ptr + idx, mask=m, other=0.0).to(tl.float32)
    if OP == 0:
        y = tl.abs(x)
    elif OP == 1:
        y = -x
    elif OP == 2:
        y = tl.math.ceil(x)
    elif OP == 3:
        y = tl.math.floor(x)
    elif OP == 4:
        y = _ld.trunc(x)
    elif OP == 5:
        fl = tl.floor(x)
        f = x - fl
        odd = (fl - 2.0 * tl.floor(fl * 0.5)) != 0.0
        y = tl.where(
            f > 0.5, fl + 1.0, tl.where(f < 0.5, fl, tl.where(odd, fl + 1.0, fl))
        )
    elif OP == 6:
        y = x - _ld.trunc(x)
    elif OP == 7:
        y = tl.where(x > 0.0, 1.0, tl.where(x < 0.0, -1.0, x))
    elif OP == 8:
        y = 1.0 / x
    elif OP == 9:
        y = tl.sqrt(x)
    elif OP == 10:
        y = _ld.rsqrt(x)
    elif OP == 11:
        y = tl.exp(x)
    elif OP == 12:
        y = tl.where(
            tl.abs(x) < 0.25,
            x + x * x * (0.5 + x * (0.16666666666666666 + x * 0.041666666666666664)),
            tl.exp(x) - 1.0,
        )
    elif OP == 13:
        y = tl.log(x)
    elif OP == 14:
        y = _ld.log10(x)
    elif OP == 15:
        y = _ld.log1p(x)
    elif OP == 16:
        y = tl.log(x) * 1.4426950408889634
    elif OP == 17:
        y = tl.sin(x)
    elif OP == 18:
        y = tl.cos(x)
    elif OP == 19:
        y = tl.sin(x) / tl.cos(x)
    elif OP == 20:
        y = (tl.exp(x) - tl.exp(-x)) * 0.5
    elif OP == 21:
        y = (tl.exp(x) + tl.exp(-x)) * 0.5
    elif OP == 22:
        ex = tl.exp(2.0 * x)
        y = 1.0 - 2.0 / (ex + 1.0)
    elif OP == 23:
        y = 1.0 / (1.0 + tl.exp(-x))
    elif OP == 24:
        y = _ld.erf(x)
    elif OP == 25:
        # plain 1 - erf(x) returns exactly 0 once erf rounds to 1 (fp32,
        # x >= ~3.5), so above the measured crossover 2.85 a 12-term
        # asymptotic series is used; the negative side has no cancellation.
        # the seed must not read x's value: x * 0.0 is nan at +inf.
        t = tl.zeros((BLOCK,), tl.float32) + 1.0
        s = t
        for n in range(1, 12):
            t = t * (-(2.0 * n - 1.0) / (2.0 * x * x))
            s = s + t
        asym = tl.exp(-x * x) / (x * 1.7724538509055159) * s
        y = tl.where(x > 2.85, asym, 1.0 - _ld.erf(x))
    elif OP == 26:
        y = _ld.acos(x)
    elif OP == 27:
        y = _ld.asin(x)
    elif OP == 28:
        y = _ld.atan(x)
    else:
        pi = 3.141592653589793
        y = tl.where(
            x < 0.5,
            tl.log(pi / tl.abs(tl.sin(pi * x))) - _lgamma_pos(1.0 - x),
            _lgamma_pos(x),
        )
    tl.store(y_ptr + idx, y.to(y_ptr.dtype.element_ty), mask=m)


def _apply_unary(xs, op, inplace):
    outs = []
    for x in xs:
        if inplace and not x.is_contiguous():
            staging = _apply_unary([x.contiguous()], op, inplace=False)[0]
            x.copy_(staging)
            outs.append(x)
            continue
        xc = x if x.is_contiguous() else x.contiguous()
        dst = x if inplace else torch.empty_like(x)
        dstc = dst if dst.is_contiguous() else torch.empty(x.shape, dtype=x.dtype)
        n = xc.numel()
        if n:
            _unary_kernel[(triton.cdiv(n, _BLOCK),)](
                xc.view(-1), dstc.view(-1), n, OP=op, BLOCK=_BLOCK
            )
        if dstc is not dst:
            dst.copy_(dstc)
        outs.append(dst)
    return None if inplace else outs


def _foreach_abs(self):
    return _apply_unary(self, _OPS["abs"], False)


def _foreach_abs_(self):
    return _apply_unary(self, _OPS["abs"], True)


def _foreach_neg(self):
    return _apply_unary(self, _OPS["neg"], False)


def _foreach_neg_(self):
    return _apply_unary(self, _OPS["neg"], True)


def _foreach_ceil(self):
    return _apply_unary(self, _OPS["ceil"], False)


def _foreach_ceil_(self):
    return _apply_unary(self, _OPS["ceil"], True)


def _foreach_floor(self):
    return _apply_unary(self, _OPS["floor"], False)


def _foreach_floor_(self):
    return _apply_unary(self, _OPS["floor"], True)


def _foreach_trunc(self):
    return _apply_unary(self, _OPS["trunc"], False)


def _foreach_trunc_(self):
    return _apply_unary(self, _OPS["trunc"], True)


def _foreach_round(self):
    return _apply_unary(self, _OPS["round"], False)


def _foreach_round_(self):
    return _apply_unary(self, _OPS["round"], True)


def _foreach_frac(self):
    return _apply_unary(self, _OPS["frac"], False)


def _foreach_frac_(self):
    return _apply_unary(self, _OPS["frac"], True)


def _foreach_sign(self):
    return _apply_unary(self, _OPS["sign"], False)


def _foreach_sign_(self):
    return _apply_unary(self, _OPS["sign"], True)


def _foreach_reciprocal(self):
    return _apply_unary(self, _OPS["reciprocal"], False)


def _foreach_reciprocal_(self):
    return _apply_unary(self, _OPS["reciprocal"], True)


def _foreach_sqrt(self):
    return _apply_unary(self, _OPS["sqrt"], False)


def _foreach_sqrt_(self):
    return _apply_unary(self, _OPS["sqrt"], True)


def _foreach_rsqrt(self):
    return _apply_unary(self, _OPS["rsqrt"], False)


def _foreach_rsqrt_(self):
    return _apply_unary(self, _OPS["rsqrt"], True)


def _foreach_exp(self):
    return _apply_unary(self, _OPS["exp"], False)


def _foreach_exp_(self):
    return _apply_unary(self, _OPS["exp"], True)


def _foreach_expm1(self):
    return _apply_unary(self, _OPS["expm1"], False)


def _foreach_expm1_(self):
    return _apply_unary(self, _OPS["expm1"], True)


def _foreach_log(self):
    return _apply_unary(self, _OPS["log"], False)


def _foreach_log_(self):
    return _apply_unary(self, _OPS["log"], True)


def _foreach_log10(self):
    return _apply_unary(self, _OPS["log10"], False)


def _foreach_log10_(self):
    return _apply_unary(self, _OPS["log10"], True)


def _foreach_log1p(self):
    return _apply_unary(self, _OPS["log1p"], False)


def _foreach_log1p_(self):
    return _apply_unary(self, _OPS["log1p"], True)


def _foreach_log2(self):
    return _apply_unary(self, _OPS["log2"], False)


def _foreach_log2_(self):
    return _apply_unary(self, _OPS["log2"], True)


def _foreach_sin(self):
    return _apply_unary(self, _OPS["sin"], False)


def _foreach_sin_(self):
    return _apply_unary(self, _OPS["sin"], True)


def _foreach_cos(self):
    return _apply_unary(self, _OPS["cos"], False)


def _foreach_cos_(self):
    return _apply_unary(self, _OPS["cos"], True)


def _foreach_tan(self):
    return _apply_unary(self, _OPS["tan"], False)


def _foreach_tan_(self):
    return _apply_unary(self, _OPS["tan"], True)


def _foreach_sinh(self):
    return _apply_unary(self, _OPS["sinh"], False)


def _foreach_sinh_(self):
    return _apply_unary(self, _OPS["sinh"], True)


def _foreach_cosh(self):
    return _apply_unary(self, _OPS["cosh"], False)


def _foreach_cosh_(self):
    return _apply_unary(self, _OPS["cosh"], True)


def _foreach_tanh(self):
    return _apply_unary(self, _OPS["tanh"], False)


def _foreach_tanh_(self):
    return _apply_unary(self, _OPS["tanh"], True)


def _foreach_sigmoid(self):
    return _apply_unary(self, _OPS["sigmoid"], False)


def _foreach_sigmoid_(self):
    return _apply_unary(self, _OPS["sigmoid"], True)


def _foreach_erf(self):
    return _apply_unary(self, _OPS["erf"], False)


def _foreach_erf_(self):
    return _apply_unary(self, _OPS["erf"], True)


def _foreach_erfc(self):
    return _apply_unary(self, _OPS["erfc"], False)


def _foreach_erfc_(self):
    return _apply_unary(self, _OPS["erfc"], True)


def _foreach_acos(self):
    return _apply_unary(self, _OPS["acos"], False)


def _foreach_acos_(self):
    return _apply_unary(self, _OPS["acos"], True)


def _foreach_asin(self):
    return _apply_unary(self, _OPS["asin"], False)


def _foreach_asin_(self):
    return _apply_unary(self, _OPS["asin"], True)


def _foreach_atan(self):
    return _apply_unary(self, _OPS["atan"], False)


def _foreach_atan_(self):
    return _apply_unary(self, _OPS["atan"], True)


def _foreach_lgamma(self):
    return _apply_unary(self, _OPS["lgamma"], False)


def _foreach_lgamma_(self):
    return _apply_unary(self, _OPS["lgamma"], True)
