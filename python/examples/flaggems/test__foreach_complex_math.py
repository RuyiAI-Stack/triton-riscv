import importlib

import pytest
import torch
import triton
import triton.language as tl

mod = importlib.import_module("flaggems._foreach_complex_math")


@triton.jit
def _apply(re_ptr, im_ptr, or_ptr, oi_ptr, n, F: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    re = tl.load(re_ptr + offs, mask=m, other=0.0)
    im = tl.load(im_ptr + offs, mask=m, other=0.0)
    if F == 0:
        r, i = mod.c_neg(re, im)
    elif F == 1:
        r, i = mod.c_exp(re, im)
    elif F == 2:
        r, i = mod.c_expm1(re, im)
    elif F == 3:
        r, i = mod.c_log(re, im)
    elif F == 4:
        r, i = mod.c_sqrt(re, im)
    elif F == 5:
        r, i = mod.c_reciprocal(re, im)
    elif F == 6:
        r, i = mod.c_rsqrt(re, im)
    elif F == 7:
        r, i = mod.c_sin(re, im)
    elif F == 8:
        r, i = mod.c_cos(re, im)
    elif F == 9:
        r, i = mod.c_sinh(re, im)
    elif F == 10:
        r, i = mod.c_cosh(re, im)
    elif F == 11:
        r, i = mod.c_tan(re, im)
    elif F == 12:
        r, i = mod.c_tanh(re, im)
    elif F == 13:
        r, i = mod.c_sigmoid(re, im)
    elif F == 14:
        r, i = mod.c_asin(re, im)
    elif F == 15:
        r, i = mod.c_acos(re, im)
    elif F == 16:
        r, i = mod.c_atan(re, im)
    elif F == 17:
        r, i = mod.c_log1p(re, im)
    elif F == 18:
        r, i = mod.c_log2(re, im)
    elif F == 19:
        r, i = mod.c_log10(re, im)
    tl.store(or_ptr + offs, r, mask=m)
    tl.store(oi_ptr + offs, i, mask=m)


def run_formula(f, re, im):
    n = re.numel()
    o_r = torch.empty_like(re)
    o_i = torch.empty_like(im)
    _apply[(triton.cdiv(n, 1024),)](re, im, o_r, o_i, n, F=f, BLOCK=1024)
    return torch.complex(o_r, o_i)


def ref_complex(name, z):
    if name == "neg":
        return -z
    if name == "exp":
        return torch.exp(z)
    if name == "expm1":
        return torch.exp(z) - 1
    if name == "log":
        return torch.log(z)
    if name == "sqrt":
        return torch.sqrt(z)
    if name == "reciprocal":
        return 1 / z
    if name == "rsqrt":
        return 1 / torch.sqrt(z)
    if name == "sin":
        return torch.sin(z)
    if name == "cos":
        return torch.cos(z)
    if name == "sinh":
        return torch.sinh(z)
    if name == "cosh":
        return torch.cosh(z)
    if name == "tan":
        return torch.tan(z)
    if name == "tanh":
        return torch.tanh(z)
    if name == "sigmoid":
        return 1 / (1 + torch.exp(-z))
    if name == "asin":
        return torch.asin(z)
    if name == "acos":
        return torch.acos(z)
    if name == "atan":
        return torch.atan(z)
    if name == "log1p":
        return torch.log1p(z)
    if name == "log2":
        return torch.log2(z)
    if name == "log10":
        return torch.log10(z)


CASES = [
    (0, "neg"), (1, "exp"), (2, "expm1"), (3, "log"), (4, "sqrt"),
    (5, "reciprocal"), (6, "rsqrt"), (7, "sin"), (8, "cos"), (9, "sinh"),
    (10, "cosh"), (11, "tan"), (12, "tanh"), (13, "sigmoid"),
    (14, "asin"), (15, "acos"), (16, "atan"), (17, "log1p"),
    (18, "log2"), (19, "log10"),
]


@pytest.mark.parametrize("fid, name", CASES, ids=[c[1] for c in CASES])
def test_complex_formula_matches_torch(fid, name):
    torch.manual_seed(0)
    re = (torch.rand(512, dtype=torch.float32) - 0.5) * 2.0
    im = (torch.rand(512, dtype=torch.float32) - 0.5) * 2.0
    z = torch.complex(re, im)
    out = run_formula(fid, re, im)
    ref = ref_complex(name, z)
    torch.testing.assert_close(out.real, ref.real, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(out.imag, ref.imag, rtol=1e-4, atol=1e-4)


def test_complex_mul_div_helpers():
    torch.manual_seed(1)
    a = torch.complex(torch.randn(256), torch.randn(256))
    # Keep |b| away from 0: the naive br^2+bi^2 division shares no scaling with
    # torch's Smith division, and a near-zero denominator amplifies the rounding
    # difference without bound.
    b = torch.complex(torch.randn(256) + 2.0, torch.randn(256) + 2.0)
    ref_mul = a * b
    ref_div = a / b

    @triton.jit
    def mul_kernel(ar_p, ai_p, br_p, bi_p, or_p, oi_p, n, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        m = offs < n
        ar = tl.load(ar_p + offs, mask=m, other=0.0)
        ai = tl.load(ai_p + offs, mask=m, other=0.0)
        br = tl.load(br_p + offs, mask=m, other=0.0)
        bi = tl.load(bi_p + offs, mask=m, other=0.0)
        r, i = mod.c_mul(ar, ai, br, bi)
        tl.store(or_p + offs, r, mask=m)
        tl.store(oi_p + offs, i, mask=m)

    @triton.jit
    def div_kernel(ar_p, ai_p, br_p, bi_p, or_p, oi_p, n, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        m = offs < n
        ar = tl.load(ar_p + offs, mask=m, other=0.0)
        ai = tl.load(ai_p + offs, mask=m, other=0.0)
        br = tl.load(br_p + offs, mask=m, other=0.0)
        bi = tl.load(bi_p + offs, mask=m, other=0.0)
        r, i = mod.c_div(ar, ai, br, bi)
        tl.store(or_p + offs, r, mask=m)
        tl.store(oi_p + offs, i, mask=m)

    for kernel, ref in ((mul_kernel, ref_mul), (div_kernel, ref_div)):
        o_r = torch.empty(256)
        o_i = torch.empty(256)
        # .real/.imag of a complex tensor are stride-2 views; the raw-pointer
        # kernels read contiguous memory, so materialize the components first.
        kernel[(1,)](
            a.real.contiguous(), a.imag.contiguous(),
            b.real.contiguous(), b.imag.contiguous(),
            o_r, o_i, 256, BLOCK=256,
        )
        torch.testing.assert_close(o_r, ref.real, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(o_i, ref.imag, rtol=1e-4, atol=1e-4)


def test_formulas_have_triton_kernels():
    for name in ("c_exp", "c_log", "c_sqrt", "c_sin", "c_atan", "c_log10"):
        assert hasattr(mod, name)
