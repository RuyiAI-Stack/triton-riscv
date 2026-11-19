import math

import torch
import triton
import triton.language as tl

_F32_MAX = torch.finfo(torch.float32).max
_F32_MIN = torch.finfo(torch.float32).tiny
_INT32_MAX = 2147483647.0


@triton.jit
def _float_power_core(x, e):
    # C pow contract on a fp32 domain: e == 0 -> 1 (even for NaN inputs),
    # x == 1 -> 1, zero base saturates, negative base takes integer
    # exponents with an odd/even sign and NaN for fractional ones.
    e = e.to(tl.float32)
    x = x.to(tl.float32)

    # clamp before the float->int cast: fptosi of 2**31 is poison, and any
    # exponent beyond int32 saturates the result anyway
    e_sat = tl.minimum(tl.maximum(e, -2147483647.0), 2147483647.0)
    e_trunc = e_sat.to(tl.int32)
    is_int = e_trunc.to(tl.float32) == e
    odd = (e_trunc & 1) == 1
    signbit_x = x.to(tl.int32, bitcast=True) < 0

    # integer exponents: binary powering of |x| ** |e| (exact while the
    # products stay representable), reciprocal in fp64 so negative
    # exponents keep full double precision
    base = tl.abs(x)
    n = tl.abs(e_trunc)
    acc = tl.zeros_like(base) + 1.0
    for _ in range(31):
        acc = tl.where((n & 1) != 0, acc * base, acc)
        base = base * base
        n = n >> 1
    p64 = acc.to(tl.float64)
    sign = tl.where(odd & signbit_x, -1.0, 1.0)
    int_res = sign * tl.where(e >= 0.0, p64, 1.0 / p64)

    # fractional exponents: exp(e * log(|x|)) in fp32 (~1e-6 relative)
    frac_res = tl.exp(e * tl.log(tl.abs(x)))
    zero_frac = tl.where(e > 0.0, 0.0, float("inf"))

    return tl.where(
        e == 0.0,
        1.0,
        tl.where(
            x == 1.0,
            1.0,
            tl.where(
                x == 0.0,
                tl.where(is_int, int_res, zero_frac),
                tl.where(
                    x < 0.0,
                    tl.where(is_int, int_res, float("nan")),
                    tl.where(is_int, int_res, frac_res),
                ),
            ),
        ),
    ).to(tl.float64)


@triton.jit
def float_power_kernel_tt(
    x_ptr,
    e_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=1.0)
    e = tl.load(e_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, _float_power_core(x, e), mask=mask)


@triton.jit
def float_power_kernel_ts(
    x_ptr,
    exponent,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=1.0)
    tl.store(out_ptr + offsets, _float_power_core(x, exponent), mask=mask)


@triton.jit
def float_power_kernel_st(
    x_val,
    e_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    e = tl.load(e_ptr + offsets, mask=mask, other=0.0)
    # broadcast the scalar base to the block so _float_power_core can
    # bitcast it for the sign test
    x = e * 0.0 + x_val
    tl.store(out_ptr + offsets, _float_power_core(x, e), mask=mask)


def _check_base_fp64(A):
    if A.dtype != torch.float64:
        raise RuntimeError(
            f"the base given to float_power_ has dtype {A.dtype} "
            "but the operation's result requires dtype Double"
        )


def _tensor_domain_ok(t):
    if t.numel() == 0:
        return True
    if not bool(torch.isfinite(t).all()):
        return False
    a = t.abs()
    return bool(((a == 0) | ((a >= _F32_MIN) & (a <= _F32_MAX))).all())


def _exponent_domain_ok(t):
    if t.numel() == 0:
        return True
    if not bool(torch.isfinite(t).all()):
        return False
    return bool((t.abs() <= _INT32_MAX).all())


def _scalar_base_ok(value):
    f = float(value)
    if not math.isfinite(f):
        return False
    a = abs(f)
    return a == 0.0 or (_F32_MIN <= a <= _F32_MAX)


def _scalar_exponent_ok(value):
    f = float(value)
    return math.isfinite(f) and abs(f) <= _INT32_MAX


def _launch_tt(A, exponent, out):
    n_elements = out.numel()
    grid = (triton.cdiv(n_elements, 1024),)
    float_power_kernel_tt[grid](
        A.contiguous(), exponent.contiguous(), out, n_elements, BLOCK_SIZE=1024
    )
    return out


def float_power_tensor_tensor_(A, exponent):
    _check_base_fp64(A)
    if not isinstance(exponent, torch.Tensor):
        return float_power_tensor_scalar_(A, exponent)
    exponent = torch.broadcast_to(exponent, A.shape)
    if A.numel() == 0:
        return A
    if not (_tensor_domain_ok(A) and _exponent_domain_ok(exponent)):
        return A.float_power_(exponent)
    out = torch.empty(A.shape, dtype=torch.float64)
    _launch_tt(A, exponent, out)
    A.copy_(out)
    return A


def float_power_tensor_scalar_(A, exponent):
    _check_base_fp64(A)
    if A.numel() == 0:
        return A
    if not (_tensor_domain_ok(A) and _scalar_exponent_ok(exponent)):
        return A.float_power_(exponent)
    n_elements = A.numel()
    out = torch.empty(A.shape, dtype=torch.float64)
    grid = (triton.cdiv(n_elements, 1024),)
    float_power_kernel_ts[grid](
        A.contiguous(), float(exponent), out, n_elements, BLOCK_SIZE=1024
    )
    A.copy_(out)
    return A


def _prepare_out(out, shape, device):
    if out.dtype != torch.float64:
        raise RuntimeError(
            f"the output given to float_power has dtype {out.dtype} "
            "but the operation's result requires dtype Double"
        )
    if out.device != device:
        raise RuntimeError(
            f"Expected out tensor to have device {device}, but got {out.device} instead"
        )
    if out.shape != shape:
        out.resize_(shape)
    return out


def float_power_tensor_tensor(A, exponent):
    if isinstance(exponent, torch.Tensor):
        A, exponent = torch.broadcast_tensors(A, exponent)
        if not (_tensor_domain_ok(A) and _exponent_domain_ok(exponent)):
            return torch.float_power(A, exponent)
        out = torch.empty(A.shape, dtype=torch.float64, device=A.device)
        return _launch_tt(A, exponent, out)
    return float_power_tensor_scalar(A, exponent)


def float_power_tensor_scalar(A, exponent):
    if not (_tensor_domain_ok(A) and _scalar_exponent_ok(exponent)):
        return torch.float_power(A, exponent)
    n_elements = A.numel()
    out = torch.empty(A.shape, dtype=torch.float64, device=A.device)
    if n_elements == 0:
        return out
    grid = (triton.cdiv(n_elements, 1024),)
    float_power_kernel_ts[grid](
        A.contiguous(), float(exponent), out, n_elements, BLOCK_SIZE=1024
    )
    return out


def float_power_scalar_tensor(A, exponent):
    if not (_scalar_base_ok(A) and _exponent_domain_ok(exponent)):
        return torch.float_power(A, exponent)
    n_elements = exponent.numel()
    out = torch.empty(exponent.shape, dtype=torch.float64, device=exponent.device)
    if n_elements == 0:
        return out
    grid = (triton.cdiv(n_elements, 1024),)
    float_power_kernel_st[grid](
        float(A), exponent.contiguous(), out, n_elements, BLOCK_SIZE=1024
    )
    return out


def float_power_tensor_tensor_out(A, exponent, *, out):
    result = float_power_tensor_tensor(A, exponent)
    _prepare_out(out, result.shape, A.device)
    out.copy_(result)
    return out


def float_power_tensor_scalar_out(A, exponent, *, out):
    result = float_power_tensor_scalar(A, exponent)
    _prepare_out(out, result.shape, A.device)
    out.copy_(result)
    return out


def float_power_scalar_tensor_out(A, exponent, *, out):
    result = float_power_scalar_tensor(A, exponent)
    _prepare_out(out, result.shape, exponent.device)
    out.copy_(result)
    return out
