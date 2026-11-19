import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# Hurwitz zeta, an fp32 port of ATen's Cephes zeta: a 9-term power sum
# (the ATen while loop always stops at i = 9) with an early-convergence
# mask standing in for its early return, followed by the Euler-Maclaurin
# correction with the 12 Bernoulli-number coefficients.  Boundary
# contract (probed on torch CPU): zeta(1, q) = +inf, x < 1 is nan,
# zeta(inf, 1) = 1, non-positive integer q is +inf, nan stays nan.


@triton.jit
def special_zeta_flat_kernel(
    x_ptr, q_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    q = tl.load(q_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    s = libdevice.pow(q, -x)
    a = q
    b = tl.zeros((BLOCK_SIZE,), tl.float32)
    done = tl.zeros((BLOCK_SIZE,), tl.int1)
    for i in range(1, 10):
        active = ~done
        a2 = a + 1.0
        b2 = libdevice.pow(a2, -x)
        s = s + tl.where(active, b2, 0.0)
        a = tl.where(active, a2, a)
        b = tl.where(active, b2, b)
        done = done | (active & (tl.abs(b2) < 5.96e-08 * tl.abs(s)))

    # Euler-Maclaurin tail (only meaningful when the sum never converged).
    w = a
    s_em = s + b * w / (x - 1.0) - 0.5 * b
    aa = tl.zeros((BLOCK_SIZE,), tl.float32) + 1.0
    bb = b
    kk = tl.zeros((BLOCK_SIZE,), tl.float32)
    A0 = 12.0
    A1 = -720.0
    A2 = 30240.0
    A3 = -1209600.0
    A4 = 47900160.0
    A5 = -1.8924375803183791606e9
    A6 = 7.47242496e10
    A7 = -2.950130727918164224e12
    A8 = 1.1646782814350067249e14
    A9 = -4.5979787224074726105e15
    A10 = 1.8152105401943546773e17
    A11 = -7.1661652561756670113e18
    for i in range(0, 12):
        if i == 0:
            c = A0
        elif i == 1:
            c = A1
        elif i == 2:
            c = A2
        elif i == 3:
            c = A3
        elif i == 4:
            c = A4
        elif i == 5:
            c = A5
        elif i == 6:
            c = A6
        elif i == 7:
            c = A7
        elif i == 8:
            c = A8
        elif i == 9:
            c = A9
        elif i == 10:
            c = A10
        else:
            c = A11
        aa = aa * (x + kk)
        bb = bb / w
        t = aa * bb / c
        s_em = s_em + t
        kk = kk + 1.0
        aa = aa * (x + kk)
        bb = bb / w
        kk = kk + 1.0

    res = tl.where(done, s, s_em)
    res = tl.where(x == 1.0, float("inf"), res)
    res = tl.where(x < 1.0, float("nan"), res)
    res = tl.where((q <= 0.0) & (q == tl.floor(q)), float("inf"), res)
    res = tl.where(
        (q <= 0.0) & (q != tl.floor(q)) & (x != tl.floor(x)), float("nan"), res
    )
    res = tl.where(x != x, x, res)
    res = tl.where(q != q, q, res)

    tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


def _run(x, q, out=None):
    if q.device != x.device:
        q = q.to(x.device)
    x, q = torch.broadcast_tensors(x, q)
    x_c = x.contiguous()
    q_c = q.contiguous()
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
    special_zeta_flat_kernel[grid](x_c, q_c, out_c, n_elements, BLOCK_SIZE=1024)
    if out_c.data_ptr() != out.data_ptr():
        out.copy_(out_c)
    return out


def _tensor_scalar(x, q, out=None):
    if not isinstance(q, torch.Tensor):
        q = torch.full((), q, dtype=x.dtype, device=x.device)
    return _run(x, q, out=out)


def _scalar_tensor(x, q, out=None):
    if not isinstance(x, torch.Tensor):
        x = torch.full((), x, dtype=q.dtype, device=q.device)
    return _run(x, q, out=out)


def special_zeta(x, q):
    return _run(x, q)


def special_zeta_out(x, q, out):
    return _run(x, q, out=out)


def special_zeta_tensor_scalar(x, q):
    return _tensor_scalar(x, q)


def special_zeta_tensor_scalar_out(x, q, out):
    return _tensor_scalar(x, q, out=out)


def special_zeta_scalar_tensor(x, q):
    return _scalar_tensor(x, q)


def special_zeta_scalar_tensor_out(x, q, out):
    return _scalar_tensor(x, q, out=out)
