import torch
import triton
import triton.language as tl

# Bessel J1, a direct fp32 port of ATen's bessel_j1_forward (aten Math.h,
# Cephes): the RP/RQ rational form for |x| <= 5 and the asymptotic
# expansion in (5/x)^2 above.  J1 is odd, so the core runs on |x| and the
# sign is restored at the end; non-finite input gives nan through the
# asymptotic trig, matching torch (probed).


@triton.jit
def _j1_core(ax):
    rp = -8.99971225705559398224e08
    rp = rp * (ax * ax) + 4.52228297998194034323e11
    rp = rp * (ax * ax) - 7.27494245221818276015e13
    rp = rp * (ax * ax) + 3.68295732863852883286e15

    rq = 6.20836478118054335476e02
    rq = rq * (ax * ax) + 2.56987256757748830383e05
    rq = rq * (ax * ax) + 8.35146791431949253037e07
    rq = rq * (ax * ax) + 2.21511595479792499675e10
    rq = rq * (ax * ax) + 4.74914122079991414898e12
    rq = rq * (ax * ax) + 7.84369607876235854894e14
    rq = rq * (ax * ax) + 8.95222336184627338078e16
    rq = rq * (ax * ax) + 5.32278620332680085395e18

    small = (
        rp
        / rq
        * ax
        * (ax * ax - 1.46819706421238932572e01)
        * (ax * ax - 4.92184563216946036703e01)
    )

    z = 5.0 / ax * (5.0 / ax)
    pp = 7.62125616208173112003e-04
    pp = pp * z + 7.31397056940917570436e-02
    pp = pp * z + 1.12719608129684925192e00
    pp = pp * z + 5.11207951146807644818e00
    pp = pp * z + 8.42404590141772420927e00
    pp = pp * z + 5.21451598682361504063e00
    pp = pp * z + 1.00000000000000000254e00

    pq = 5.71323128072548699714e-04
    pq = pq * z + 6.88455908754495404082e-02
    pq = pq * z + 1.10514232634061696926e00
    pq = pq * z + 5.07386386128601488557e00
    pq = pq * z + 8.39985554327604159757e00
    pq = pq * z + 5.20982848682361821619e00
    pq = pq * z + 9.99999999999999997461e-01

    qp = 5.10862594750176621635e-02
    qp = qp * z + 4.98213872951233449420e00
    qp = qp * z + 7.58238284132545283818e01
    qp = qp * z + 3.66779609360150777800e02
    qp = qp * z + 7.10856304998926107277e02
    qp = qp * z + 5.97489612400613639965e02
    qp = qp * z + 2.11688757100572135698e02
    qp = qp * z + 2.52070205858023719784e01

    qq = 7.42373277035675149943e01
    qq = qq * z + 1.05644886038262816351e03
    qq = qq * z + 4.98641058337653607651e03
    qq = qq * z + 9.56231892404756170795e03
    qq = qq * z + 7.99704160447350683650e03
    qq = qq * z + 2.82619278517639096600e03
    qq = qq * z + 3.36093607810698293419e02

    arg = ax - 2.3561944901923449288469825
    large = (pp / pq * tl.cos(arg) - 5.0 / ax * (qp / qq) * tl.sin(arg)) * (
        0.797884560802865355879892119868763737 / tl.sqrt(ax)
    )

    return tl.where(ax <= 5.0, small, large)


@triton.jit
def j1_func(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    v = _j1_core(tl.abs(x))
    res = tl.where(x < 0.0, -v, v)

    tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


def special_bessel_j1(A):
    A_c = A.contiguous()
    out = torch.empty_like(A_c)
    n_elements = A_c.numel()
    if n_elements == 0:
        return out
    grid = (triton.cdiv(n_elements, 1024),)
    j1_func[grid](A_c, out, n_elements, BLOCK_SIZE=1024)
    return out
