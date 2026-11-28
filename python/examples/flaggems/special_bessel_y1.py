import torch
import triton
import triton.language as tl

# Bessel Y1, an fp32 port of ATen's bessel_y1_forward (aten Math.h,
# Cephes): the yp/yq polynomial joined to J1 through the 2/pi *
# (J1 log x - 1/x) coupling term below 5, the asymptotic expansion in
# (5/x)^2 above (sin/cos swapped relative to J1).  Boundary contract
# (probed on torch CPU): Y1(0) = -inf, Y1(x < 0) = nan, non-finite input
# is nan.


@triton.jit
def _y1_j1(ax):
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

    return (
        rp
        / rq
        * ax
        * (ax * ax - 1.46819706421238932572e01)
        * (ax * ax - 4.92184563216946036703e01)
    )


@triton.jit
def _bessel_y1_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    yp = 1.26320474790178026440e09
    yp = yp * (x * x) - 6.47355876379160291031e11
    yp = yp * (x * x) + 1.14509511541823727583e14
    yp = yp * (x * x) - 8.12770255501325109621e15
    yp = yp * (x * x) + 2.02439475713594898196e17
    yp = yp * (x * x) - 7.78877196265950026825e17

    yq = 5.94301592346128195359e02
    yq = yq * (x * x) + 2.35564092943068577943e05
    yq = yq * (x * x) + 7.34811944459721705660e07
    yq = yq * (x * x) + 1.87601316108706159478e10
    yq = yq * (x * x) + 3.88231277496238566008e12
    yq = yq * (x * x) + 6.20557727146953693363e14
    yq = yq * (x * x) + 6.87141087355300489866e16
    yq = yq * (x * x) + 3.97270608116560655612e18

    small = x * (yp / yq) + (
        0.636619772367581343075535053490057448 * (_y1_j1(x) * tl.log(x) - 1.0 / x)
    )

    z = 5.0 / x * (5.0 / x)
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

    arg = x - 2.3561944901923449288469825
    large = (pp / pq * tl.sin(arg) + 5.0 / x * (qp / qq) * tl.cos(arg)) * (
        0.797884560802865355879892119868763737 / tl.sqrt(x)
    )

    res = tl.where(x <= 5.0, small, large)
    res = tl.where(x == 0.0, float("-inf"), res)
    res = tl.where(x < 0.0, float("nan"), res)
    res = tl.where(x != x, x, res)

    tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


def special_bessel_y1(A):
    A_c = A.contiguous()
    out = torch.empty_like(A_c)
    n_elements = A_c.numel()
    if n_elements == 0:
        return out
    grid = (triton.cdiv(n_elements, 1024),)
    _bessel_y1_kernel[grid](A_c, out, n_elements, BLOCK_SIZE=1024)
    return out
