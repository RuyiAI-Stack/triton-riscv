import torch
import triton
import triton.language as tl

# Bessel Y0, an fp32 port of ATen's bessel_y0_forward (aten Math.h,
# Cephes): the J0 rational form and the yp/yq polynomial joined by the
# 2/pi * log(x) * J0 coupling term below 5, the asymptotic expansion in
# 25/x^2 above.  Boundary contract (probed on torch CPU): Y0(0) = -inf,
# Y0(x < 0) = nan, non-finite input is nan.


@triton.jit
def _j0_small(x, x2):
    rp = -4.79443220978201773821e09
    rp = rp * x2 + 1.95617491946556577543e12
    rp = rp * x2 - 2.49248344360967716204e14
    rp = rp * x2 + 9.70862251047306323952e15

    rq = 4.99563147152651017219e02
    rq = rq * x2 + 1.73785401676374683123e05
    rq = rq * x2 + 4.84409658339962045305e07
    rq = rq * x2 + 1.11855537045356834862e10
    rq = rq * x2 + 2.11277520115489217587e12
    rq = rq * x2 + 3.10518229857422583814e14
    rq = rq * x2 + 3.18121955943204943306e16
    rq = rq * x2 + 1.71086294081043136091e18

    small = (
        (x2 - 5.78318596294678452118e00) * (x2 - 3.04712623436620863991e01) * (rp / rq)
    )
    tiny = 1.0 - 0.25 * x2
    return tl.where(x < 1.0e-5, tiny, small)


@triton.jit
def _y0_small(x, x2):
    yp = 1.55924367855235737965e04
    yp = yp * x2 - 1.46639295903971606143e07
    yp = yp * x2 + 5.43526477051876500413e09
    yp = yp * x2 - 9.82136065717911466409e11
    yp = yp * x2 + 8.75906394395366999549e13
    yp = yp * x2 - 3.46628303384729719441e15
    yp = yp * x2 + 4.42733268572569800351e16
    yp = yp * x2 - 1.84950800436986690637e16

    yq = 1.04128353664259848412e03
    yq = yq * x2 + 6.26107330137134956842e05
    yq = yq * x2 + 2.68919633393814121987e08
    yq = yq * x2 + 8.64002487103935000337e10
    yq = yq * x2 + 2.02979612750105546709e13
    yq = yq * x2 + 3.17157752842975028269e15
    yq = yq * x2 + 2.50596256172653059228e17

    return yp / yq + 0.636619772367581343075535053490057448 * tl.log(x) * _j0_small(
        x, x2
    )


@triton.jit
def _y0_large(x, x2):
    z = 25.0 / x2
    pp = 7.96936729297347051624e-04
    pp = pp * z + 8.28352392107440799803e-02
    pp = pp * z + 1.23953371646414299388e00
    pp = pp * z + 5.44725003058768775090e00
    pp = pp * z + 8.74716500199817011941e00
    pp = pp * z + 5.30324038235394892183e00
    pp = pp * z + 9.99999999999999997821e-01

    pq = 9.24408810558863637013e-04
    pq = pq * z + 8.56288474354474431428e-02
    pq = pq * z + 1.25352743901058953537e00
    pq = pq * z + 5.47097740330417105182e00
    pq = pq * z + 8.76190883237069594232e00
    pq = pq * z + 5.30605288235394617618e00
    pq = pq * z + 1.00000000000000000218e00

    qp = -1.13663838898469149931e-02
    qp = qp * z - 1.28252718670509318512e00
    qp = qp * z - 1.95539544257735972385e01
    qp = qp * z - 9.32060152123768231369e01
    qp = qp * z - 1.77681167980488050595e02
    qp = qp * z - 1.47077505154951170175e02
    qp = qp * z - 5.14105326766599330220e01
    qp = qp * z - 6.05014350600728481186e00

    qq = 6.43178256118178023184e01
    qq = qq * z + 8.56430025976980587198e02
    qq = qq * z + 3.88240183605401609683e03
    qq = qq * z + 7.24046774195652478189e03
    qq = qq * z + 5.93072701187316984827e03
    qq = qq * z + 2.06209331660327847417e03
    qq = qq * z + 2.42005740240291393179e02

    arg = x - 0.7853981633974483096156608
    return (pp / pq * tl.sin(arg) + 5.0 / x * (qp / qq) * tl.cos(arg)) * (
        0.797884560802865355879892119868763737 / tl.sqrt(x)
    )


@triton.jit
def _bessel_y0_kernel_contig(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    x2 = x * x
    y = tl.where(x <= 5.0, _y0_small(x, x2), _y0_large(x, x2))
    y = tl.where(x == 0.0, float("-inf"), y)
    y = tl.where(x < 0.0, float("nan"), y)
    y = tl.where(x != x, x, y)

    tl.store(out_ptr + offsets, y.to(out_ptr.dtype.element_ty), mask=mask)


def special_bessel_y0(A):
    A_c = A.contiguous()
    out = torch.empty_like(A_c)
    n_elements = A_c.numel()
    if n_elements == 0:
        return out
    grid = (triton.cdiv(n_elements, 1024),)
    _bessel_y0_kernel_contig[grid](A_c, out, n_elements, BLOCK_SIZE=1024)
    return out
