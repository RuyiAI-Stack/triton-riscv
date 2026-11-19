import torch
import triton
import triton.language as tl

# Bessel J0, a direct fp32 port of ATen's bessel_j0_forward (aten Math.h,
# Cephes): the RP/RQ rational form for x <= 5 with the 1 - x^2/4 tiny
# branch, and the asymptotic expansion in 25/x^2 above.  J0 is even;
# non-finite input reaches the asymptotic branch and yields nan through
# cos(inf), matching torch (probed).


@triton.jit
def _j0_flat_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x = tl.abs(x)

    rp = -4.79443220978201773821e09
    rp = rp * (x * x) + 1.95617491946556577543e12
    rp = rp * (x * x) - 2.49248344360967716204e14
    rp = rp * (x * x) + 9.70862251047306323952e15

    rq = 4.99563147152651017219e02
    rq = rq * (x * x) + 1.73785401676374683123e05
    rq = rq * (x * x) + 4.84409658339962045305e07
    rq = rq * (x * x) + 1.11855537045356834862e10
    rq = rq * (x * x) + 2.11277520115489217587e12
    rq = rq * (x * x) + 3.10518229857422583814e14
    rq = rq * (x * x) + 3.18121955943204943306e16
    rq = rq * (x * x) + 1.71086294081043136091e18

    small = (
        (x * x - 5.78318596294678452118e00)
        * (x * x - 3.04712623436620863991e01)
        * (rp / rq)
    )
    tiny = 1.0 - x * x / 4.0

    z = 25.0 / (x * x)
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
    large = (pp / pq * tl.cos(arg) - 5.0 / x * (qp / qq) * tl.sin(arg)) * (
        0.797884560802865355879892119868763737 / tl.sqrt(x)
    )

    res = tl.where(x <= 5.0, tl.where(x < 1.0e-5, tiny, small), large)

    tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


def special_bessel_j0(A):
    A_c = A.contiguous()
    out = torch.empty_like(A_c)
    n_elements = A_c.numel()
    if n_elements == 0:
        return out
    grid = (triton.cdiv(n_elements, 1024),)
    _j0_flat_kernel[grid](A_c, out, n_elements, BLOCK_SIZE=1024)
    return out
