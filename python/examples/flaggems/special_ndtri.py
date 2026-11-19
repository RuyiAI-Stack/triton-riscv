import torch
import triton
import triton.language as tl

# ndtri, an fp32 port of ATen's calc_ndtri (aten Math.h, Cephes): the
# central rational in (y - 1/2)^2 for y in (e^-2, 1 - e^-2), and the tail
# form sqrt(-2 log y) - log(x)/x - P/Q(z) with a second coefficient set
# above x = 8.  Boundary contract (probed on torch CPU): ndtri(0) = -inf,
# ndtri(1) = +inf, y outside [0, 1] = nan, nan stays nan.


@triton.jit
def _polevl(x, c0, c1, c2, c3, c4, c5, c6, c7, c8):
    y = c0
    y = y * x + c1
    y = y * x + c2
    y = y * x + c3
    y = y * x + c4
    y = y * x + c5
    y = y * x + c6
    y = y * x + c7
    y = y * x + c8
    return y


@triton.jit
def _polevl5(x, c0, c1, c2, c3, c4):
    y = c0
    y = y * x + c1
    y = y * x + c2
    y = y * x + c3
    y = y * x + c4
    return y


@triton.jit
def _ndtri_flat_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    y0 = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # ATen's mirror: for y close to 1 evaluate at 1 - y and flip the sign.
    flip = y0 > 1.0 - 0.13533528323661269189
    y = tl.where(flip, 1.0 - y0, y0)

    central = y > 0.13533528323661269189
    yc = tl.where(central, y - 0.5, 0.0)
    y2 = yc * yc
    p0 = _polevl5(
        y2,
        -5.99633501014107895267e1,
        9.80010754185999661536e1,
        -5.66762857469070293439e1,
        1.39312609387279679503e1,
        -1.23916583867381258016e0,
    )
    q0 = _polevl(
        y2,
        1.0,
        1.95448858338141759834e0,
        4.67627912898881538453e0,
        8.63602421390890590575e1,
        -2.25462687854119370527e2,
        2.00260212380060660359e2,
        -8.20372256168333339912e1,
        1.59056225126211695515e1,
        -1.18331621121330003142e0,
    )
    x_central = (yc + yc * (y2 * p0 / q0)) * 2.50662827463100050242

    xt = tl.sqrt(-2.0 * tl.log(tl.where(central, 1e-30, y)))
    x0 = xt - tl.log(xt) / xt
    z = 1.0 / xt
    p1 = _polevl(
        z,
        4.05544892305962419923e0,
        3.15251094599893866154e1,
        5.71628192246421288162e1,
        4.40805073893200834700e1,
        1.46849561928858024014e1,
        2.18663306850790267539e0,
        -1.40256079171354495875e-1,
        -3.50424626827848203418e-2,
        -8.57456785154685413611e-4,
    )
    q1 = _polevl(
        z,
        1.0,
        1.57799883256466749731e1,
        4.53907635128879210584e1,
        4.13172038254672030440e1,
        1.50425385692907503408e1,
        2.50464946208309415979e0,
        -1.42182922854787788574e-1,
        -3.80806407691578277194e-2,
        -9.33259480895457427372e-4,
    )
    p2 = _polevl(
        z,
        3.23774891776946035970e0,
        6.91522889068984211695e0,
        3.93881025292474443415e0,
        1.33303460815807542389e0,
        2.01485389549179081538e-1,
        1.23716634817820021358e-2,
        3.01581553508235416007e-4,
        2.65806974686737550832e-6,
        6.23974539184983293730e-9,
    )
    q2 = _polevl(
        z,
        1.0,
        6.02427039364742014255e0,
        3.67983563856160859403e0,
        1.37702099489081330271e0,
        2.16236993594496635890e-1,
        1.34204006088543189037e-2,
        3.28014464682127739104e-4,
        2.28824067106518537412e-6,
        6.79019408009981274425e-9,
    )
    x1 = tl.where(xt < 8.0, z * p1 / q1, z * p2 / q2)
    x_tail = x0 - x1
    # ATen negates the tail form unless the input was mirrored (y close
    # to 1 evaluated at 1 - y), so the low tail comes out negative.
    x_tail = tl.where(flip, x_tail, -x_tail)

    res = tl.where(central, x_central, x_tail)
    res = tl.where(y0 == 0.0, float("-inf"), res)
    res = tl.where(y0 == 1.0, float("inf"), res)
    res = tl.where((y0 < 0.0) | (y0 > 1.0), float("nan"), res)
    res = tl.where(y0 != y0, y0, res)

    tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


def special_ndtri(A):
    A_c = A.contiguous()
    out = torch.empty_like(A_c)
    n_elements = A_c.numel()
    if n_elements == 0:
        return out
    grid = (triton.cdiv(n_elements, 1024),)
    _ndtri_flat_kernel[grid](A_c, out, n_elements, BLOCK_SIZE=1024)
    return out
