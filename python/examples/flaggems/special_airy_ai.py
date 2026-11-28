import torch
import triton
import triton.language as tl

# Airy Ai, an fp32 port of ATen's airy_ai_forward (aten Math.h): the
# oscillating asymptotic form below -2.09, the decaying form above 2.09
# (returned directly above 8.32), and the power series in x^3 between.
# The power-series while loop becomes a fixed 20-iteration loop (the
# term ratio x^3/(k(k+1)) dies long before that on the band).
# Boundary contract (probed on torch CPU): Ai at either infinity is nan
# on this torch build; nan stays nan.


@triton.jit
def special_airy_ai_func(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # below -2.09: oscillating asymptotic form
    sqm = tl.sqrt(-x)
    z = 1.0 / (-2.0 * x * sqm / 3.0)
    z2 = z * z
    afn = -1.31696323418331795333e-1
    afn = afn * z2 - 6.26456544431912369773e-1
    afn = afn * z2 - 6.93158036036933542233e-1
    afn = afn * z2 - 2.79779981545119124951e-1
    afn = afn * z2 - 4.91900132609500318020e-2
    afn = afn * z2 - 4.06265923594885404393e-3
    afn = afn * z2 - 1.59276496239262096340e-4
    afn = afn * z2 - 2.77649108155232920844e-6
    afn = afn * z2 - 1.67787698489114633780e-8

    afd = 1.33560420706553243746e01
    afd = afd * z2 + 3.26825032795224613948e01
    afd = afd * z2 + 2.67367040941499554804e01
    afd = afd * z2 + 9.18707402907259625840e00
    afd = afd * z2 + 1.47529146771666414581e00
    afd = afd * z2 + 1.15687173795188044134e-1
    afd = afd * z2 + 4.40291641615211203805e-3
    afd = afd * z2 + 7.54720348287414296618e-5
    afd = afd * z2 + 4.51850092970580378464e-7

    agn = 1.97339932091685679179e-2
    agn = agn * z2 + 3.91103029615688277255e-1
    agn = agn * z2 + 1.06579897599595591108e00
    agn = agn * z2 + 9.39169229816650230044e-1
    agn = agn * z2 + 3.51465656105547619242e-1
    agn = agn * z2 + 6.33888919628925490927e-2
    agn = agn * z2 + 5.85804113048388458567e-3
    agn = agn * z2 + 2.82851600836737019778e-4
    agn = agn * z2 + 6.98793669997260967291e-6
    agn = agn * z2 + 8.11789239554389293311e-8
    agn = agn * z2 + 3.41551784765923618484e-10

    agd = 9.30892908077441974853e00
    agd = agd * z2 + 1.98352928718312140417e01
    agd = agd * z2 + 1.55646628932864612953e01
    agd = agd * z2 + 5.47686069422975497931e00
    agd = agd * z2 + 9.54293611618961883998e-1
    agd = agd * z2 + 8.64580826352392193095e-2
    agd = agd * z2 + 4.12656523824222607191e-3
    agd = agd * z2 + 1.01259085116509135510e-4
    agd = agd * z2 + 1.17166733214413521882e-6
    agd = agd * z2 + 4.91834570062930015649e-9

    tneg = -2.0 * x * sqm / 3.0 + 0.78539816339744830962
    neg = (
        5.64189583547756286948e-01
        / tl.sqrt(sqm)
        * (tl.sin(tneg) * (1.0 + z2 * afn / afd) - tl.cos(tneg) * (z * agn / agd))
    )

    # above 2.09: decaying form
    sqp = tl.sqrt(x)
    zeta = 2.0 * x * sqp / 3.0
    r = 1.0 / zeta
    an = 3.46538101525629032477e-1
    an = an * r + 1.20075952739645805542e01
    an = an * r + 7.62796053615234516538e01
    an = an * r + 1.68089224934630576269e02
    an = an * r + 1.59756391350164413639e02
    an = an * r + 7.05360906840444183113e01
    an = an * r + 1.40264691163389668864e01
    an = an * r + 9.99999999999999995305e-01

    ad = 5.67594532638770212846e-1
    ad = ad * r + 1.47562562584847203173e01
    ad = ad * r + 8.45138970141474626562e01
    ad = ad * r + 1.77318088145400459522e02
    ad = ad * r + 1.64234692871529701831e02
    ad = ad * r + 7.14778400825575695274e01
    ad = ad * r + 1.40959135607834029598e01
    ad = ad * r + 1.00000000000000000470e00

    pos = 5.64189583547756286948e-01 * (an / ad) / (2.0 * tl.sqrt(sqp) * tl.exp(zeta))

    # between -2.09 and 2.09: power series f, g in x^3
    z3 = x * x * x
    f = tl.zeros((BLOCK_SIZE,), tl.float32) + 1.0
    g = x
    m = tl.zeros((BLOCK_SIZE,), tl.float32) + 1.0
    n = x
    k = tl.zeros((BLOCK_SIZE,), tl.float32) + 1.0
    for _ in range(0, 20):
        m = m * z3
        k = k + 1.0
        m = m / k
        n = n * z3
        k = k + 1.0
        n = n / k
        m = m / k
        f = f + m
        k = k + 1.0
        n = n / k
        g = g + n
    series = 0.355028053887817239260 * f - 0.258819403792806798405 * g

    res = tl.where(x < -2.09, neg, series)
    res = tl.where(x >= 2.09, pos, res)
    res = tl.where(x > 103.892, 0.0, res)
    # ATen checks isinf: both infinities are nan (even past the cutoff).
    res = tl.where(tl.abs(x) == float("inf"), float("nan"), res)
    res = tl.where(x != x, x, res)

    tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


def _run(A, out=None):
    A_c = A.contiguous()
    if out is None:
        out = torch.empty_like(A_c)
        out_c = out
    else:
        out_c = out if out.is_contiguous() else out.contiguous()
    n_elements = A_c.numel()
    if n_elements == 0:
        if out_c.data_ptr() != out.data_ptr():
            out.copy_(out_c)
        return out
    grid = (triton.cdiv(n_elements, 1024),)
    special_airy_ai_func[grid](A_c, out_c, n_elements, BLOCK_SIZE=1024)
    if out_c.data_ptr() != out.data_ptr():
        out.copy_(out_c)
    return out


def special_airy_ai(A):
    return _run(A)


def special_airy_ai_out(A, *, out=None):
    return _run(A, out=out)
