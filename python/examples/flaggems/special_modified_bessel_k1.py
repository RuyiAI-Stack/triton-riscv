import torch
import triton
import triton.language as tl

# Modified Bessel K1, an fp32 port of ATen's modified_bessel_k1_forward
# (aten Math.h): a Clenshaw recurrence over the A coefficients joined to
# I1 through log(x/2) below 2, and the exp(-x)/sqrt(x) Clenshaw over the
# B coefficients above.  Boundary contract (probed on torch CPU):
# K1(0) = +inf, K1(x < 0) = nan, K1(+inf) = 0.


@triton.jit
def _k1_i1(ax):
    # I1(ax) for ax >= 0, the Cephes small/large split (shared with the
    # modified_bessel_i1 port).
    y = ax / 3.75
    y2 = y * y
    p = 0.00032411
    p = 0.00301532 + y2 * p
    p = 0.02658733 + y2 * p
    p = 0.15084934 + y2 * p
    p = 0.51498869 + y2 * p
    p = 0.87890594 + y2 * p
    p = 0.5 + y2 * p
    ans_small = ax * p

    t = 3.75 / tl.maximum(ax, 1e-20)
    q = -0.00420059
    q = 0.01787654 + t * q
    q = -0.02895312 + t * q
    q = 0.02282967 + t * q
    q = -0.01031555 + t * q
    q = 0.00163801 + t * q
    q = -0.00362018 + t * q
    q = -0.03988024 + t * q
    q = 0.39894228 + t * q
    ans_large = tl.exp(ax) / tl.sqrt(tl.maximum(ax, 1e-20)) * q

    return tl.where(ax <= 3.75, ans_small, ans_large)


@triton.jit
def special_modified_bessel_k1_func(
    x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    a = -7.02386347938628759343e-18
    p = tl.zeros((BLOCK_SIZE,), tl.float32)
    q = tl.zeros((BLOCK_SIZE,), tl.float32)
    a = (x * x - 2.0) * q - p + (-2.42744985051936593393e-15)
    p = q
    q = a
    a = (x * x - 2.0) * q - p + (-6.66690169419932900609e-13)
    p = q
    q = a
    a = (x * x - 2.0) * q - p + (-1.41148839263352776110e-10)
    p = q
    q = a
    a = (x * x - 2.0) * q - p + (-2.21338763073472585583e-08)
    p = q
    q = a
    a = (x * x - 2.0) * q - p + (-2.43340614156596823496e-06)
    p = q
    q = a
    a = (x * x - 2.0) * q - p + (-1.73028895751305206302e-04)
    p = q
    q = a
    a = (x * x - 2.0) * q - p + (-6.97572385963986435018e-03)
    p = q
    q = a
    a = (x * x - 2.0) * q - p + (-1.22611180822657148235e-01)
    p = q
    q = a
    a = (x * x - 2.0) * q - p + (-3.53155960776544875667e-01)
    p = q
    q = a
    a = (x * x - 2.0) * q - p + 1.52530022733894777053e00
    small = tl.log(0.5 * x) * _k1_i1(x) + 0.5 * (a - p) / x

    # The Clenshaw seed must be a tensor: a scalar b would change the
    # loop-carried type between iterations.
    b = tl.zeros((BLOCK_SIZE,), tl.float32) + (-5.75674448366501715755e-18)
    p = tl.zeros((BLOCK_SIZE,), tl.float32)
    q = tl.zeros((BLOCK_SIZE,), tl.float32)
    for i in range(1, 25):
        if i == 1:
            c = 1.79405087314755922667e-17
        elif i == 2:
            c = -5.68946255844285935196e-17
        elif i == 3:
            c = 1.83809354436663880070e-16
        elif i == 4:
            c = -6.05704724837331885336e-16
        elif i == 5:
            c = 2.03870316562433424052e-15
        elif i == 6:
            c = -7.01983709041831346144e-15
        elif i == 7:
            c = 2.47715442448130437068e-14
        elif i == 8:
            c = -8.97670518232499435011e-14
        elif i == 9:
            c = 3.34841966607842919884e-13
        elif i == 10:
            c = -1.28917396095102890680e-12
        elif i == 11:
            c = 5.13963967348173025100e-12
        elif i == 12:
            c = -2.12996783842756842877e-11
        elif i == 13:
            c = 9.21831518760500529508e-11
        elif i == 14:
            c = -4.19035475934189648750e-10
        elif i == 15:
            c = 2.01504975519703286596e-09
        elif i == 16:
            c = -1.03457624656780970260e-08
        elif i == 17:
            c = 5.74108412545004946722e-08
        elif i == 18:
            c = -3.50196060308781257119e-07
        elif i == 19:
            c = 2.40648494783721712015e-06
        elif i == 20:
            c = -1.93619797416608296024e-05
        elif i == 21:
            c = 1.95215518471351631108e-04
        elif i == 22:
            c = -2.85781685962277938680e-03
        elif i == 23:
            c = 1.03923736576817238437e-01
        else:
            c = 2.72062619048444266945e00
        p = q
        q = b
        b = (8.0 / x - 2.0) * q - p + c
    large = tl.exp(-x) * (0.5 * (b - p)) / tl.sqrt(x)

    res = tl.where(x <= 2.0, small, large)
    res = tl.where(x == 0.0, float("inf"), res)
    res = tl.where(x < 0.0, float("nan"), res)
    res = tl.where(x != x, x, res)

    tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


def _run(x, out=None):
    x_c = x.contiguous()
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
    special_modified_bessel_k1_func[grid](x_c, out_c, n_elements, BLOCK_SIZE=1024)
    if out_c.data_ptr() != out.data_ptr():
        out.copy_(out_c)
    return out


def special_modified_bessel_k1(A):
    return _run(A)


def special_modified_bessel_k1_out(A, out):
    return _run(A, out=out)
