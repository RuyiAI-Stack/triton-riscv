import torch
import triton
import triton.language as tl

# Exponentially scaled modified Bessel I1 * exp(-|x|), the branch-free
# two-form evaluation from upstream FlagGems (minimax polynomial in
# |x|/4 - 1 below 8, an rsqrt series in v = 1/sqrt(|x|) above; no
# transcendental is evaluated).  i1e is odd; i1e(+inf) = +0, i1e(-inf)
# = -0 (both natural: rsqrt(inf) = 0 kills the tail series), nan stays
# nan (probed on torch CPU).


@triton.jit
def _i1e_flat_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    ax = tl.abs(x)

    z = ax * 0.25 - 1.0
    p = -0.0077510885501093675
    p = p * z + 0.015981503558593128
    p = p * z - 0.0023144844861462963
    p = p * z - 0.0012737304389683303
    p = p * z - 0.03028052576970696
    p = p * z + 0.04672187221986636
    p = p * z - 0.03726202724060249
    p = p * z + 0.03795708373248614
    p = p * z - 0.042995892253954865
    p = p * z + 0.0352493571860574
    p = p * z - 0.018807060387149147
    p = p * z - 0.0018218735995076604
    p = p * z + 0.027731601232007415
    p = p * z - 0.06574323250762948
    p = p * z + 0.17875083950243528

    v = tl.rsqrt(ax)
    v2 = v * v
    q = -0.05752570418996489
    q = q * v2 - 0.04090716742397503
    q = q * v2 - 0.046751048484542895
    q = q * v2 - 0.14960335515053727
    q = q * v2 + 0.3989422804014327
    tail = v * q

    res = tl.where(ax <= 8.0, p, tail)
    res = tl.where(x < 0.0, -res, res)
    # the minimax polynomial only approximates i1e(0) = 0 to ~4e-7; pin
    # the exact zero at the origin (probed: torch returns exact +-0).
    res = tl.where(x == 0.0, 0.0, res)

    tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


def special_i1e(x):
    x_c = x.contiguous()
    out = torch.empty_like(x_c)
    n_elements = x_c.numel()
    if n_elements == 0:
        return out
    grid = (triton.cdiv(n_elements, 1024),)
    _i1e_flat_kernel[grid](x_c, out, n_elements, BLOCK_SIZE=1024)
    return out
