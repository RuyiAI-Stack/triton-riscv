import torch
import triton
import triton.language as tl

# log_softmax along a row: max-subtract, exp-sum, log -- three block
# sweeps over the row so arbitrarily wide rows work.  A fully -inf row
# yields nan on both sides (torch semantics); the max is clamped to 0
# there only to keep exp(x - m) from smearing nan into the sum.


@triton.jit
def small_n_log_softmax_kernel(x_ptr, out_ptr, INNER, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    base = row * INNER
    offs = tl.arange(0, BLOCK)

    m = tl.full((BLOCK,), float("-inf"), tl.float32)
    for i in range(0, INNER, BLOCK):
        v = tl.load(x_ptr + base + i + offs, mask=i + offs < INNER, other=float("-inf"))
        m = tl.maximum(m, v)
    row_max = tl.max(m, axis=0)
    m_eff = tl.where(row_max == float("-inf"), 0.0, row_max)

    s = tl.zeros((BLOCK,), tl.float32)
    for i in range(0, INNER, BLOCK):
        v = tl.load(x_ptr + base + i + offs, mask=i + offs < INNER, other=float("-inf"))
        s = s + tl.where(i + offs < INNER, tl.exp(v - m_eff), 0.0)
    log_sum = tl.log(tl.sum(s, axis=0))

    # subtract the max and the log-sum separately: folding them into one
    # lse first loses the row's low bits to the big-max cancellation.
    for i in range(0, INNER, BLOCK):
        idx = i + offs
        v = tl.load(x_ptr + base + idx, mask=idx < INNER, other=0.0)
        tl.store(out_ptr + base + idx, (v - m_eff) - log_sum, mask=idx < INNER)


def special_log_softmax(x, dim, dtype=None):
    if dtype is not None:
        x = x.to(dtype)
    dim = dim if dim >= 0 else dim + x.dim()
    if dim != x.dim() - 1:
        x = x.movedim(dim, -1)
    x_c = x.contiguous()
    out = torch.empty_like(x_c)
    inner = x_c.shape[-1] if x_c.dim() > 0 else 1
    if x_c.numel() == 0:
        res = out
    else:
        rows = x_c.numel() // inner
        grid = (rows,)
        small_n_log_softmax_kernel[grid](x_c, out, inner, BLOCK=1024)
        res = out
    if dim != x.dim() - 1:
        res = res.movedim(-1, dim)
    return res
