import torch
import triton
import triton.language as tl

# logsumexp along the innermost row: max-subtract, exp-sum, log -- two
# block sweeps per row so arbitrarily wide rows work.  An all -inf row
# reduces to -inf (torch semantics, probed): the clamped max keeps the
# sum at 0 and log(0) drives the row to -inf instead of nan.  Non-inner
# dims are moved to the innermost axis host-side and restored.


@triton.jit
def logsumexp_kernel_inner(x_ptr, out_ptr, INNER, BLOCK: tl.constexpr):
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
    res = row_max + tl.log(tl.sum(s, axis=0))
    tl.store(out_ptr + row, res)


def special_logsumexp(x, dim, keepdim=False):
    dim = dim if dim >= 0 else dim + x.dim()
    moved = dim != x.dim() - 1
    if moved:
        x = x.movedim(dim, -1)
    x_c = x.contiguous()
    inner = x_c.shape[-1] if x_c.dim() > 0 else 1
    # rows comes from the shape, not numel: a (3, 0) input still has three
    # rows, each reducing to -inf over the empty axis.
    rows = 1
    for d in x_c.shape[:-1]:
        rows *= d
    out = torch.empty((rows,), dtype=x_c.dtype, device=x_c.device)
    if rows > 0:
        grid = (rows,)
        logsumexp_kernel_inner[grid](x_c, out, inner, BLOCK=1024)
    out = out.view(x_c.shape[:-1])
    if keepdim:
        out = out.unsqueeze(-1)
    if moved:
        out = out.movedim(-1, dim)
    return out
