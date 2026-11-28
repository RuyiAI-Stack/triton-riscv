import torch
import triton
import triton.language as tl

# multilabel_margin_loss_backward: gradient of the pinned ranking formula
#   loss = sum_j sum_{d not in I} relu(1 - x[t[j]] + x[d]) / C.
# For an output element k:
#   k in I  : -sum over sequence entries j with t[j] == k of
#             #{d not in I : 1 - x[k] + x[d] > 0}
#   k not in I: +#{(j, d=k) : 1 - x[t[j]] + x[k] > 0}
# all divided by C, scaled by the upstream gradient (row-wise for
# reduction=none, 1/N for mean, plain for sum). The strict marg > 0 matches
# the relu subgradient aten uses. reduction must be an integer (0, 1 or 2).


@triton.jit
def _multilabel_margin_loss_backward_kernel(
    x_ptr,
    t_ptr,
    ist_ptr,
    g_ptr,
    out_ptr,
    n_lanes,
    c,
    scale,
    BLOCK: tl.constexpr,
):
    lane = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    lmask = lane < n_lanes
    row = lane // c
    k = lane - row * c
    row_base = row * c

    stop = tl.zeros([BLOCK], dtype=tl.int32) + c
    for jj in range(c):
        tj = tl.load(t_ptr + row_base + jj, mask=lmask, other=-1).to(tl.int32)
        stop = tl.minimum(stop, tl.where(tj < 0, jj, c))
    member = tl.load(ist_ptr + lane, mask=lmask, other=0.0) != 0.0

    xk = tl.load(x_ptr + lane, mask=lmask, other=0.0).to(tl.float32)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    if True:
        # k as a non-target class: +1 per active (j, k) pair.
        for jj in range(c):
            tj = tl.load(t_ptr + row_base + jj, mask=lmask, other=-1).to(tl.int32)
            active = lmask & (~member) & (jj < stop)
            xj = tl.load(x_ptr + row_base + tj, mask=active, other=0.0).to(tl.float32)
            marg_k = 1.0 - xj + xk
            acc += tl.where(active & (marg_k > 0.0), 1.0, 0.0)
        # k as a target label: -1 per active (k as t[j], d) pair.
        for jj in range(c):
            tj = tl.load(t_ptr + row_base + jj, mask=lmask, other=-1).to(tl.int32)
            is_entry = lmask & (jj < stop) & (tj == k.to(tl.int32))
            xj = tl.load(x_ptr + row_base + tj, mask=is_entry, other=0.0).to(tl.float32)
            for dd in range(c):
                xd = tl.load(x_ptr + row_base + dd, mask=is_entry, other=0.0).to(
                    tl.float32
                )
                # dd must be a non-target class: recheck membership cheaply
                # via the ist buffer.
                dd_member = (
                    tl.load(ist_ptr + row_base + dd, mask=is_entry, other=1.0) != 0.0
                )
                marg_d = 1.0 - xj + xd
                acc -= tl.where(is_entry & (~dd_member) & (marg_d > 0.0), 1.0, 0.0)
    g = tl.load(g_ptr + row, mask=lmask, other=0.0).to(tl.float32)
    tl.store(out_ptr + lane, acc / c * g * scale, mask=lmask)


def multilabel_margin_loss_backward(grad_output, self, target, reduction, is_target):
    """Gradient of multilabel_margin_loss (matches autograd)."""
    if isinstance(reduction, bool) or not isinstance(reduction, int):
        raise RuntimeError(
            "multilabel_margin_loss_backward: reduction must be an integer (0, 1 or 2)"
        )
    n, c = self.shape
    x = self.contiguous()
    ist = is_target.contiguous()
    if (
        isinstance(grad_output, torch.Tensor)
        and grad_output.dim() == 1
        and grad_output.numel() == n
    ):
        g = grad_output.contiguous().to(torch.float32)
    else:
        g = torch.full((n,), float(grad_output), dtype=torch.float32, device=x.device)
    scale = 1.0 / n if (reduction == 1 and n) else 1.0
    lanes = n * c
    out = torch.empty(lanes, dtype=torch.float32, device=x.device)
    if lanes:
        _multilabel_margin_loss_backward_kernel[(triton.cdiv(lanes, 128),)](
            x, target.contiguous(), ist, g, out, lanes, c, scale, BLOCK=128
        )
    return out.reshape(n, c).to(x.dtype)
