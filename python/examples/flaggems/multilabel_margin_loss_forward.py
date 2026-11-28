import torch
import triton
import triton.language as tl

# multilabel_margin_loss_forward: the forward flavor returning
# (loss, is_target). is_target is the probed 0/1 float32 class-membership
# mask of the target sequence prefix (duplicates mark the class once);
# the loss follows the pinned ranking formula
#   sum over sequence entries j of sum over non-target classes d of
#   relu(1 - x[t[j]] + x[d]), divided by C.


@triton.jit
def _multilabel_margin_loss_forward_kernel(
    x_ptr,
    t_ptr,
    out_ptr,
    ist_ptr,
    n_lanes,
    c,
    BLOCK: tl.constexpr,
):
    lane = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    lmask = lane < n_lanes
    row = lane // c
    d = lane - row * c
    row_base = row * c

    stop = tl.zeros([BLOCK], dtype=tl.int32) + c
    for jj in range(c):
        tj = tl.load(t_ptr + row_base + jj, mask=lmask, other=-1).to(tl.int32)
        stop = tl.minimum(stop, tl.where(tj < 0, jj, c))
    member = tl.zeros([BLOCK], dtype=tl.int1)
    for jj in range(c):
        tj = tl.load(t_ptr + row_base + jj, mask=lmask, other=-1).to(tl.int32)
        member = member | (lmask & (jj < stop) & (tj == d.to(tl.int32)))
    xd = tl.load(x_ptr + lane, mask=lmask & (~member), other=0.0).to(tl.float32)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for jj in range(c):
        tj = tl.load(t_ptr + row_base + jj, mask=lmask, other=-1).to(tl.int32)
        active = lmask & (~member) & (jj < stop)
        xj = tl.load(x_ptr + row_base + tj, mask=active, other=0.0).to(tl.float32)
        marg = 1.0 - xj + xd
        acc += tl.where(active & (marg > 0.0), marg, 0.0)
    tl.store(out_ptr + lane, acc / c, mask=lmask)
    tl.store(ist_ptr + lane, tl.where(member, 1.0, 0.0), mask=lmask)


def _prefix_bad_target(t, c):
    n = t.shape[0]
    neg = t < 0
    has_neg = neg.any(dim=1)
    first_neg = torch.where(has_neg, neg.float().argmax(dim=1), torch.full((n,), c))
    idx = torch.arange(c, device=t.device).unsqueeze(0)
    prefix = idx < first_neg.unsqueeze(1)
    return bool((prefix & (t >= c)).any())


def multilabel_margin_loss_forward(self, target, reduction=1):
    """Ranking loss forward returning (loss, is_target)."""
    n, c = self.shape
    x = self.contiguous()
    t = target.contiguous()
    if _prefix_bad_target(t, c):
        raise RuntimeError("argument #2 'target' is out of range")
    lanes = n * c
    contrib = torch.empty(lanes, dtype=torch.float32, device=x.device)
    is_target = torch.empty(lanes, dtype=torch.float32, device=x.device)
    if lanes:
        _multilabel_margin_loss_forward_kernel[(triton.cdiv(lanes, 128),)](
            x, t, contrib, is_target, lanes, c, BLOCK=128
        )
    rows = contrib.reshape(n, c).sum(dim=1)
    if reduction == 0:
        return rows.to(x.dtype), is_target.reshape(n, c)
    if reduction == 2:
        return rows.sum().to(x.dtype), is_target.reshape(n, c)
    total = (rows.sum() / n).to(x.dtype) if n else rows.sum().to(x.dtype)
    return total, is_target.reshape(n, c)
