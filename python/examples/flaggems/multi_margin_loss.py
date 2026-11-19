import torch
import triton
import triton.language as tl

# multi_margin_loss: loss_i = weight[y_i] * (sum_{j != y_i}
# max(0, margin - x[y_i] + x[j])^p) / C, probed on torch CPU (the weight
# multiplies the whole row and the sum skips j == y_i). p in {1, 2};
# anything else raises ValueError. Out-of-range targets raise RuntimeError
# ("target out of range") and non-int64 targets raise RuntimeError.
#
# The kernel writes per-(i, j) contributions; the row sum, /C division and
# the reduction run on the host.


@triton.jit
def _multi_margin_loss_kernel(
    x_ptr,
    t_ptr,
    w_ptr,
    out_ptr,
    n_elements,
    c,
    margin,
    P2: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    row = offsets // c
    col = offsets - row * c
    y = tl.load(t_ptr + row, mask=mask, other=0).to(tl.int64)
    xj = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    xy = tl.load(x_ptr + row * c + y, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + y, mask=mask, other=0.0).to(tl.float32)
    hinge = margin - xy + xj
    hinge = tl.where((col != y) & (hinge > 0), hinge, 0.0)
    if P2:
        hinge = hinge * hinge
    tl.store(out_ptr + offsets, w * hinge, mask=mask)


def multi_margin_loss(self, target, p=1, margin=1.0, weight=None, reduction=1):
    """Hinge margin loss over (N, C) inputs."""
    if p not in (1, 2):
        raise ValueError("only p == 1 and p == 2 supported")
    if target.dtype != torch.long:
        raise RuntimeError(f"expected scalar type Long but found {target.dtype}")
    n, c = self.shape
    x = self.contiguous()
    t = target.contiguous()
    bad = (t < 0) | (t >= c)
    if bool(bad.any()):
        raise RuntimeError("target out of range")
    if weight is None:
        w = torch.ones(c, dtype=x.dtype, device=x.device)
    else:
        w = weight.contiguous()
    elems = n * c
    contrib = torch.empty(elems, dtype=torch.float32, device=x.device)
    if elems:
        _multi_margin_loss_kernel[(triton.cdiv(elems, 1024),)](
            x,
            t,
            w,
            contrib,
            elems,
            c,
            float(margin),
            P2=p == 2,
            BLOCK=1024,
        )
    rows = contrib.reshape(n, c).sum(dim=1) / c
    if reduction == 0:
        return rows.to(x.dtype)
    if reduction == 2:
        return rows.sum().to(x.dtype)
    return (rows.sum() / n).to(x.dtype) if n else rows.sum().to(x.dtype)
