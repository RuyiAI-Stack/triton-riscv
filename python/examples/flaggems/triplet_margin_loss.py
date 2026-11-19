import torch
import triton
import triton.language as tl

# triplet_margin_loss: relu(d(a, p) - d(a, n) + margin) where the distance
# follows pairwise_distance semantics: eps is broadcast onto the difference
# VECTOR, d(x, y) = (sum |x - y + eps|^p)^(1/p) (probed against torch CPU:
# adding eps inside the norm or after it both mismatch). swap replaces d(a,n)
# with min(d(a, n), d(p, n)). Probed on torch CPU: margin must be > 0
# (ValueError otherwise). ATen accepts any real p, but pow does not lower
# on this toolchain, so this port supports p in {1, 2} and rejects anything
# else with a ValueError (documented deviation).
#
# One lane per row: the distance accumulations loop over the feature axis
# in-kernel; the reduction runs on the host.


@triton.jit
def _triplet_margin_loss_kernel(
    a_ptr,
    p_ptr,
    n_ptr,
    out_ptr,
    rows,
    d,
    margin,
    eps,
    P2: tl.constexpr,
    SWAP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    rmask = row < rows
    acc_ap = tl.zeros([BLOCK], dtype=tl.float32)
    acc_an = tl.zeros([BLOCK], dtype=tl.float32)
    acc_pn = tl.zeros([BLOCK], dtype=tl.float32)
    for k in range(d):
        av = tl.load(a_ptr + row * d + k, mask=rmask, other=0.0).to(tl.float32)
        pv = tl.load(p_ptr + row * d + k, mask=rmask, other=0.0).to(tl.float32)
        nv = tl.load(n_ptr + row * d + k, mask=rmask, other=0.0).to(tl.float32)
        # eps lands on every element of the difference vector.
        dap = av - pv + eps
        dan = av - nv + eps
        if P2:
            acc_ap += dap * dap
            acc_an += dan * dan
        else:
            acc_ap += tl.abs(dap)
            acc_an += tl.abs(dan)
        if SWAP:
            dpn = pv - nv + eps
            if P2:
                acc_pn += dpn * dpn
            else:
                acc_pn += tl.abs(dpn)
    if P2:
        d_ap = tl.sqrt(acc_ap)
        d_an = tl.sqrt(acc_an)
    else:
        d_ap = acc_ap
        d_an = acc_an
    eff_an = d_an
    if SWAP:
        if P2:
            d_pn = tl.sqrt(acc_pn)
        else:
            d_pn = acc_pn
        eff_an = tl.minimum(d_an, d_pn)
    loss = tl.maximum(d_ap - eff_an + margin, 0.0)
    tl.store(out_ptr + row, loss, mask=rmask)


def triplet_margin_loss(
    anchor, positive, negative, margin=1.0, p=2, eps=1e-6, swap=False, reduction=1
):
    """Triplet margin loss with p in {1, 2}."""
    if margin is None or float(margin) <= 0:
        raise ValueError(f"margin must be greater than 0, got {margin}")
    if p not in (1, 2):
        raise ValueError("only p == 1 and p == 2 supported on this backend")
    shape = torch.broadcast_shapes(anchor.shape, positive.shape, negative.shape)
    a = anchor.expand(shape).contiguous()
    pos = positive.expand(shape).contiguous()
    neg = negative.expand(shape).contiguous()
    rows = shape[0]
    d = shape[1]
    out = torch.empty(rows, dtype=a.dtype, device=a.device)
    if rows:
        _triplet_margin_loss_kernel[(triton.cdiv(rows, 128),)](
            a,
            pos,
            neg,
            out,
            rows,
            d,
            float(margin),
            float(eps),
            P2=p == 2,
            SWAP=bool(swap),
            BLOCK=128,
        )
    if reduction == 0:
        return out
    if reduction == 2:
        return out.sum()
    return out.sum() / rows if rows else out.sum()
