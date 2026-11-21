import torch
import triton
import triton.language as tl

# nll_loss2d (aten::nll_loss2d): input (N, C, H, W), target (N, H, W) int64.
# Probed on torch CPU: reduction=none writes 0 at ignored positions; mean
# divides by the summed class weight of the non-ignored positions (nan when
# everything is ignored); a target value >= C raises IndexError and a
# non-int64 target raises RuntimeError.
#
# The kernel writes per-element loss contributions and weight counts; the
# reduction is summed on the host.


@triton.jit
def _nll_loss2d_kernel(
    x_ptr,
    t_ptr,
    w_ptr,
    out_ptr,
    wcount_ptr,
    n_elements,
    spatial,
    channels,
    ignore_index,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    t = tl.load(t_ptr + offsets, mask=mask, other=0).to(tl.int64)
    sample = offsets // spatial
    within = offsets - sample * spatial
    valid = mask & (t != ignore_index) & (t >= 0)
    x_off = (sample * channels + t) * spatial + within
    x = tl.load(x_ptr + x_off, mask=valid, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + t, mask=valid, other=0.0).to(tl.float32)
    contrib = tl.where(valid, -w * x, 0.0)
    wcount = tl.where(valid, w, 0.0)
    tl.store(out_ptr + offsets, contrib, mask=mask)
    tl.store(wcount_ptr + offsets, wcount, mask=mask)


def nll_loss2d(self, target, weight=None, reduction=1, ignore_index=-100):
    """Negative log likelihood over (N, C, H, W) inputs."""
    if target.dtype not in (torch.long, torch.uint8):
        raise RuntimeError(
            f"expected target dtype to be Long or Byte, but got {target.dtype}"
        )
    n, c = self.shape[0], self.shape[1]
    x = self.contiguous()
    t = target.contiguous()
    bad = (t != ignore_index) & ((t < 0) | (t >= c))
    if bool(bad.any()):
        raise IndexError(f"Target {int(t[bad][0])} is out of bounds.")
    if weight is None:
        w = torch.ones(c, dtype=x.dtype, device=x.device)
    else:
        w = weight.contiguous()
    spatial = x.numel() // (n * c) if n and c else 0
    elems = n * spatial
    contrib = torch.empty(elems, dtype=torch.float32, device=x.device)
    wcount = torch.empty(elems, dtype=torch.float32, device=x.device)
    if elems:
        _nll_loss2d_kernel[(triton.cdiv(elems, 1024),)](
            x, t, w, contrib, wcount, elems, spatial, c, int(ignore_index), BLOCK=1024
        )
    out_shape = target.shape
    if reduction == 0:
        return contrib.reshape(out_shape).to(x.dtype)
    total = contrib.sum()
    if reduction == 2:
        return total.to(x.dtype)
    return (total / wcount.sum()).to(x.dtype)
