import torch
import triton
import triton.language as tl


@triton.jit
def huber_loss_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    delta,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    x32 = x.to(tl.float32)
    y32 = y.to(tl.float32)
    diff = x32 - y32
    absdiff = tl.abs(diff)
    loss = tl.where(
        absdiff <= delta, 0.5 * diff * diff, delta * (absdiff - 0.5 * delta)
    )

    tl.store(out_ptr + offsets, loss.to(x.dtype), mask=mask)


def huber_loss(x, y, reduction="mean", delta=1.0):
    if not x.dtype.is_floating_point or (
        isinstance(y, torch.Tensor) and not y.dtype.is_floating_point
    ):
        raise NotImplementedError('"huber_cpu" not implemented for integer inputs')
    if isinstance(y, torch.Tensor):
        x, y = torch.broadcast_tensors(x, y)
    if x.dtype == torch.float64:
        return torch.nn.functional.huber_loss(x, y, reduction=reduction, delta=delta)

    if delta <= 0:
        raise RuntimeError("huber_loss requires a positive delta")
    if reduction not in ("none", "mean", "sum"):
        raise ValueError("reduction must be one of 'none', 'mean', or 'sum'")
    x_c = x.contiguous()
    y_c = y.contiguous()
    out = torch.empty_like(x_c)
    n_elements = x_c.numel()
    if n_elements == 0:
        return (
            out.mean()
            if reduction == "mean"
            else (out.sum() if reduction == "sum" else out)
        )

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    huber_loss_kernel[grid](x_c, y_c, out, n_elements, delta, BLOCK_SIZE=BLOCK_SIZE)

    if reduction == "mean":
        return out.mean()
    if reduction == "sum":
        return out.sum()
    return out
