import torch
import triton
import triton.language as tl


@triton.jit
def kl_div_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    log_target: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    x32 = x.to(tl.float32)
    y32 = y.to(tl.float32)
    if log_target:
        # target holds log-probabilities: pointwise = exp(y) * (y - x).
        loss = tl.exp(y32) * (y32 - x32)
    else:
        # xlogy(target, target) - target * input; zero targets give zero.
        tlog = tl.where(y32 > 0.0, tl.log(tl.where(y32 > 0.0, y32, 1.0)), 0.0)
        loss = y32 * (tlog - x32)

    tl.store(out_ptr + offsets, loss.to(x.dtype), mask=mask)


def kl_div(x, y, reduction="mean", log_target=False):
    if not x.dtype.is_floating_point or (
        isinstance(y, torch.Tensor) and not y.dtype.is_floating_point
    ):
        raise RuntimeError("kl_div: Integral inputs not supported.")
    if isinstance(y, torch.Tensor):
        x, y = torch.broadcast_tensors(x, y)
    if x.dtype == torch.float64:
        return torch.nn.functional.kl_div(
            x, y, reduction=reduction, log_target=log_target
        )

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
    kl_div_kernel[grid](x_c, y_c, out, n_elements, log_target, BLOCK_SIZE=BLOCK_SIZE)

    if reduction == "mean":
        return out.mean()
    if reduction == "sum":
        return out.sum()
    return out
