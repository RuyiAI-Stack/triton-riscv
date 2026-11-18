import torch
import triton
import triton.language as tl


@triton.jit
def binary_cross_entropy_with_logits_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    x32 = x.to(tl.float32)
    y32 = y.to(tl.float32)
    # Stable form matching torch's infinite-logit corners: x*(1-y) +
    # softplus(-x) with softplus(z) = max(z,0) + log(1+exp(-|z|)). This
    # gives (inf,0)->inf, (inf,1)->NaN, (-inf,*)->NaN like torch (the
    # max(x,0)-x*y form mirrors those corners the other way around).
    loss = (
        x32 * (1.0 - y32) + tl.maximum(-x32, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(x32)))
    )

    tl.store(out_ptr + offsets, loss.to(x.dtype), mask=mask)


def binary_cross_entropy_with_logits(x, y, reduction="mean"):
    if isinstance(y, torch.Tensor) and y.shape != x.shape:
        # torch: "Target size ... must be the same as input size"
        raise ValueError(
            f"Target size ({tuple(y.shape)}) must be the same as input "
            f"size ({tuple(x.shape)})"
        )
    if isinstance(y, torch.Tensor) and y.dtype != x.dtype:
        dt = torch.promote_types(x.dtype, y.dtype)
        x = x.to(dt)
        y = y.to(dt)
    if x.dtype == torch.float64:
        return torch.nn.functional.binary_cross_entropy_with_logits(
            x, y, reduction=reduction
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
    binary_cross_entropy_with_logits_kernel[grid](
        x_c, y_c, out, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )

    if reduction == "mean":
        return out.mean()
    if reduction == "sum":
        return out.sum()
    return out
