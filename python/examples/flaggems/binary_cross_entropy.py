import torch
import triton
import triton.language as tl


@triton.jit
def binary_cross_entropy_kernel(
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
    # aten clamps the logarithms to -100.
    log_x = tl.maximum(tl.log(x32), -100.0)
    log_1mx = tl.maximum(tl.log(1.0 - x32), -100.0)
    loss = -(y32 * log_x + (1.0 - y32) * log_1mx)

    tl.store(out_ptr + offsets, loss.to(x.dtype), mask=mask)


def binary_cross_entropy(x, y, reduction="mean"):
    if isinstance(y, torch.Tensor) and y.shape != x.shape:
        raise ValueError(
            f"Using a target size ({tuple(y.shape)}) that is different "
            f"to the input size ({tuple(x.shape)}) is deprecated. "
            "Please ensure they have the same size."
        )
    if isinstance(y, torch.Tensor) and y.dtype != x.dtype:
        raise RuntimeError(f"Found dtype {y.dtype} but expected {x.dtype}")
    if x.dtype == torch.float64:
        return torch.nn.functional.binary_cross_entropy(x, y, reduction=reduction)

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
    binary_cross_entropy_kernel[grid](x_c, y_c, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    if reduction == "mean":
        return out.mean()
    if reduction == "sum":
        return out.sum()
    return out
