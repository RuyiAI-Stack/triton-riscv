import torch
import triton
import triton.language as tl


@triton.jit
def _binary_cross_entropy_backward_kernel(
    grad_output_ptr,
    input_ptr,
    target_ptr,
    weight_ptr,
    out_ptr,
    n_elements,
    reduction_elements,
    reduction: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    GRAD_OUTPUT_SCALAR: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # BCE backward: grad = grad_output * (p - t) / (p * (1 - p)), optionally
    # weighted.  Computed in fp32 to avoid fp16/bf16 precision loss in the
    # division.
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    p = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(target_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    denom = p * (1.0 - p)
    # Clamp the denominator to 1e-6 so p at (or numerically at) 0/1 stays
    # finite (upstream FlagGems semantics; torch returns inf/NaN there).
    denom = tl.where(denom < 1e-6, 1e-6, denom)
    grad = (p - y) / denom

    if HAS_WEIGHT:
        w = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        grad = grad * w

    if GRAD_OUTPUT_SCALAR:
        g = tl.load(grad_output_ptr).to(tl.float32)
    else:
        g = tl.load(grad_output_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    if reduction == 1:  # mean
        g = g * (1.0 / reduction_elements)
    # reduction == 0 (none) and 2 (sum): the per-element grad is unchanged.

    tl.store(out_ptr + offsets, grad * g, mask=mask)


def _normalize_reduction(reduction):
    if isinstance(reduction, str):
        if reduction == "none":
            return 0
        if reduction == "mean":
            return 1
        if reduction == "sum":
            return 2
    elif isinstance(reduction, int) and reduction in (0, 1, 2):
        return reduction
    raise ValueError("reduction must be one of 'none', 'mean', or 'sum'")


def binary_cross_entropy_backward(
    grad_output, self, target, weight=None, reduction: str | int = "mean"
):
    reduction = _normalize_reduction(reduction)

    if not self.dtype.is_floating_point:
        raise RuntimeError(
            "binary_cross_entropy_backward(): expected a floating-point "
            f"tensor, but got {self.dtype}"
        )
    for other in (grad_output, target):
        if other.dtype != self.dtype:
            raise RuntimeError(f"Found dtype {other.dtype} but expected {self.dtype}")
    if weight is not None and weight.dtype != self.dtype:
        raise RuntimeError(f"Found dtype {weight.dtype} but expected {self.dtype}")

    if self.dtype == torch.float64:
        # aten keeps full double precision here; mirror it instead of
        # rounding through the fp32 kernel.
        return torch.ops.aten.binary_cross_entropy_backward(
            grad_output, self, target, weight, reduction
        )

    # aten takes the element count for the mean reduction before
    # broadcasting self against target.
    reduction_elements = self.numel()

    self, target = torch.broadcast_tensors(self, target)
    if weight is not None:
        self, target, weight = torch.broadcast_tensors(self, target, weight)
    if grad_output.numel() != 1:
        grad_output = torch.broadcast_to(grad_output, self.shape)

    out = torch.empty_like(self)
    n_elements = self.numel()
    if n_elements == 0:
        return out

    g_c = grad_output.contiguous()
    p_c = self.contiguous()
    t_c = target.contiguous()
    w_c = weight.contiguous() if weight is not None else g_c

    out_c = out if out.is_contiguous() else torch.empty_like(p_c)

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _binary_cross_entropy_backward_kernel[grid](
        g_c,
        p_c,
        t_c,
        w_c,
        out_c,
        n_elements,
        reduction_elements,
        reduction=reduction,
        HAS_WEIGHT=weight is not None,
        GRAD_OUTPUT_SCALAR=g_c.numel() == 1,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    if out_c is not out:
        out.copy_(out_c)

    return out
