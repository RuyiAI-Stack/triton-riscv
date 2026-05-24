import torch
import triton
import triton.language as tl


@triton.jit
def margin_ranking_loss_kernel(
    x1_ptr,
    x2_ptr,
    target_ptr,
    out_ptr,
    n_elements,
    margin,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x1 = tl.load(x1_ptr + offsets, mask=mask, other=0)
    x2 = tl.load(x2_ptr + offsets, mask=mask, other=0)
    y = tl.load(target_ptr + offsets, mask=mask, other=0)

    diff = x1 - x2
    m = tl.full([BLOCK_SIZE], margin, tl.float32)
    val = -y * diff + m
    zero = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    loss = tl.maximum(val, zero)

    tl.store(out_ptr + offsets, loss.to(x1.dtype), mask=mask)


@triton.jit
def margin_ranking_loss_backward_kernel(
    grad_output_ptr,
    x1_ptr,
    x2_ptr,
    y_ptr,
    grad_x1_ptr,
    grad_x2_ptr,
    margin,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    grad_output = tl.load(grad_output_ptr + offsets, mask=mask, other=0.0)
    x1 = tl.load(x1_ptr + offsets, mask=mask, other=0.0)
    x2 = tl.load(x2_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    diff = x1 - x2
    m = tl.full([BLOCK_SIZE], margin, tl.float32)
    val = -y * diff + m
    active_mask = val > 0

    grad_x1 = tl.where(active_mask, -y * grad_output, 0.0)
    grad_x2 = tl.where(active_mask, y * grad_output, 0.0)

    tl.store(grad_x1_ptr + offsets, grad_x1.to(x1.dtype), mask=mask)
    tl.store(grad_x2_ptr + offsets, grad_x2.to(x1.dtype), mask=mask)


class MarginRankingLossOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1, x2, target, margin=0.0, reduction="mean"):
        if not (
            x1.is_floating_point()
            and x2.is_floating_point()
            and target.is_floating_point()
        ):
            raise ValueError("All inputs must be floating point tensors")

        if isinstance(reduction, int):
            reduction = {0: "none", 1: "mean", 2: "sum"}.get(reduction, "mean")
        if reduction not in ("none", "mean", "sum"):
            raise ValueError(
                "reduction must be one of 'none', 'mean', or 'sum'"
            )

        x1_b, x2_b, tgt_b = torch.broadcast_tensors(x1, x2, target)

        common_dtype = (
            x1_b.dtype if x1_b.is_floating_point() else torch.float32
        )
        x1_b = x1_b.to(dtype=common_dtype)
        x2_b = x2_b.to(dtype=common_dtype)
        tgt_b = tgt_b.to(dtype=common_dtype)

        x1_c = x1_b.contiguous().view(-1)
        x2_c = x2_b.contiguous().view(-1)
        tgt_c = tgt_b.contiguous().view(-1)

        out = torch.empty_like(x1_c)
        n_elements = out.numel()

        if n_elements == 0:
            if reduction == "none":
                return out.view(x1_b.shape)
            elif reduction == "sum":
                return out.sum()
            else:
                return out.mean()

        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        margin_ranking_loss_kernel[grid](
            x1_c,
            x2_c,
            tgt_c,
            out,
            n_elements,
            float(margin),
            BLOCK_SIZE=BLOCK_SIZE,
        )

        ctx.save_for_backward(x1_c, x2_c, tgt_c)
        ctx.reduction = reduction
        ctx.margin = margin
        ctx.n_elements = n_elements
        ctx.original_shape = x1_b.shape

        if reduction == "none":
            return out.view(x1_b.shape)
        elif reduction == "sum":
            return out.sum()
        else:
            return out.mean()

    @staticmethod
    def backward(ctx, grad_output):
        x1, x2, y = ctx.saved_tensors
        margin = ctx.margin
        reduction = ctx.reduction
        n_elements = ctx.n_elements

        if n_elements == 0:
            grad_x1 = torch.zeros_like(x1)
            grad_x2 = torch.zeros_like(x2)
            grad_target = torch.zeros_like(y)
            return grad_x1, grad_x2, grad_target, None, None

        if reduction == "mean":
            grad_output = grad_output.expand(n_elements) / n_elements
        elif reduction == "sum":
            grad_output = grad_output.expand(n_elements)
        else:
            grad_output = grad_output.contiguous().view(-1)

        grad_output = grad_output.contiguous()

        grad_x1 = torch.empty_like(x1)
        grad_x2 = torch.empty_like(x2)

        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        margin_ranking_loss_backward_kernel[grid](
            grad_output,
            x1,
            x2,
            y,
            grad_x1,
            grad_x2,
            float(margin),
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        original_shape = ctx.original_shape
        grad_x1 = grad_x1.view(original_shape)
        grad_x2 = grad_x2.view(original_shape)

        grad_target = torch.zeros_like(y).view(original_shape)
        return grad_x1, grad_x2, grad_target, None, None


def margin_ranking_loss(x1, x2, target, margin=0.0, reduction="mean"):
    return MarginRankingLossOp.apply(x1, x2, target, margin, reduction)
