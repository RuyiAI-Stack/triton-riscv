import torch
import triton
import triton.language as tl

from .nllloss import (
    nll_loss_backward as nll_loss_2d_backward,
)
from .nllloss import (
    nll_loss_forward as nll_loss_2d,
)


@triton.jit
def nll_loss_nd_forward_kernel(
    input_ptr,
    target_ptr,
    weight_ptr,
    out_ptr,
    scratch_ptr,
    C,
    S,
    stride_in_n,
    stride_in_c,
    stride_in_s,
    stride_tgt_n,
    stride_tgt_s,
    ignore_index,
    HAS_WEIGHT: tl.constexpr,
    REDUCTION: tl.constexpr,
    BLOCK_S: tl.constexpr = 1024,
):
    pid_s = tl.program_id(0)
    pid_n = tl.program_id(1)

    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = s_offsets < S

    tgt_offsets = pid_n * stride_tgt_n + s_offsets * stride_tgt_s
    t = tl.load(target_ptr + tgt_offsets, mask=mask_s, other=ignore_index).to(tl.int32)

    valid = mask_s & (t != ignore_index) & (t >= 0) & (t < C)

    in_offsets = pid_n * stride_in_n + t * stride_in_c + s_offsets * stride_in_s
    val = tl.load(input_ptr + in_offsets, mask=valid, other=0.0).to(tl.float32)

    if HAS_WEIGHT:
        w = tl.load(weight_ptr + t, mask=valid, other=0.0).to(tl.float32)
        loss_val = tl.where(valid, -val * w, 0.0)
    else:
        w = tl.where(valid, 1.0, 0.0).to(tl.float32)
        loss_val = tl.where(valid, -val, 0.0)

    # none
    if REDUCTION == 0:
        out_offset = pid_n * S + s_offsets
        tl.store(
            out_ptr + out_offset,
            loss_val.to(out_ptr.dtype.element_ty),
            mask=mask_s,
        )
    else:
        block_loss_sum = tl.sum(loss_val, axis=0)
        # mean
        if REDUCTION == 1:
            block_weight_sum = tl.sum(w, axis=0)

            tl.atomic_add(scratch_ptr, block_loss_sum, sem="relaxed")
            tl.atomic_add(scratch_ptr + 1, block_weight_sum, sem="relaxed")

            old_cnt = tl.atomic_add(scratch_ptr + 2, 1.0, sem="release")

            total_programs = tl.num_programs(0) * tl.num_programs(1)

            if old_cnt == total_programs - 1.0:
                total_loss = tl.load(scratch_ptr)
                total_weight = tl.load(scratch_ptr + 1)
                final_val = tl.where(
                    total_weight == 0.0, 0.0, total_loss / total_weight
                )
                tl.store(out_ptr, final_val.to(out_ptr.dtype.element_ty))
        # Sum
        else:
            tl.atomic_add(scratch_ptr, block_loss_sum, sem="relaxed")

            old_cnt = tl.atomic_add(scratch_ptr + 2, 1.0, sem="release")
            total_programs = tl.num_programs(0) * tl.num_programs(1)

            if old_cnt == total_programs - 1.0:
                total_loss = tl.load(scratch_ptr)
                tl.store(out_ptr, total_loss.to(out_ptr.dtype.element_ty))


@triton.jit
def nll_loss_nd_reduce_kernel(
    input_ptr,
    target_ptr,
    weight_ptr,
    out_ptr,
    total_weight_ptr,
    N,
    C,
    S,
    stride_in_n,
    stride_in_c,
    stride_in_s,
    stride_tgt_n,
    stride_tgt_s,
    ignore_index,
    HAS_WEIGHT: tl.constexpr,
    REDUCTION: tl.constexpr,
):
    total_loss = tl.full((), 0.0, dtype=tl.float32)
    total_weight = tl.full((), 0.0, dtype=tl.float32)
    for offset in range(0, N * S):
        n_offset = offset // S
        s_offset = offset % S
        target_offset = n_offset * stride_tgt_n + s_offset * stride_tgt_s
        target = tl.load(target_ptr + target_offset).to(tl.int32)
        valid = (target != ignore_index) & (target >= 0) & (target < C)
        safe_target = tl.maximum(0, tl.minimum(target, C - 1))
        input_offset = (
            n_offset * stride_in_n + safe_target * stride_in_c + s_offset * stride_in_s
        )
        input_value = tl.load(input_ptr + input_offset).to(tl.float32)
        if HAS_WEIGHT:
            loaded_weight = tl.load(weight_ptr + safe_target).to(tl.float32)
            sample_weight = tl.where(valid, loaded_weight, 0.0)
        else:
            sample_weight = valid.to(tl.float32)
        total_loss += tl.where(valid, -input_value * sample_weight, 0.0)
        total_weight += sample_weight
    if REDUCTION == 1:
        result = tl.where(total_weight == 0.0, 0.0, total_loss / total_weight)
    else:
        result = total_loss
    tl.store(out_ptr, result.to(out_ptr.dtype.element_ty))
    tl.store(total_weight_ptr, total_weight)


@triton.jit
def nll_loss_nd_backward_kernel(
    grad_out_ptr,
    target_ptr,
    weight_ptr,
    grad_in_ptr,
    input_ptr,
    total_weight_ptr,
    C,
    S,
    stride_in_n,
    stride_in_c,
    stride_in_s,
    stride_tgt_n,
    stride_tgt_s,
    stride_go_n,
    stride_go_s,
    ignore_index,
    HAS_WEIGHT: tl.constexpr,
    REDUCTION: tl.constexpr,
    FUSE_LOG_SOFTMAX: tl.constexpr,
    BLOCK_S: tl.constexpr = 1024,
):
    pid_s = tl.program_id(0)
    pid_n = tl.program_id(1)

    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = s_offsets < S

    tgt_offsets = pid_n * stride_tgt_n + s_offsets * stride_tgt_s
    t = tl.load(target_ptr + tgt_offsets, mask=mask_s, other=ignore_index).to(tl.int32)

    valid = mask_s & (t != ignore_index) & (t >= 0) & (t < C)

    if REDUCTION == 0:  # none
        out_grad_offsets = pid_n * stride_go_n + s_offsets * stride_go_s
        out_grad = tl.load(grad_out_ptr + out_grad_offsets, mask=valid, other=0.0).to(
            tl.float32
        )
    else:  # mean or sum
        out_grad = tl.load(grad_out_ptr).to(tl.float32)

    if HAS_WEIGHT:
        w = tl.load(weight_ptr + t, mask=valid, other=0.0).to(tl.float32)
    else:
        w = tl.where(valid, 1.0, 0.0).to(tl.float32)

    if REDUCTION == 1:  # mean
        total_weight = tl.load(total_weight_ptr).to(tl.float32)
        grad_in_val = tl.where(total_weight != 0.0, -w * out_grad / total_weight, 0.0)
    else:  # sum or none
        grad_in_val = -w * out_grad

    if FUSE_LOG_SOFTMAX:
        for channel in range(0, C):
            in_offsets = (
                pid_n * stride_in_n + channel * stride_in_c + s_offsets * stride_in_s
            )
            log_probability = tl.load(
                input_ptr + in_offsets, mask=mask_s, other=0.0
            ).to(tl.float32)
            channel_grad = grad_in_val * (
                (t == channel).to(tl.float32) - tl.exp(log_probability)
            )
            tl.store(
                grad_in_ptr + in_offsets,
                channel_grad.to(grad_in_ptr.dtype.element_ty),
                mask=mask_s,
            )
    else:
        in_offsets = pid_n * stride_in_n + t * stride_in_c + s_offsets * stride_in_s
        tl.store(
            grad_in_ptr + in_offsets,
            grad_in_val.to(grad_in_ptr.dtype.element_ty),
            mask=valid,
        )


def nll_loss_nd_forward(
    input: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor = None,
    reduction: int = 1,
    ignore_index: int = -100,
):
    if input.dim() < 3:
        out, total_weight = nll_loss_2d(
            input,
            target,
            weight=weight,
            reduction=reduction,
            ignore_index=ignore_index,
        )
        return out, total_weight
    else:
        N = input.shape[0]
        C = input.shape[1]
        S = input.numel() // (N * C)

        inp = input.reshape(N, C, S)

        if target.numel() != N * S:
            raise ValueError(
                f"Target size {target.shape} doesn't match input size (N={N}, S={S})"
            )
        else:
            tgt = target.reshape(N, S)

        stride_in_n, stride_in_c, stride_in_s = inp.stride()
        stride_tgt_n, stride_tgt_s = tgt.stride()

        if weight is None:
            has_weight = False
            w = input
        else:
            has_weight = True
            if weight.numel() != C:
                raise ValueError(f"Weight shape {weight.shape} must be ({C},)")
            w = weight.contiguous()

        if reduction not in [0, 1, 2]:
            raise ValueError("reduction must be 0 ('none'), 1 ('mean'), or 2 ('sum')")

        BLOCK_S = 1024
        grid = (triton.cdiv(S, BLOCK_S), N)
        if reduction == 0:
            out = torch.empty((N, S), device=input.device, dtype=input.dtype)
            scratch = torch.empty(1, device=input.device)

            nll_loss_nd_forward_kernel[grid](
                inp,
                tgt,
                w,
                out,
                scratch,
                C,
                S,
                stride_in_n,
                stride_in_c,
                stride_in_s,
                stride_tgt_n,
                stride_tgt_s,
                ignore_index,
                HAS_WEIGHT=has_weight,
                REDUCTION=reduction,
                BLOCK_S=BLOCK_S,
            )

            if target.dim() == input.dim() - 1:
                res = out.view_as(target)
            else:
                res = out.reshape(target.shape)

            total_weight = torch.empty([], device=input.device, dtype=input.dtype)
            return res, total_weight

        else:
            out = torch.empty(1, device=input.device, dtype=input.dtype)
            scratch = torch.empty(1, device=input.device, dtype=torch.float32)

            nll_loss_nd_reduce_kernel[(1,)](
                inp,
                tgt,
                w,
                out,
                scratch,
                N,
                C,
                S,
                stride_in_n,
                stride_in_c,
                stride_in_s,
                stride_tgt_n,
                stride_tgt_s,
                ignore_index,
                HAS_WEIGHT=has_weight,
                REDUCTION=reduction,
            )
            out = out[0]

            if reduction == 1:
                total_weight = scratch[0]
            else:
                total_weight = torch.empty([], device=input.device, dtype=input.dtype)

            return out, total_weight


def nll_loss_nd_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor = None,
    reduction: int = 1,
    ignore_index: int = -100,
    total_weight: torch.Tensor = None,
    _fuse_log_softmax: bool = False,
):
    if input.dim() < 3:
        return nll_loss_2d_backward(
            grad_output,
            input,
            target,
            weight=weight,
            reduction=reduction,
            ignore_index=ignore_index,
            total_weight=total_weight,
        )
    else:
        grad_input = torch.zeros_like(input)

        N = input.shape[0]
        C = input.shape[1]
        S = input.numel() // (N * C)

        grad_inp = grad_input.reshape(N, C, S)
        tgt = target.reshape(N, S)

        stride_in_n, stride_in_c, stride_in_s = grad_inp.stride()
        stride_tgt_n, stride_tgt_s = tgt.stride()

        if weight is None:
            has_weight = False
            w = input
        else:
            has_weight = True
            w = weight.contiguous()

        if reduction == 0:
            grad_out = grad_output.reshape(N, S)
            stride_go_n, stride_go_s = grad_out.stride()
        else:
            grad_out = grad_output
            stride_go_n, stride_go_s = 0, 0

        BLOCK_S = 1024
        grid = (triton.cdiv(S, BLOCK_S), N)

        nll_loss_nd_backward_kernel[grid](
            grad_out,
            tgt,
            w,
            grad_input,
            input,
            total_weight,
            C,
            S,
            stride_in_n,
            stride_in_c,
            stride_in_s,
            stride_tgt_n,
            stride_tgt_s,
            stride_go_n,
            stride_go_s,
            ignore_index,
            HAS_WEIGHT=has_weight,
            REDUCTION=reduction,
            FUSE_LOG_SOFTMAX=_fuse_log_softmax,
            BLOCK_S=BLOCK_S,
        )

        return grad_input


def nll_loss_nd(input, target, weight=None, reduction=1, ignore_index=-100):
    out, total_weight = nll_loss_nd_forward(
        input, target, weight, reduction, ignore_index
    )
    if input.requires_grad:

        class NLLLossNdFn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, inp, tgt, w, red, ign):
                out, tw = nll_loss_nd_forward(inp, tgt, w, red, ign)
                ctx.save_for_backward(inp, tgt, w, tw)
                ctx.reduction = red
                ctx.ignore_index = ign
                return out

            @staticmethod
            def backward(ctx, grad_output):
                inp, tgt, w, tw = ctx.saved_tensors
                grad = nll_loss_nd_backward(
                    grad_output,
                    inp,
                    tgt,
                    w,
                    ctx.reduction,
                    ctx.ignore_index,
                    tw,
                    _fuse_log_softmax=False,
                )
                return grad, None, None, None, None

        return NLLLossNdFn.apply(input, target, weight, reduction, ignore_index)
    return out
