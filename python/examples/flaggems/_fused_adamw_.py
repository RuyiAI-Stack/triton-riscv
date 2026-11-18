import torch
import triton
import triton.language as tl


@triton.jit
def _fused_adamw_kernel(
    param,
    grad,
    exp_avg,
    exp_avg_sq,
    max_exp_avg_sq,
    n,
    BLOCK_SIZE: tl.constexpr,
    lr: tl.constexpr,
    beta1: tl.constexpr,
    beta2: tl.constexpr,
    weight_decay: tl.constexpr,
    eps: tl.constexpr,
    bias_correction1: tl.constexpr,
    bias_correction2: tl.constexpr,
    amsgrad: tl.constexpr,
    maximize: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    param_load = tl.load(param + offsets, mask=mask, other=0.0)
    grad_load = tl.load(grad + offsets, mask=mask, other=0.0)
    if maximize:
        grad_load = -grad_load
    if weight_decay != 0:
        param_load = param_load * (1 - lr * weight_decay)
    exp_avg_load = tl.load(exp_avg + offsets, mask=mask, other=0.0)
    exp_avg_sq_load = tl.load(exp_avg_sq + offsets, mask=mask, other=0.0)

    exp_avg_new = beta1 * exp_avg_load + (1 - beta1) * grad_load
    exp_avg_sq_new = beta2 * exp_avg_sq_load + (1 - beta2) * grad_load * grad_load

    corrected_exp_avg = exp_avg_new / bias_correction1

    if amsgrad:
        max_load = tl.load(max_exp_avg_sq + offsets, mask=mask, other=0.0)
        max_new = tl.maximum(max_load, exp_avg_sq_new)
        tl.store(max_exp_avg_sq + offsets, max_new, mask=mask)
        denom = tl.sqrt(max_new / bias_correction2) + eps
    else:
        denom = tl.sqrt(exp_avg_sq_new / bias_correction2) + eps

    param_new = param_load - lr * (corrected_exp_avg / denom)
    tl.store(param + offsets, param_new, mask=mask)
    tl.store(exp_avg + offsets, exp_avg_new, mask=mask)
    tl.store(exp_avg_sq + offsets, exp_avg_sq_new, mask=mask)


def _fused_adamw(
    params,
    grads,
    exp_avgs,
    exp_avg_sqs,
    max_exp_avg_sqs,
    state_steps,
    *,
    lr=0.001,
    beta1=0.9,
    beta2=0.999,
    weight_decay=0.0,
    eps=1e-8,
    amsgrad=False,
    maximize=False,
    grad_scale=None,
    found_inf=None,
):
    for p in params:
        assert p.dtype == torch.float32, "_fused_adamw_ only supports float32 inputs"

    if grad_scale is not None:
        grads = [g * grad_scale for g in grads]
    if found_inf is not None:
        if found_inf.item() > 0:
            return (params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs)

    for i in range(len(params)):
        param = params[i]
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i].item() if state_steps[i].numel() > 0 else 0
        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step

        n = param.numel()
        BLOCK_SIZE = max(min(triton.next_power_of_2(n), 4096), 128)
        grid = (triton.cdiv(n, BLOCK_SIZE),)
        _fused_adamw_kernel[grid](
            param,
            grad,
            exp_avg,
            exp_avg_sq,
            (max_exp_avg_sqs[i] if max_exp_avg_sqs else exp_avg_sq),
            n,
            BLOCK_SIZE,
            lr,
            beta1,
            beta2,
            weight_decay,
            eps,
            bias_correction1,
            bias_correction2,
            amsgrad,
            maximize,
        )
    return (params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs)


def _fused_adamw_(
    self,
    grads,
    exp_avgs,
    exp_avg_sqs,
    max_exp_avg_sqs,
    state_steps,
    *,
    lr=0.001,
    beta1=0.9,
    beta2=0.999,
    weight_decay=0.0,
    eps=1e-8,
    amsgrad=False,
    maximize=False,
    grad_scale=None,
    found_inf=None,
):
    _fused_adamw(
        self,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        weight_decay=weight_decay,
        eps=eps,
        amsgrad=amsgrad,
        maximize=maximize,
        grad_scale=grad_scale,
        found_inf=found_inf,
    )
    return None
