import torch
import triton
import triton.language as tl


@triton.jit
def _fused_adam_kernel(
    param,
    grad,
    exp_avg,
    exp_avg_sq,
    max_exp_avg_sq,
    n: tl.constexpr,
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
    # Get the block/thread index
    pid = tl.program_id(0)
    # Calculate the starting offset for this block
    param_offset = pid * BLOCK_SIZE
    # Create a range of offsets for this block
    offsets = param_offset + tl.arange(0, BLOCK_SIZE)
    # Create a mask to avoid out-of-bounds access
    mask = offsets < n

    # Load parameters
    # param: the model parameter
    param_load = tl.load(param + offsets, mask=mask, other=0.0)
    # grad: the gradient
    grad_load = tl.load(grad + offsets, mask=mask, other=0.0)

    # Match aten::_fused_adam_: maximize applies before coupled weight decay.
    if maximize:
        grad_load = -grad_load
    if weight_decay > 0:
        grad_load += weight_decay * param_load

    # Load first moment estimate (exp_avg)
    exp_avg_load = tl.load(exp_avg + offsets, mask=mask, other=0.0)
    # Load second moment estimate (exp_avg_sq)
    exp_avg_sq_load = tl.load(exp_avg_sq + offsets, mask=mask, other=0.0)

    # Update first moment estimate: m = beta1 * m + (1 - beta1) * g
    exp_avg_new = beta1 * exp_avg_load + (1 - beta1) * grad_load
    # Update second moment estimate: v = beta2 * v + (1 - beta2) * g^2
    exp_avg_sq_new = beta2 * exp_avg_sq_load + (1 - beta2) * grad_load * grad_load

    # Apply bias correction
    corrected_exp_avg = exp_avg_new / bias_correction1
    corrected_exp_avg_sq = exp_avg_sq_new / bias_correction2

    # Compute the denominator: sqrt(v_hat) + eps
    if amsgrad:
        # Load max second moment estimate
        max_exp_avg_sq_load = tl.load(max_exp_avg_sq + offsets, mask=mask, other=0.0)
        # Keep the optimizer state in uncorrected second-moment units, matching
        # aten::_fused_adam_.
        max_exp_avg_sq_new = tl.maximum(max_exp_avg_sq_load, exp_avg_sq_new)
        # Store updated max
        tl.store(max_exp_avg_sq + offsets, max_exp_avg_sq_new, mask=mask)
        # Apply bias correction only when forming the denominator.
        denom = tl.sqrt(max_exp_avg_sq_new / bias_correction2) + eps
    else:
        denom = tl.sqrt(corrected_exp_avg_sq) + eps

    update = corrected_exp_avg / denom

    # Update parameters
    param_new = param_load - lr * update

    # Store updated values
    tl.store(param + offsets, param_new, mask=mask)
    tl.store(exp_avg + offsets, exp_avg_new, mask=mask)
    tl.store(exp_avg_sq + offsets, exp_avg_sq_new, mask=mask)


def _fused_adam(
    params,
    grads,
    exp_avgs,
    exp_avg_sqs,
    max_exp_avg_sqs,
    state_steps,
    *,
    lr: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    weight_decay: float = 0.0,
    eps: float = 1e-8,
    amsgrad: bool = False,
    maximize: bool = False,
    grad_scale=None,
    found_inf=None,
):
    """Fused Adam optimizer step.

    Performs one step of the Adam optimizer algorithm:
    - m = beta1 * m + (1 - beta1) * g
    - v = beta2 * v + (1 - beta2) * g^2
    - if amsgrad: v_hat = max(v_hat, v)
    - applies coupled L2 weight decay, matching aten::_fused_adam_.
    """
    # Adam optimizer state must be float32
    for p in params:
        assert p.dtype == torch.float32, "_fused_adam only supports float32 inputs"

    # Handle grad_scale and found_inf (for gradient scaling and gradient skipping)
    if grad_scale is not None:
        grads = [g / grad_scale for g in grads]
    if found_inf is not None:
        # Skip update if found_inf is True
        if found_inf.item() > 0:
            return (params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs)

    # Process each parameter group
    for i in range(len(params)):
        param = params[i]
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i].item() if state_steps[i].numel() > 0 else 0

        # Bias correction for first and second moment estimates
        # bias_correction1 = 1 - beta1^step
        # bias_correction2 = 1 - beta2^step
        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step

        n = param.numel()
        if n == 0:
            continue

        # Define block size
        BLOCK_SIZE = triton.next_power_of_2(n)
        # Use at least 128 threads for better occupancy
        BLOCK_SIZE = max(BLOCK_SIZE, 128)
        # Cap at 4096 to avoid excessive registers
        BLOCK_SIZE = min(BLOCK_SIZE, 4096)

        # Calculate grid
        grid = (triton.cdiv(n, BLOCK_SIZE),)

        # Run the kernel - pass tensors directly
        _fused_adam_kernel[grid](
            param,
            grad,
            exp_avg,
            exp_avg_sq,
            max_exp_avg_sqs[i]
            if max_exp_avg_sqs
            else exp_avg_sq,  # dummy if no amsgrad
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


def _fused_adam_(
    self,
    grads,
    exp_avgs,
    exp_avg_sqs,
    max_exp_avg_sqs,
    state_steps,
    *,
    lr: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    weight_decay: float = 0.0,
    eps: float = 1e-8,
    amsgrad: bool = False,
    maximize: bool = False,
    grad_scale=None,
    found_inf=None,
):
    """In-place version of fused Adam."""
    _fused_adam(
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
