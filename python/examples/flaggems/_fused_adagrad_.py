import torch
import triton
import triton.language as tl


@triton.jit
def _fused_adagrad_kernel(
    param,
    grad,
    state_sum,
    n,
    step,
    BLOCK_SIZE: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    lr: tl.constexpr,
    lr_decay: tl.constexpr,
    weight_decay: tl.constexpr,
    eps: tl.constexpr,
    maximize: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    param_load = tl.load(param + offsets, mask=mask, other=0.0)
    grad_load = tl.load(grad + offsets, mask=mask, other=0.0)
    state_sum_load = tl.load(state_sum + offsets, mask=mask, other=0.0)

    param_f = param_load.to(ACC_DTYPE)
    grad_f = grad_load.to(ACC_DTYPE)
    state_sum_f = state_sum_load.to(ACC_DTYPE)

    if maximize:
        grad_f = -grad_f
    if weight_decay > 0:
        grad_f = grad_f + param_f * weight_decay

    state_sum_f = state_sum_f + grad_f * grad_f
    corrected_lr = lr / (1.0 + (step - 1.0) * lr_decay)
    param_f = param_f - corrected_lr * grad_f / (tl.sqrt(state_sum_f) + eps)

    tl.store(param + offsets, param_f.to(param_load.dtype), mask=mask)
    tl.store(state_sum + offsets, state_sum_f.to(state_sum_load.dtype), mask=mask)


_SUPPORTED = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


def _fused_adagrad_(
    params,
    grads,
    state_sums,
    state_steps,
    *,
    lr=1e-2,
    lr_decay=0.0,
    weight_decay=0.0,
    eps=1e-10,
    maximize=False,
    grad_scale=None,
    found_inf=None,
):
    n_params = len(params)
    for name, seq in (("grads", grads), ("state_sums", state_sums), ("state_steps", state_steps)):
        if len(seq) != n_params:
            raise RuntimeError(
                f"_fused_adagrad_: expected {name} to have the same length as params ({n_params}), got {len(seq)}"
            )
    if n_params > 0:
        expected_dtype = params[0].dtype
        expected_device = params[0].device
        if expected_dtype not in _SUPPORTED:
            raise RuntimeError(
                "_fused_adagrad_ only supports float16/bfloat16/float32/float64 "
                f"inputs, got {expected_dtype} at index 0"
            )
        for i in range(n_params):
            for name, tensor in (
                ("param", params[i]),
                ("grad", grads[i]),
                ("state_sum", state_sums[i]),
            ):
                if tensor.dtype != expected_dtype:
                    raise RuntimeError(
                        "params, grads, and state_sums must have same dtype, device, and layout: "
                        f"{name}[{i}] dtype {tensor.dtype} differs from params[0] dtype {expected_dtype}"
                    )
                if tensor.device != expected_device:
                    raise RuntimeError(
                        "params, grads, and state_sums must have same dtype, device, and layout: "
                        f"{name}[{i}] device differs from params[0]"
                    )

    if found_inf is not None and found_inf.item() == 1:
        return None

    for i in range(n_params):
        param = params[i]
        grad = grads[i]
        state_sum = state_sums[i]
        if grad_scale is not None:
            grad = grad / grad_scale
        step = float(state_steps[i].item())
        acc_dtype = tl.float64 if param.dtype == torch.float64 else tl.float32
        n = param.numel()
        BLOCK_SIZE = min(max(triton.next_power_of_2(n), 128), 4096)
        _fused_adagrad_kernel[(triton.cdiv(n, BLOCK_SIZE),)](
            param,
            grad,
            state_sum,
            n,
            step,
            BLOCK_SIZE,
            acc_dtype,
            lr,
            lr_decay,
            weight_decay,
            eps,
            maximize,
        )
    return None
