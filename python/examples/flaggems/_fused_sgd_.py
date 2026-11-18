import torch
import triton
import triton.language as tl


@triton.jit
def _fused_sgd_kernel(
    param,
    grad,
    momentum_buffer,
    n,
    lr,
    BLOCK_SIZE: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    out_dtype: tl.constexpr,
    weight_decay: tl.constexpr,
    momentum: tl.constexpr,
    dampening: tl.constexpr,
    nesterov: tl.constexpr,
    maximize: tl.constexpr,
    is_first_step: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    param_load = tl.load(param + offsets, mask=mask, other=0.0).to(ACC_DTYPE)
    grad_load = tl.load(grad + offsets, mask=mask, other=0.0).to(ACC_DTYPE)

    if maximize:
        grad_load = -grad_load
    if weight_decay != 0:
        grad_load = grad_load + param_load * weight_decay

    if momentum != 0:
        if is_first_step:
            buf = grad_load
        else:
            buf_load = tl.load(momentum_buffer + offsets, mask=mask, other=0.0).to(
                ACC_DTYPE
            )
            buf = momentum * buf_load + (1 - dampening) * grad_load
        tl.store(momentum_buffer + offsets, buf.to(out_dtype), mask=mask)
        if nesterov:
            update = grad_load + momentum * buf
        else:
            update = buf
    else:
        update = grad_load

    param_new = param_load - lr * update
    tl.store(param + offsets, param_new.to(out_dtype), mask=mask)


def _fused_sgd_(
    self,
    grads,
    momentum_buffer_list,
    *,
    weight_decay=0.0,
    momentum=0.0,
    lr=0.0,
    dampening=0.0,
    nesterov=False,
    maximize=False,
    is_first_step=False,
    grad_scale=None,
    found_inf=None,
):
    n_params = len(self)
    if len(grads) != n_params:
        raise RuntimeError("_fused_sgd_: grads length mismatch")
    if momentum != 0 and len(momentum_buffer_list) != n_params:
        raise RuntimeError("_fused_sgd_: momentum_buffer_list length mismatch")

    if found_inf is not None and found_inf.item() == 1:
        return None

    for i in range(n_params):
        param = self[i]
        grad = grads[i]
        buf = momentum_buffer_list[i] if momentum != 0 else param
        if grad_scale is not None:
            grad = grad / grad_scale
            grads[i].copy_(grad)
        if momentum != 0 and is_first_step:
            buf.copy_(grad)
            momentum_buffer_list[i] = buf

        acc_dtype = tl.float64 if param.dtype == torch.float64 else tl.float32
        out_dtype = tl.float64 if param.dtype == torch.float64 else tl.float32
        n = param.numel()
        BLOCK_SIZE = min(max(triton.next_power_of_2(n), 128), 4096)
        _fused_sgd_kernel[(triton.cdiv(n, BLOCK_SIZE),)](
            param,
            grads[i],
            momentum_buffer_list[i] if momentum != 0 else param,
            n,
            lr,
            BLOCK_SIZE,
            acc_dtype,
            out_dtype,
            weight_decay,
            momentum,
            dampening,
            nesterov,
            maximize,
            is_first_step,
        )
    return None
