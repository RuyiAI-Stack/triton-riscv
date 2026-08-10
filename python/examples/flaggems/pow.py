import torch
import triton
import triton.language as tl


@triton.jit
def pow_kernel_tt(
    x_ptr,
    exponent_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    exponent = tl.load(exponent_ptr + offsets, mask=mask).to(tl.float32)
    log_x = tl.where(x > 0, tl.log(x), 0.0)
    out = tl.exp(exponent * log_x)
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def pow_kernel_ts(
    x_ptr,
    exponent_val,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    exponent = exponent_val.to(tl.float32)
    log_x = tl.where(x > 0, tl.log(x), 0.0)
    out = tl.exp(exponent * log_x)
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def pow_kernel_st(
    x_val,
    exponent_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = x_val.to(tl.float32)
    exponent = tl.load(exponent_ptr + offsets, mask=mask).to(tl.float32)
    log_x = tl.where(x > 0, tl.log(x), 0.0)
    out = tl.exp(exponent * log_x)
    tl.store(out_ptr + offsets, out, mask=mask)


def pow_tensor_tensor(A, exponent):
    if exponent.device != A.device:
        exponent = exponent.to(A.device)
    A, exponent = torch.broadcast_tensors(A, exponent)
    A_c = A.contiguous()
    exponent_c = exponent.contiguous()
    out = torch.empty_like(A_c, dtype=torch.float32)
    n_elements = A_c.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    pow_kernel_tt[grid](A_c, exponent_c, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out.view_as(A).to(torch.result_type(A, exponent))


def pow_tensor_scalar(A, exponent):
    A_c = A.contiguous()
    out = torch.empty_like(A_c, dtype=torch.float32)
    n_elements = A_c.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    pow_kernel_ts[grid](A_c, exponent, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out.view_as(A).to(torch.result_type(A, exponent))


def pow_scalar(A, exponent):
    exponent_c = exponent.contiguous()
    out = torch.empty_like(exponent_c, dtype=torch.float32)
    n_elements = exponent_c.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    pow_kernel_st[grid](A, exponent_c, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out.view_as(exponent).to(torch.result_type(A, exponent))


def pow_tensor_tensor_(A, exponent):
    result = pow_tensor_tensor(A, exponent)
    A.copy_(result)
    return A


def pow_tensor_scalar_(A, exponent):
    result = pow_tensor_scalar(A, exponent)
    A.copy_(result)
    return A


def pow(A, exponent):
    if isinstance(A, torch.Tensor) and isinstance(exponent, torch.Tensor):
        return pow_tensor_tensor(A, exponent)
    elif isinstance(A, torch.Tensor):
        return pow_tensor_scalar(A, exponent)
    else:
        return pow_scalar(A, exponent)
