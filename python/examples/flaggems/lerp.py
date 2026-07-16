import torch
import triton
import triton.language as tl


@triton.jit
def lerp_tensor_kernel(
    input_ptr,
    end_ptr,
    weight_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    inp = tl.load(input_ptr + offsets, mask=mask)
    end = tl.load(end_ptr + offsets, mask=mask)
    weight = tl.load(weight_ptr + offsets, mask=mask)
    res = tl.where(
        tl.abs(weight) < 0.5,
        inp + weight * (end - inp),
        end - (end - inp) * (1.0 - weight),
    )
    tl.store(out_ptr + offsets, res, mask=mask)


@triton.jit
def lerp_scalar_kernel(
    input_ptr,
    end_ptr,
    out_ptr,
    n_elements,
    weight_val,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    inp = tl.load(input_ptr + offsets, mask=mask)
    end = tl.load(end_ptr + offsets, mask=mask)
    res = inp + weight_val * (end - inp)
    tl.store(out_ptr + offsets, res, mask=mask)


@triton.jit
def lerp_scalar_kernel_alt(
    input_ptr,
    end_ptr,
    out_ptr,
    n_elements,
    weight_val,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    inp = tl.load(input_ptr + offsets, mask=mask)
    end = tl.load(end_ptr + offsets, mask=mask)
    res = end - (end - inp) * (1.0 - weight_val)
    tl.store(out_ptr + offsets, res, mask=mask)


def lerp_tensor(input, end, weight):
    A, B, W = torch.broadcast_tensors(input, end, weight)
    A_c = A.contiguous()
    B_c = B.contiguous()
    W_c = W.contiguous()
    out = torch.empty_like(A_c)
    n_elements = A_c.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    lerp_tensor_kernel[grid](A_c, B_c, W_c, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out.view_as(A)


def lerp_tensor_(input, end, weight):
    A, B, W = torch.broadcast_tensors(input, end, weight)
    A_c = A.contiguous()
    B_c = B.contiguous()
    W_c = W.contiguous()
    n_elements = A_c.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    lerp_tensor_kernel[grid](A_c, B_c, W_c, A_c, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    input.copy_(A_c)
    return input


def lerp_scalar(input, end, weight):
    A, B = torch.broadcast_tensors(input, end)
    A_c = A.contiguous()
    B_c = B.contiguous()
    out = torch.empty_like(A_c)
    n_elements = A_c.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    if weight < 0.5:
        lerp_scalar_kernel[grid](
            A_c, B_c, out, n_elements, weight, BLOCK_SIZE=BLOCK_SIZE
        )
    else:
        lerp_scalar_kernel_alt[grid](
            A_c, B_c, out, n_elements, weight, BLOCK_SIZE=BLOCK_SIZE
        )
    return out.view_as(A)


def lerp_scalar_(input, end, weight):
    A, B = torch.broadcast_tensors(input, end)
    A_c = A.contiguous()
    B_c = B.contiguous()
    n_elements = A_c.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    if weight < 0.5:
        lerp_scalar_kernel[grid](
            A_c, B_c, A_c, n_elements, weight, BLOCK_SIZE=BLOCK_SIZE
        )
    else:
        lerp_scalar_kernel_alt[grid](
            A_c, B_c, A_c, n_elements, weight, BLOCK_SIZE=BLOCK_SIZE
        )
    input.copy_(A_c)
    return input


def lerp(input, end, weight):
    if isinstance(weight, torch.Tensor):
        return lerp_tensor(input, end, weight)
    return lerp_scalar(input, end, weight)
