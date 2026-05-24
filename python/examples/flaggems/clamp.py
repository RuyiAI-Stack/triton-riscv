import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["mini", "maxi"])
def clamp_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    mini,
    maxi,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    x_f32 = x.to(tl.float32)
    result = tl.minimum(maxi, tl.maximum(mini, x_f32))
    tl.store(out_ptr + offsets, result.to(x.dtype), mask=mask)


@triton.jit(do_not_specialize=["mini"])
def clamp_min_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    mini,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    x_f32 = x.to(tl.float32)
    result = tl.maximum(mini, x_f32)
    tl.store(out_ptr + offsets, result.to(x.dtype), mask=mask)


@triton.jit(do_not_specialize=["maxi"])
def clamp_max_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    maxi,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    x_f32 = x.to(tl.float32)
    result = tl.minimum(maxi, x_f32)
    tl.store(out_ptr + offsets, result.to(x.dtype), mask=mask)


# Tensor-boundary kernels (both min and max are tensors)
@triton.jit
def clamp_kernel_tensor(
    x_ptr,
    min_ptr,
    max_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    mini = tl.load(min_ptr + offsets, mask=mask)
    maxi = tl.load(max_ptr + offsets, mask=mask)
    x_f32 = x.to(tl.float32)
    mini_f32 = mini.to(tl.float32)
    maxi_f32 = maxi.to(tl.float32)
    result = tl.minimum(maxi_f32, tl.maximum(mini_f32, x_f32))
    tl.store(out_ptr + offsets, result.to(x.dtype), mask=mask)


@triton.jit
def clamp_min_kernel_tensor(
    x_ptr,
    min_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    mini = tl.load(min_ptr + offsets, mask=mask)
    x_f32 = x.to(tl.float32)
    mini_f32 = mini.to(tl.float32)
    result = tl.maximum(mini_f32, x_f32)
    tl.store(out_ptr + offsets, result.to(x.dtype), mask=mask)


@triton.jit
def clamp_max_kernel_tensor(
    x_ptr,
    max_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    maxi = tl.load(max_ptr + offsets, mask=mask)
    x_f32 = x.to(tl.float32)
    maxi_f32 = maxi.to(tl.float32)
    result = tl.minimum(maxi_f32, x_f32)
    tl.store(out_ptr + offsets, result.to(x.dtype), mask=mask)


def _clamp_launch(A, kernel, *args):
    A_c = A.contiguous()
    out = torch.empty_like(A_c)
    n_elements = A_c.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    kernel[grid](A_c, out, n_elements, *args, BLOCK_SIZE=BLOCK_SIZE)
    return out.view_as(A)


def _clamp_launch_tensor(A, min_tensor, max_tensor, kernel):
    A_c = A.contiguous()
    out = torch.empty_like(A_c)
    n_elements = A_c.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    min_c = (
        min_tensor.expand_as(A_c).contiguous()
        if min_tensor is not None
        else None
    )
    max_c = (
        max_tensor.expand_as(A_c).contiguous()
        if max_tensor is not None
        else None
    )
    if min_c is not None and max_c is not None:
        kernel[grid](A_c, min_c, max_c, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    elif min_c is not None:
        kernel[grid](A_c, min_c, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    elif max_c is not None:
        kernel[grid](A_c, max_c, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out.view_as(A)


def clamp(A, mini=None, maxi=None):
    if mini is None and maxi is None:
        raise ValueError("At least one of mini or maxi must not be None")
    if isinstance(mini, torch.Tensor) and isinstance(maxi, torch.Tensor):
        return _clamp_launch_tensor(A, mini, maxi, clamp_kernel_tensor)
    if isinstance(mini, torch.Tensor):
        return _clamp_launch_tensor(A, mini, None, clamp_min_kernel_tensor)
    if isinstance(maxi, torch.Tensor):
        return _clamp_launch_tensor(A, None, maxi, clamp_max_kernel_tensor)
    if maxi is None:
        return _clamp_launch(A, clamp_min_kernel, float(mini))
    if mini is None:
        return _clamp_launch(A, clamp_max_kernel, float(maxi))
    return _clamp_launch(A, clamp_kernel, float(mini), float(maxi))


def clamp_(A, mini=None, maxi=None):
    result = clamp(A, mini=mini, maxi=maxi)
    A.copy_(result)
    return A


def clamp_min(A, mini):
    if isinstance(mini, torch.Tensor):
        return _clamp_launch_tensor(A, mini, None, clamp_min_kernel_tensor)
    return _clamp_launch(A, clamp_min_kernel, float(mini))


def clamp_min_(A, mini):
    result = clamp_min(A, mini)
    A.copy_(result)
    return A


def clamp_max(A, maxi):
    if isinstance(maxi, torch.Tensor):
        return _clamp_launch_tensor(A, None, maxi, clamp_max_kernel_tensor)
    return _clamp_launch(A, clamp_max_kernel, float(maxi))


def clamp_tensor(A, mini=None, maxi=None):
    if mini is None and maxi is None:
        raise ValueError("At least one of mini or maxi must not be None")
    return clamp(A, mini=mini, maxi=maxi)


def clamp_tensor_(A, mini=None, maxi=None):
    result = clamp_tensor(A, mini=mini, maxi=maxi)
    A.copy_(result)
    return A
