import torch
import triton
import triton.language as tl


@triton.jit
def threshold_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    threshold,
    value,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, tl.where(x > threshold, x, value), mask=mask)


@triton.jit
def threshold_backward_kernel(
    grad_output_ptr,
    self_ptr,
    out_ptr,
    n_elements,
    threshold,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    gy = tl.load(grad_output_ptr + offsets, mask=mask)
    x = tl.load(self_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, tl.where(x > threshold, gy, 0), mask=mask)


def threshold(self, threshold, value):
    x = self.contiguous() if not self.is_contiguous() else self
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    threshold_kernel[grid](
        x, out, n_elements, threshold, value, BLOCK_SIZE=BLOCK_SIZE
    )
    return out.view_as(self)


def threshold_backward(grad_output, self, threshold):
    x = self.contiguous() if not self.is_contiguous() else self
    gy = (
        grad_output.contiguous()
        if not grad_output.is_contiguous()
        else grad_output
    )
    gx = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    threshold_backward_kernel[grid](
        gy, x, gx, n_elements, threshold, BLOCK_SIZE=BLOCK_SIZE
    )
    return gx.view_as(self)
