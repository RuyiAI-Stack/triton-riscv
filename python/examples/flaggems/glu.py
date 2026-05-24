import torch
import triton
import triton.language as tl


@triton.jit
def glu_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    sigmoid_b = 1 / (1 + tl.exp(-b.to(tl.float32)))
    result = a * sigmoid_b
    tl.store(out_ptr + offsets, result, mask=mask)


@triton.jit
def glu_backward_kernel(
    grad_output_ptr,
    a_ptr,
    b_ptr,
    da_ptr,
    db_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    grad_output = tl.load(grad_output_ptr + offsets, mask=mask)
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    sigmoid_b = 1 / (1 + tl.exp(-b.to(tl.float32)))
    da = grad_output * sigmoid_b
    db = grad_output.to(tl.float32) * a * sigmoid_b * (1.0 - sigmoid_b)
    tl.store(da_ptr + offsets, da, mask=mask)
    tl.store(db_ptr + offsets, db, mask=mask)


def glu(self, dim=-1):
    assert self.shape[dim] % 2 == 0, "Split dimension must be even"
    a, b = torch.chunk(self, 2, dim=dim)
    a_c = a.contiguous()
    b_c = b.contiguous()
    out = torch.empty_like(a_c)
    n_elements = a_c.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    glu_kernel[grid](a_c, b_c, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


def glu_backward(grad_output, self, dim=-1):
    assert self.shape[dim] % 2 == 0, "Split dimension must be even"
    a, b = torch.chunk(self, 2, dim=dim)
    grad_output_c = grad_output.contiguous()
    a_c = a.contiguous()
    b_c = b.contiguous()
    da = torch.empty_like(a_c)
    db = torch.empty_like(b_c)
    n_elements = a_c.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    glu_backward_kernel[grid](
        grad_output_c, a_c, b_c, da, db, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    grad_input = torch.cat([da, db], dim=dim)
    return grad_input
