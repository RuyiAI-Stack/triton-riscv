import torch
import triton
import triton.language as tl


@triton.jit
def addcmul_kernel(
    x_ptr,
    t1_ptr,
    t2_ptr,
    out_ptr,
    n_elements,
    value,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    t1 = tl.load(t1_ptr + offsets, mask=mask)
    t2 = tl.load(t2_ptr + offsets, mask=mask)

    res = x + value * t1 * t2
    tl.store(out_ptr + offsets, res, mask=mask)


def addcmul(inp, tensor1, tensor2, *, value=1.0):
    b_inp, b_t1, b_t2 = torch.broadcast_tensors(inp, tensor1, tensor2)
    c_inp = b_inp.contiguous()
    c_t1 = b_t1.contiguous()
    c_t2 = b_t2.contiguous()

    out = torch.empty_like(c_inp)

    n_elements = c_inp.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    addcmul_kernel[grid](
        c_inp, c_t1, c_t2, out, n_elements, value, BLOCK_SIZE=BLOCK_SIZE
    )
    return out.view_as(inp)


def addcmul_out(inp, tensor1, tensor2, *, value=1.0, out):
    b_inp, b_t1, b_t2 = torch.broadcast_tensors(inp, tensor1, tensor2)
    c_inp = b_inp.contiguous()
    c_t1 = b_t1.contiguous()
    c_t2 = b_t2.contiguous()

    if not out.is_contiguous():
        out = out.contiguous()

    n_elements = c_inp.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    addcmul_kernel[grid](
        c_inp, c_t1, c_t2, out, n_elements, value, BLOCK_SIZE=BLOCK_SIZE
    )
    return out
