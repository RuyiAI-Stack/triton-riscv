import torch
import triton
import triton.language as tl


@triton.jit
def addcdiv_kernel(
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

    res = x + value * (t1 / t2)
    tl.store(out_ptr + offsets, res, mask=mask)


def addcdiv_(inp, tensor1, tensor2, *, value=1.0):
    b_inp, b_t1, b_t2 = torch.broadcast_tensors(inp, tensor1, tensor2)
    if b_inp.shape != inp.shape:
        raise RuntimeError(
            f"output with shape {tuple(inp.shape)} doesn't match the broadcast shape {tuple(b_inp.shape)}"
        )

    c_inp = inp.contiguous()
    c_t1 = b_t1.contiguous()
    c_t2 = b_t2.contiguous()

    n_elements = c_inp.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    addcdiv_kernel[grid](
        c_inp, c_t1, c_t2, c_inp, n_elements, value, BLOCK_SIZE=BLOCK_SIZE
    )
    if c_inp is not inp:
        inp.copy_(c_inp)
    return inp
