import torch
import triton
import triton.language as tl


@triton.jit
def rrelu_with_noise_backward_kernel(
    grad_out_ptr,
    input_ptr,
    noise_ptr,
    grad_in_ptr,
    n_elements,
    lower,
    upper,
    training,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    go = tl.load(grad_out_ptr + offsets, mask=mask, other=0)
    x = tl.load(input_ptr + offsets, mask=mask, other=0)
    nz = tl.load(noise_ptr + offsets, mask=mask, other=0)

    go_f32 = go.to(tl.float32)
    x_f32 = x.to(tl.float32)
    nz_f32 = nz.to(tl.float32)

    slope = (lower + upper) * 0.5

    grad_train = go_f32 * nz_f32
    grad_eval = go_f32 * tl.where(x_f32 > 0, 1.0, slope)

    cond = tl.full(go_f32.shape, training, tl.int1)
    grad_f32 = tl.where(cond, grad_train, grad_eval)

    grad_cast = grad_f32.to(go.dtype)
    tl.store(grad_in_ptr + offsets, grad_cast, mask=mask)


def rrelu_with_noise_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    noise: torch.Tensor,
    lower: float,
    upper: float,
    training: bool,
):
    go = grad_output.contiguous()
    x = input.contiguous()
    nz = noise.contiguous()

    out = torch.empty_like(go)
    out_t = out.contiguous()
    n_elements = out.numel()

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    rrelu_with_noise_backward_kernel[grid](
        go,
        x,
        nz,
        out,
        n_elements,
        float(lower),
        float(upper),
        1 if training else 0,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    if out is not out_t:
        out.copy_(out_t)
    return out
