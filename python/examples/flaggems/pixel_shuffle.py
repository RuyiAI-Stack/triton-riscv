import torch
import triton
import triton.language as tl


@triton.jit
def pixel_shuffle_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    C,
    H,
    W,
    R,
    C_out,
    H_out,
    W_out,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    ow = offsets % W_out
    tmp = offsets // W_out
    oh = tmp % H_out
    tmp2 = tmp // H_out
    c_out = tmp2 % C_out
    n = tmp2 // C_out

    h_in = oh // R
    dh = oh % R
    w_in = ow // R
    dw = ow % R
    c_in = c_out * R * R + dh * R + dw

    in_idx = n * (C * H * W) + c_in * (H * W) + h_in * W + w_in

    val = tl.load(in_ptr + in_idx, mask=mask)
    tl.store(out_ptr + offsets, val, mask=mask)


def pixel_shuffle(input, upscale_factor):
    r = int(upscale_factor)
    assert input.ndim == 4
    N, C, H, W = input.shape
    assert C % (r * r) == 0
    C_out = C // (r * r)
    H_out = H * r
    W_out = W * r

    input = input.contiguous()
    output = torch.empty(
        (N, C_out, H_out, W_out), device=input.device, dtype=input.dtype
    )

    n_elements = output.numel()
    if n_elements == 0:
        return output

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    pixel_shuffle_kernel[grid](
        input,
        output,
        n_elements,
        C,
        H,
        W,
        r,
        C_out,
        H_out,
        W_out,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output
