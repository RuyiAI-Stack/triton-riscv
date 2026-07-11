import torch
import triton
import triton.language as tl


@triton.jit
def masked_fill_kernel(
    inp_ptr,
    mask_ptr,
    out_ptr,
    n_elements,
    value,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(inp_ptr + offsets, mask=mask)
    m = tl.load(mask_ptr + offsets, mask=mask).to(tl.int1)
    x = tl.where(m == 1, value, x)
    tl.store(out_ptr + offsets, x, mask=mask)


def masked_fill(inp, mask, value):
    if torch.is_tensor(value):
        value = value.item()
    assert mask.shape == inp.shape or mask.numel() == inp.numel(), (
        "The shape of mask must be broadcastable with the shape of the underlying tensor"
    )

    if inp.ndim == 0:
        return (
            torch.tensor(value, dtype=inp.dtype, device=inp.device)
            if mask.item()
            else inp.clone()
        )

    expand_mask = mask.expand(inp.shape)
    inp_c = inp.contiguous()
    mask_c = expand_mask.to(torch.uint8).contiguous()
    out = torch.empty_like(inp_c)
    n_elements = inp_c.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    masked_fill_kernel[grid](
        inp_c, mask_c, out, n_elements, value, BLOCK_SIZE=BLOCK_SIZE
    )
    return out.view_as(inp)


def masked_fill_(inp, mask, value):
    if torch.is_tensor(value):
        value = value.item()
    assert mask.shape == inp.shape or mask.numel() == inp.numel(), (
        "The shape of mask must be broadcastable with the shape of the underlying tensor"
    )

    if inp.ndim == 0:
        if mask.item():
            inp[()] = value
        return inp

    expand_mask = mask.expand(inp.shape)
    mask_c = expand_mask.to(torch.uint8).contiguous()
    inp_c = inp.contiguous()
    n_elements = inp_c.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    masked_fill_kernel[grid](
        inp_c, mask_c, inp_c, n_elements, value, BLOCK_SIZE=BLOCK_SIZE
    )
    inp.copy_(inp_c)
    return inp
