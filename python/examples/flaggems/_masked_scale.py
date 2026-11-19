import torch
import triton
import triton.language as tl


@triton.jit
def _masked_scale_kernel(
    inp_ptr,
    mask_ptr,
    out_ptr,
    n_elements,
    scale,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(inp_ptr + offsets, mask=mask)
    m = tl.load(mask_ptr + offsets, mask=mask)
    # upstream formula: where(mask != 0, input * scale, 0.0). The multiply
    # runs in fp32 (scale is an fp32 arg) and narrows to the element dtype at
    # the store, matching the upstream pointwise_dynamic promotion behavior.
    y = tl.where(m != 0, x * scale, 0.0)
    tl.store(out_ptr + offsets, y.to(out_ptr.dtype.element_ty), mask=mask)


def _masked_scale(input, mask, scale):
    # out = where(mask != 0, input * scale, 0.0); floating-point input only,
    # output dtype == input dtype, mask broadcast to input's shape.
    if not input.is_floating_point():
        raise ValueError(f"Only floating-point dtype is supported, got {input.dtype}")
    if torch.is_tensor(scale):
        scale = scale.item()

    out = torch.empty_like(input)
    n_elements = input.numel()
    if n_elements == 0:
        return out

    inp_c = input.contiguous()
    mask_c = mask.expand(input.shape).contiguous()
    out_c = out if out.is_contiguous() else out.contiguous()

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _masked_scale_kernel[grid](
        inp_c, mask_c, out_c, n_elements, scale, BLOCK_SIZE=BLOCK_SIZE
    )

    if not out.is_contiguous():
        out.copy_(out_c)
    return out
