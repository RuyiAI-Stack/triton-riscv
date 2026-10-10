import torch
import triton
import triton.language as tl


@triton.jit
def _amp_foreach_non_finite_check_and_unscale_kernel(
    tensor_ptr,
    inv_scale_ptr,
    found_inf_ptr,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    inp = tl.load(tensor_ptr + offsets, mask=mask, other=0.0)
    if tensor_ptr.dtype.element_ty == tl.float64:
        compute_dtype = tl.float64
    else:
        compute_dtype = tl.float32
    inp = inp.to(compute_dtype)
    scale = tl.load(inv_scale_ptr).to(compute_dtype)

    is_non_finite = (inp != inp) | (tl.abs(inp) == float("inf"))
    non_finite_count = tl.sum(tl.where(is_non_finite, 1.0, 0.0), axis=0)
    tl.atomic_add(found_inf_ptr, non_finite_count)

    scaled = tl.where(is_non_finite, inp, inp * scale)
    tl.store(tensor_ptr + offsets, scaled.to(tensor_ptr.dtype.element_ty), mask=mask)


def _amp_foreach_non_finite_check_and_unscale_(
    tensors: list[torch.Tensor],
    found_inf: torch.Tensor,
    inv_scale: torch.Tensor,
):
    if not isinstance(tensors, (list, tuple)):
        raise TypeError(f"Expected list or tuple of tensors, got {type(tensors)}")
    if len(tensors) == 0:
        return

    inv_scale_fp32 = inv_scale.to(dtype=torch.float32)
    local_found_inf = torch.zeros((), dtype=torch.float32, device=found_inf.device)
    BLOCK_SIZE = 1024

    for tensor in tensors:
        if not tensor.is_floating_point():
            raise NotImplementedError(
                "_amp_foreach_non_finite_check_and_unscale_ only supports floating tensors"
            )
        if tensor.numel() == 0:
            continue

        working = tensor if tensor.is_contiguous() else tensor.contiguous()
        grid = (triton.cdiv(working.numel(), BLOCK_SIZE),)
        _amp_foreach_non_finite_check_and_unscale_kernel[grid](
            working,
            inv_scale_fp32,
            local_found_inf,
            working.numel(),
            BLOCK_SIZE=BLOCK_SIZE,
        )
        if working is not tensor:
            tensor.copy_(working)

    if local_found_inf.item() != 0.0:
        found_inf.fill_(1.0)
