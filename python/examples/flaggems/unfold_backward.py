import math

import torch
import triton
import triton.language as tl


@triton.jit
def unfold_backward_output_kernel(
    grad_in_ptr,
    grad_out_ptr,
    prod_after,
    L,
    size,
    step,
    D,
):
    output_offset = tl.program_id(0)
    after = output_offset % prod_after
    before_and_position = output_offset // prod_after
    position = before_and_position % D
    before = before_and_position // D
    grad = tl.full((), 0.0, dtype=tl.float32)

    for window in range(0, L):
        within_window = position - window * step
        selected = (within_window >= 0) & (within_window < size)
        safe_within_window = tl.maximum(0, tl.minimum(size - 1, within_window))
        grad_offset = (
            ((before * L + window) * prod_after + after) * size
            + safe_within_window
        )
        value = tl.load(grad_in_ptr + grad_offset).to(tl.float32)
        grad += tl.where(selected, value, 0.0)

    tl.store(grad_out_ptr + output_offset, grad)


def unfold_backward(
    grad_in: torch.Tensor, input_sizes, dim: int, size: int, step: int
) -> torch.Tensor:
    if step <= 0:
        raise ValueError("step must be > 0")
    input_sizes = [int(value) for value in input_sizes]
    dim %= len(input_sizes)
    D = input_sizes[dim]
    L = (D - int(size)) // int(step) + 1
    prod_after = math.prod(input_sizes[dim + 1 :])
    output_f32 = torch.empty(
        input_sizes, dtype=torch.float32, device=grad_in.device
    )
    unfold_backward_output_kernel[(output_f32.numel(),)](
        grad_in.contiguous(),
        output_f32,
        prod_after,
        L,
        int(size),
        int(step),
        D,
        num_warps=1,
    )
    return output_f32 if grad_in.dtype == torch.float32 else output_f32.to(grad_in.dtype)
