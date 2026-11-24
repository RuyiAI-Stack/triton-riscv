import torch
import triton
import triton.language as tl


@triton.jit
def _prelu_kernel_backward_kernel(
    grad_output_ptr,
    x_ptr,
    weight_ptr,
    grad_input_ptr,
    grad_weight_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    grad_output = tl.load(grad_output_ptr + offsets, mask=mask)
    x = tl.load(x_ptr + offsets, mask=mask)

    weight = tl.load(weight_ptr + offsets, mask=mask)

    # grad_input = grad_output if x > 0 else grad_output * weight
    grad_input = tl.where(x > 0, grad_output, grad_output * weight)

    # grad_weight = grad_output * x if x < 0 else 0
    grad_weight = tl.where(x < 0, grad_output * x, 0.0)

    tl.store(grad_input_ptr + offsets, grad_input, mask=mask)
    tl.store(grad_weight_ptr + offsets, grad_weight, mask=mask)


def _prelu_kernel_backward(*args, **kwargs):

    # Extract inputs
    if len(args) >= 3:
        grad_output, x, weight = args[0], args[1], args[2]
    else:
        grad_output = kwargs.get("grad_output")
        x = kwargs.get("self")
        weight = kwargs.get("weight")

    if grad_output is None or x is None or weight is None:
        raise ValueError(
            "_prelu_kernel_backward expects (grad_output, self, weight) as arguments."
        )

    if grad_output.device != x.device or weight.device != x.device:
        raise RuntimeError("_prelu_kernel_backward: all tensors must share a device")

    # Ensure dtype match
    if weight.dtype != x.dtype:
        weight = weight.to(dtype=x.dtype)
    if grad_output.dtype != x.dtype:
        grad_output = grad_output.to(dtype=x.dtype)

    # ATen returns elementwise gradients over the broadcast shape.
    grad_output, x, weight = torch.broadcast_tensors(grad_output, x, weight)

    # Ensure contiguous
    grad_output = grad_output.contiguous()
    x = x.contiguous()
    weight = weight.contiguous()

    grad_input = torch.empty_like(x)
    grad_weight = torch.empty_like(x)

    n_elements = x.numel()
    if n_elements == 0:
        return grad_input, grad_weight

    # BLOCK_SIZE of 1024 balances occupancy and parallelism for typical element-wise backward kernels
    BLOCK_SIZE = 1024

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    _prelu_kernel_backward_kernel[grid](
        grad_output,
        x,
        weight,
        grad_input,
        grad_weight,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return grad_input, grad_weight
