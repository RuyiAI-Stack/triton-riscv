import torch
import triton
import triton.language as tl


@triton.jit
def _functional_sym_constrain_range_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(y_ptr + offsets, x, mask=mask)


def _functional_sym_constrain_range(*args, **kwargs):
    """Identity copy of the dependency token.

    The symbolic range constraint is recorded by torch's symbolic tracing,
    not at runtime; like aten (whose CPU kernel returns dep_token.clone()),
    the runtime behavior is a value-preserving copy of the first tensor
    argument: contiguous non-empty tensors go through the Triton copy kernel,
    everything else falls back to a host clone.
    """
    tensor_arg = None
    for a in args:
        if isinstance(a, torch.Tensor):
            tensor_arg = a
            break
    if tensor_arg is None:
        for v in kwargs.values():
            if isinstance(v, torch.Tensor):
                tensor_arg = v
                break

    if tensor_arg is not None:
        if tensor_arg.is_contiguous() and tensor_arg.numel() > 0:
            n_elements = tensor_arg.numel()
            output = torch.empty_like(tensor_arg)
            BLOCK_SIZE = 1024
            grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
            _functional_sym_constrain_range_kernel[grid](
                tensor_arg, output, n_elements, BLOCK_SIZE=BLOCK_SIZE
            )
            return output
        # Fallback for non-contiguous or empty tensors
        return tensor_arg.clone()

    # Fallback: return args[0] if no tensor was found
    return args[0] if len(args) > 0 else None
