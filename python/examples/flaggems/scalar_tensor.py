import torch
import triton
import triton.language as tl

# scalar_tensor: aten::scalar_tensor -- a 0-d tensor holding the value,
# written by a one-program Triton kernel. A bool dtype converts the value
# through bool() first (upstream behavior).


@triton.jit
def scalar_tensor_kernel(out, value_scalar):
    tl.store(out, value_scalar)


def scalar_tensor(s, *, dtype=None, layout=None, device=None, pin_memory=None):
    """0-d tensor holding ``s`` (bool inputs convert through bool(), as upstream)."""
    out = torch.empty(
        (), dtype=dtype, layout=layout, device=device, pin_memory=pin_memory
    )
    if dtype == torch.bool:
        s = bool(s)
    scalar_tensor_kernel[(1,)](out, s)
    return out
