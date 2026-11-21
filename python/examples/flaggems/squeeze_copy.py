import torch
import triton
import triton.language as tl

# squeeze_copy: copy of the input with all size-1 dimensions removed.
# The kernel does the flat element copy in Triton; non-contiguous inputs
# are gathered by .contiguous() first (upstream view_copy does the same).
# All-ones input squeezes to a 0-d tensor.


@triton.jit
def _squeeze_copy_kernel(src_ptr, dst_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    vals = tl.load(src_ptr + offsets, mask=mask)
    tl.store(dst_ptr + offsets, vals, mask=mask)


def squeeze_copy(x: torch.Tensor) -> torch.Tensor:
    """Copy of ``x`` with all size-1 dimensions removed (empty shape means 0-d)."""
    squeezed_shape = [s for s in x.shape if s != 1]
    out = torch.empty(squeezed_shape, dtype=x.dtype, device=x.device, layout=x.layout)
    n_elements = out.numel()
    if n_elements == 0:
        return out
    src = x.contiguous() if not x.is_contiguous() else x
    if not out.is_contiguous():
        out = out.contiguous()
    grid = (triton.cdiv(n_elements, 1024),)
    _squeeze_copy_kernel[grid](src, out, n_elements, BLOCK_SIZE=1024)
    return out
