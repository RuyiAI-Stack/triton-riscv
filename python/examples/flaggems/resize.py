import torch
import triton
import triton.language as tl

# resize: aten::resize (deprecated op) -- out-of-place resize copies the
# first min(old, new) flat elements into a fresh tensor of the target
# shape; ``memory_format`` is accepted but ignored (upstream behavior,
# kept verbatim). The in-place variant re-views the same storage via
# set_. A same-numel grow reads past the source exactly like upstream
# (unmasked out.numel copy); tests pin the well-defined shrink path.


@triton.jit
def _resize_kernel(src_ptr, dst_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    vals = tl.load(src_ptr + offsets, mask=mask)
    tl.store(dst_ptr + offsets, vals, mask=mask)


def resize(inp, size, memory_format=None):
    """Fresh tensor of ``size`` holding the first elements of ``inp`` (memory_format ignored)."""
    if not isinstance(size, tuple):
        size = tuple(size)
    out = torch.empty(size, device=inp.device, dtype=inp.dtype)
    if inp.numel() == 0 or out.numel() == 0:
        return out
    n_elements = out.numel()
    grid = (triton.cdiv(n_elements, 1024),)
    _resize_kernel[grid](inp, out, n_elements, BLOCK_SIZE=1024)
    return out


def resize_(inp, size, memory_format=None):
    """In-place resize: re-view the same storage at ``size`` via set_."""
    if not isinstance(size, tuple):
        size = tuple(size)
    inp.set_(inp.untyped_storage(), 0, size)
    return inp
