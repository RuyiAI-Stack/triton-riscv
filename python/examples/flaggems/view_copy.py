import torch
import triton
import triton.language as tl

# view_copy: aten::view as a value copy. Exactly one -1 dimension is
# inferred (more than one raises); a numel mismatch raises; empty inputs
# return an empty tensor of the target shape; non-contiguous inputs are
# gathered with .contiguous() before the flat Triton copy (upstream order).


@triton.jit
def _view_copy_kernel(src_ptr, dst_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    vals = tl.load(src_ptr + offsets, mask=mask)
    tl.store(dst_ptr + offsets, vals, mask=mask)


def view_copy(x: torch.Tensor, size) -> torch.Tensor:
    """Copy of ``x`` reshaped to ``size`` (one -1 may be used for inference)."""
    if isinstance(size, torch.SymInt):
        size = (int(size),)
    elif isinstance(size, (list, tuple)):
        size = tuple(int(s) if isinstance(s, torch.SymInt) else s for s in size)

    n_elements = x.numel()

    if -1 in size:
        if size.count(-1) > 1:
            raise RuntimeError(f"view_copy: only one dimension can be -1, got {size}")
        target_numel_except_minus1 = 1
        for s in size:
            if s != -1:
                target_numel_except_minus1 *= s
        inferred_dim = n_elements // target_numel_except_minus1
        size = tuple(inferred_dim if s == -1 else s for s in size)

    target_numel = 1
    for s in size:
        target_numel *= s
    if n_elements != target_numel:
        raise RuntimeError(
            f"view_copy: cannot reshape tensor of size {n_elements} into shape {size}"
        )

    if n_elements == 0:
        return torch.empty(size, dtype=x.dtype, device=x.device)

    out = torch.empty(size, dtype=x.dtype, device=x.device)

    src = x.contiguous() if not x.is_contiguous() else x
    if not out.is_contiguous():
        out = out.contiguous()

    grid = (triton.cdiv(n_elements, 1024),)
    _view_copy_kernel[grid](src, out, n_elements, BLOCK_SIZE=1024)
    return out
