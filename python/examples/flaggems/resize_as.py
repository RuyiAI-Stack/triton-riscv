from typing import Optional

import torch
import triton
import triton.language as tl

# resize_as: aten::resize_as -- out-of-place resize to the template's
# shape. Same shape returns self; a numel mismatch raises (autograd's
# resize preserves element count); otherwise the values are copied into a
# fresh tensor of the template shape. The in-place variant re-views the
# same storage via set_ (upstream behavior).


@triton.jit
def _resize_as_kernel(src_ptr, dst_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    vals = tl.load(src_ptr + offsets, mask=mask)
    tl.store(dst_ptr + offsets, vals, mask=mask)


def resize_as(
    self: torch.Tensor,
    the_template: torch.Tensor,
    *,
    memory_format: Optional[torch.memory_format] = None,
):
    """Fresh tensor of the template's shape holding self's elements (numel preserved)."""
    if memory_format is None:
        memory_format = torch.contiguous_format
    if self.shape == the_template.shape:
        return self
    if self.numel() != the_template.numel():
        raise RuntimeError(
            f"requested resize to {tuple(the_template.shape)} ({the_template.numel()} elements in total), "
            f"but the given tensor has a size of {tuple(self.shape)} ({self.numel()} elements in total). "
            f"autograd's resize can only change the shape of a given tensor, while preserving the number of elements."
        )
    out = torch.empty(the_template.size(), device=self.device, dtype=self.dtype)
    # Reshape source to destination shape (numel already verified equal).
    src = self.reshape(the_template.shape)
    if self.numel() == 0:
        return out
    grid = (triton.cdiv(self.numel(), 1024),)
    _resize_as_kernel[grid](src, out, self.numel(), BLOCK_SIZE=1024)
    return out


def resize_as_(
    self: torch.Tensor,
    the_template: torch.Tensor,
    *,
    memory_format: Optional[torch.memory_format] = None,
):
    """In-place resize to the template's shape via set_ (numel preserved)."""
    if memory_format is None:
        memory_format = torch.contiguous_format
    if self.shape == the_template.shape:
        return self
    if self.numel() != the_template.numel():
        raise RuntimeError(
            f"requested resize to {tuple(the_template.shape)} ({the_template.numel()} elements in total), "
            f"but the given tensor has a size of {tuple(self.shape)} ({self.numel()} elements in total). "
            f"autograd's resize can only change the shape of a given tensor, while preserving the number of elements."
        )
    self.set_(self.untyped_storage(), 0, the_template.shape)
    return self
