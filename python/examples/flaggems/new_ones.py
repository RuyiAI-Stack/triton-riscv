import torch
import triton

from .ones import ones_kernel

# new_ones: aten::new_ones -- ones tensor of ``size`` inheriting self's
# dtype/device (kwargs override). The fill reuses the package's ones kernel
# (same pattern as upstream importing flag_gems.ops.ones).


def new_ones(self, size, *, dtype=None, layout=None, device=None, pin_memory=None):
    """Ones tensor of ``size`` defaulting to self's dtype and device."""
    if device is None:
        device = self.device
    if dtype is None:
        dtype = self.dtype

    out = torch.empty(
        size,
        device=device,
        dtype=dtype,
        layout=layout if layout is not None else torch.strided,
        pin_memory=pin_memory,
    )
    N = out.numel()
    if N > 0:
        grid = (triton.cdiv(N, 1024),)
        ones_kernel[grid](out, N, BLOCK_SIZE=1024)
    return out
