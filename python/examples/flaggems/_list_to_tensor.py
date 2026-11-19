import torch
import triton
import triton.language as tl


@triton.jit
def copy_kernel(src_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    src = tl.load(src_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, src, mask=mask)


def _list_to_tensor(self):
    """Triton implementation of aten::_list_to_tensor.

    Converts a List[int] into a 1-D int32 tensor: the host materializes the
    source with torch.tensor(self, dtype=int32) and the Triton copy kernel
    fills the output buffer. Empty lists return an empty int32 tensor without
    launching the kernel. Note: the upstream docstring calls this a JIT-only
    prim without a dispatcher kernel, but current torch builds expose
    torch.ops.aten._list_to_tensor (also int32), which the tests use as the
    bit-exact reference.
    """
    n = len(self)
    out = torch.empty((n,), dtype=torch.int32)
    if n == 0:
        return out

    src = torch.tensor(self, dtype=torch.int32)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    copy_kernel[grid](src, out, n, BLOCK_SIZE=BLOCK_SIZE)
    return out
