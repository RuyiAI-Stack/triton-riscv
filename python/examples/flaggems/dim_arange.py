import torch
import triton
import triton.language as tl

# dim_arange: aten::_dim_arange -- a 1-D int64 ``[0, ..., size-1]`` where
# size is ``self.shape[dim]`` (tensor values and strides are irrelevant;
# negative dim is normalized). The exact path drops the bounds mask when n
# is a power of two >= 256.


@triton.jit
def _dim_arange_kernel(out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < n_elements
    tl.store(out_ptr + offs, offs, mask=mask)


@triton.jit
def _dim_arange_kernel_exact(out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    tl.store(out_ptr + offs, offs)


def _is_pow2(x):
    return x > 0 and (x & (x - 1)) == 0


def _launch_dim_arange(n, like):
    out = torch.empty(n, dtype=torch.int64, device=like.device)
    if n == 0:
        return out
    # Exact path requires n to be an exact multiple of BLOCK; n is a power
    # of two >= 256 here, so no bounds mask is needed.
    if _is_pow2(n) and n >= 256:
        if n <= 1024:
            block = 256
        elif n < 131072:
            block = 1024
        else:
            block = 2048
        _dim_arange_kernel_exact[(n // block,)](out, BLOCK=block)
    else:
        block = 1024
        grid = (triton.cdiv(n, block),)
        _dim_arange_kernel[grid](out, n, BLOCK=block)
    return out


def _dim_arange(self: torch.Tensor, dim: int) -> torch.Tensor:
    """1-D int64 ``[0, 1, ..., self.shape[dim] - 1]``; only the dim size is used."""
    ndim = self.dim()
    if dim < 0:
        dim += ndim
    return _launch_dim_arange(self.shape[dim], self)
