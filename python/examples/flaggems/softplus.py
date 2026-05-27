import torch
import triton
import triton.language as tl


@triton.jit
def softplus_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    beta,
    threshold,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    x_fp32 = x.to(tl.float32)
    z = x_fp32 * beta
    soft_z = tl.where(z > threshold, z, tl.log(1.0 + tl.exp(z)))
    y = (soft_z / beta).to(x.dtype)
    tl.store(out_ptr + offsets, y, mask=mask)


def softplus(self, beta=1.0, threshold=20.0):
    self_c = self.contiguous()
    out = torch.empty_like(self_c)
    n_elements = self_c.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    softplus_kernel[grid](
        self_c, out, n_elements, beta, threshold, BLOCK_SIZE=BLOCK_SIZE
    )
    return out.view_as(self)
