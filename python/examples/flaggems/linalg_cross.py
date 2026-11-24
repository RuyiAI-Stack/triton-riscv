import torch
import triton
import triton.language as tl


@triton.jit
def _cross_kernel(a_ptr, b_ptr, out_ptr, nvec, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= nvec:
        return
    a0 = tl.load(a_ptr + pid * 3)
    a1 = tl.load(a_ptr + pid * 3 + 1)
    a2 = tl.load(a_ptr + pid * 3 + 2)
    b0 = tl.load(b_ptr + pid * 3)
    b1 = tl.load(b_ptr + pid * 3 + 1)
    b2 = tl.load(b_ptr + pid * 3 + 2)
    tl.store(out_ptr + pid * 3, a1 * b2 - a2 * b1)
    tl.store(out_ptr + pid * 3 + 1, a2 * b0 - a0 * b2)
    tl.store(out_ptr + pid * 3 + 2, a0 * b1 - a1 * b0)
    _ = BLOCK


def linalg_cross(input, other, dim=-1):
    if dim < 0:
        dim += input.ndim
    if input.shape[dim] != 3:
        raise RuntimeError("linalg_cross: dimension size must be 3")
    a = input.transpose(dim, -1).contiguous()
    b = other.transpose(dim, -1).contiguous()
    out = torch.empty_like(a)
    nvec = a.numel() // 3
    if nvec:
        _cross_kernel[(nvec,)](a, b, out, nvec, BLOCK=1)
    return out.transpose(dim, -1).contiguous()


__all__ = ["linalg_cross"]
