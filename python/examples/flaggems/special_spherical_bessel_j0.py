import torch
import triton
import triton.language as tl

# Spherical Bessel j0(x) = sin(x)/x with the removable singularity at 0
# pinned to 1.  ATen also pins the infinities to 0 (probed on torch
# CPU) even though sin(inf) alone would be nan, so the port adds the
# explicit row.


@triton.jit
def spherical_bessel_j0_func(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    ax = tl.abs(x)

    is_small = ax < 1e-6
    safe_x = tl.where(is_small, 1.0, x)
    res = tl.sin(x) / safe_x
    res = tl.where(is_small, 1.0, res)
    res = tl.where(ax == float("inf"), 0.0, res)
    res = tl.where(x != x, x, res)

    tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


def special_spherical_bessel_j0(A):
    A_c = A.contiguous()
    out = torch.empty_like(A_c)
    n_elements = A_c.numel()
    if n_elements == 0:
        return out
    grid = (triton.cdiv(n_elements, 1024),)
    spherical_bessel_j0_func[grid](A_c, out, n_elements, BLOCK_SIZE=1024)
    return out
