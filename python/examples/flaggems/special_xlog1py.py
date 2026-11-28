import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# xlog1py semantics (probed on torch CPU fp32): x * log1p(y) in plain
# IEEE arithmetic, with x == +-0 mapped to 0 whenever y <= -1 (where the
# log term is -inf or nan); a NaN y stays NaN.  libdevice.log1p lowers
# here but cancels for tiny y (behaves as log(1 + y)), so |y| < 0.01
# takes the same 3-term Taylor branch as the log1p port.


@triton.jit
def xlog1py_func(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    y = tl.load(y_ptr + offsets, mask=mask).to(tl.float32)

    taylor = y - y * y * 0.5 + y * y * y * (1.0 / 3.0)
    l1p = tl.where(tl.abs(y) < 0.01, taylor, libdevice.log1p(y))
    res = x * l1p
    res = tl.where((x == 0.0) & (y <= -1.0), 0.0, res)
    res = tl.where(y != y, float("nan"), res)

    tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


def special_xlog1py(A, B):
    if B.device != A.device:
        B = B.to(A.device)
    A, B = torch.broadcast_tensors(A, B)
    A_c = A.contiguous()
    B_c = B.contiguous()
    out = torch.empty_like(A_c)
    n_elements = A_c.numel()
    if n_elements == 0:
        return out
    grid = (triton.cdiv(n_elements, 1024),)
    xlog1py_func[grid](A_c, B_c, out, n_elements, BLOCK_SIZE=1024)
    return out
