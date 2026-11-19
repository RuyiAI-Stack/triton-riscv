import torch
import triton
import triton.language as tl

# special_xlogy shares the xlogy port's ATen precedence (probed on torch
# CPU fp32): NaN when y is NaN -- even for x == 0; then 0 when x == +-0
# for every other y (including -inf); otherwise x * log(y) in plain IEEE
# arithmetic.  The order of the two overrides carries the precedence.


@triton.jit
def special_xlogy_func(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    y = tl.load(y_ptr + offsets, mask=mask).to(tl.float32)

    res = x * tl.log(y)
    res = tl.where(x == 0.0, 0.0, res)
    res = tl.where(y != y, float("nan"), res)

    tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


def _run(A, B, out=None):
    if B.device != A.device:
        B = B.to(A.device)
    A, B = torch.broadcast_tensors(A, B)
    A_c = A.contiguous()
    B_c = B.contiguous()
    if out is None:
        out = torch.empty_like(A_c)
        out_c = out
    else:
        out_c = out if out.is_contiguous() else out.contiguous()
    n_elements = A_c.numel()
    if n_elements == 0:
        if out_c.data_ptr() != out.data_ptr():
            out.copy_(out_c)
        return out
    grid = (triton.cdiv(n_elements, 1024),)
    special_xlogy_func[grid](A_c, B_c, out_c, n_elements, BLOCK_SIZE=1024)
    if out_c.data_ptr() != out.data_ptr():
        out.copy_(out_c)
    return out


def special_xlogy(A, B):
    return _run(A, B)


def special_xlogy_(A, B):
    res = _run(A, B)
    A.copy_(res)
    return A
