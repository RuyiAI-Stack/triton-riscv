import torch
import triton
import triton.language as tl


@triton.jit
def _add_relu_kernel(a_ptr, b_ptr, out_ptr, n_elements, alpha, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    res = a + alpha * b
    tl.store(out_ptr + offsets, tl.where(res > 0, res, 0.0), mask=mask)


def _add_relu_(A, B, *, alpha=1):
    """In-place relu(A + alpha * B); A is mutated and returned.

    Note: aten::_add_relu_ has no alpha parameter -- the coefficient is a
    FlagGems extension; the default alpha=1 is the aten behavior.
    """
    if not isinstance(A, torch.Tensor) or not isinstance(B, torch.Tensor):
        raise ValueError("_add_relu_ expects two tensors")
    if B.device != A.device:
        B = B.to(A.device)
    if not A.is_contiguous():
        raise AssertionError("_add_relu_ requires a contiguous tensor")
    b = B.to(A.dtype).expand(A.shape).contiguous()
    n = A.numel()
    if n:
        _add_relu_kernel[(triton.cdiv(n, 1024),)](A, b, A, n, float(alpha), BLOCK=1024)
    return A
