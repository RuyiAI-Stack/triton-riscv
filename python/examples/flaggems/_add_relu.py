import torch
import triton
import triton.language as tl


@triton.jit
def _add_relu_kernel(a_ptr, b_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    res = a + b
    tl.store(out_ptr + offsets, tl.where(res > 0, res, 0.0), mask=mask)


def _add_relu(A, B):
    """relu(A + B) with tensor/scalar operands (aten::_add_relu)."""
    if not isinstance(A, torch.Tensor) and not isinstance(B, torch.Tensor):
        return torch.tensor(max(0, A + B))
    ref = A if isinstance(A, torch.Tensor) else B
    dtype, device = ref.dtype, ref.device
    a = (
        A
        if isinstance(A, torch.Tensor)
        else torch.tensor(A, dtype=dtype, device=device)
    )
    b = (
        B
        if isinstance(B, torch.Tensor)
        else torch.tensor(B, dtype=dtype, device=device)
    )
    shape = torch.broadcast_shapes(a.shape, b.shape)
    a = a.expand(shape).contiguous()
    b = b.expand(shape).contiguous()
    out = torch.empty(shape, dtype=dtype, device=device)
    n = out.numel()
    if n:
        _add_relu_kernel[(triton.cdiv(n, 1024),)](a, b, out, n, BLOCK=1024)
    return out
