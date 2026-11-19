import torch
import triton
import triton.language as tl

_INT_DTYPES = (
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
)


@triton.jit
def rshift_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x >> y, mask=mask)


def _rshift_inputs(self, other):
    # torch promotion: tensor/tensor widens (int8 >> int16 -> int16),
    # python scalars are weak (int32 tensor >> 2 -> int32).
    out_dtype = torch.result_type(self, other)
    if out_dtype not in _INT_DTYPES:
        raise TypeError(
            "__rshift__ only supports integer dtypes "
            f"(uint8/int8/int16/int32/int64), got {out_dtype}"
        )
    if not torch.is_tensor(other):
        other = torch.full((), other, dtype=out_dtype, device=self.device)
    A = self.to(out_dtype)
    B = other.to(out_dtype)
    A, B = torch.broadcast_tensors(A, B)
    return A.contiguous(), B.contiguous(), out_dtype


def __rshift__(self: torch.Tensor, other, *, out=None) -> torch.Tensor:
    """Elementwise arithmetic right shift: self >> other."""
    A_c, B_c, out_dtype = _rshift_inputs(self, other)
    result = torch.empty(A_c.shape, dtype=out_dtype, device=A_c.device)

    n_elements = result.numel()
    if n_elements == 0:
        if out is None:
            return result
        out.copy_(result)
        return out

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    rshift_kernel[grid](A_c, B_c, result, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    if out is None:
        return result.view_as(A_c)
    out.copy_(result.view_as(out))
    return out


def __rshift___out(self: torch.Tensor, other, *, out) -> torch.Tensor:
    """Elementwise arithmetic right shift with mandatory out tensor."""
    return __rshift__(self, other, out=out)
