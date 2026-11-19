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
def irshift_kernel(
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


def __irshift__(self: torch.Tensor, other) -> torch.Tensor:
    """In-place elementwise arithmetic right shift: self >>= other."""
    if self.dtype not in _INT_DTYPES:
        raise TypeError(
            "__irshift__ only supports integer dtypes "
            f"(uint8/int8/int16/int32/int64), got {self.dtype}"
        )
    if not torch.is_tensor(other):
        other = torch.tensor(other, dtype=self.dtype, device=self.device)
    # torch casts the other operand down to self's dtype for in-place shifts.
    B = other.to(self.dtype)
    if tuple(B.shape) != tuple(self.shape):
        B = B.expand(self.shape)
    B_c = B.contiguous()

    n_elements = self.numel()
    if n_elements == 0:
        return self

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    if self.is_contiguous():
        irshift_kernel[grid](self, B_c, self, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        return self

    tmp = self.contiguous()
    irshift_kernel[grid](tmp, B_c, tmp, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    self.copy_(tmp)
    return self
