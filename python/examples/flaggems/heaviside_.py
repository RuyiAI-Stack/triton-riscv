import torch
import triton
import triton.language as tl


@triton.jit
def heaviside_kernel__tt(
    x_ptr,
    values_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    values = tl.load(values_ptr + offsets, mask=mask, other=0.0)

    y = tl.where(x > 0, 1.0, tl.where(x < 0, 0.0, tl.where(x == 0, values, 0.0))).to(
        out_ptr.dtype.element_ty
    )

    tl.store(out_ptr + offsets, y, mask=mask)


@triton.jit
def heaviside_kernel__ts(
    x_ptr,
    values,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    y = tl.where(x > 0, 1.0, tl.where(x < 0, 0.0, tl.where(x == 0, values, 0.0))).to(
        out_ptr.dtype.element_ty
    )

    tl.store(out_ptr + offsets, y, mask=mask)


def heaviside_(x: torch.Tensor, values):
    if isinstance(values, torch.Tensor) and values.dtype != x.dtype:
        raise RuntimeError(
            "heaviside is not yet implemented for tensors with different datatypes"
        )
    if x.dtype == torch.bfloat16 or (
        isinstance(values, torch.Tensor) and values.dtype == torch.bfloat16
    ):
        # tl.where miscompiles on bf16 (issue #119); the torch op keeps
        # the in-place identity contract.
        if not isinstance(values, torch.Tensor):
            values = torch.tensor(values, dtype=x.dtype)
        return torch.heaviside(x, values, out=x)
    # Compute into a contiguous buffer of x's shape, then copy back so the
    # original (possibly non-contiguous) tensor is updated in place.
    if isinstance(values, torch.Tensor):
        values = torch.broadcast_to(values, x.shape)
        x_c = x.contiguous()
        values_c = values.contiguous()
        out = torch.empty_like(x_c)
        n_elements = x_c.numel()
        if n_elements == 0:
            return x
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        heaviside_kernel__tt[grid](
            x_c, values_c, out, n_elements, BLOCK_SIZE=BLOCK_SIZE
        )
    else:
        x_c = x.contiguous()
        out = torch.empty_like(x_c)
        n_elements = x_c.numel()
        if n_elements == 0:
            return x
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        heaviside_kernel__ts[grid](x_c, values, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    x.copy_(out)
    return x
