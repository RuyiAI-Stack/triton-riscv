import torch
import triton
import triton.language as tl


@triton.jit
def heaviside_kernel_tt(
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
def heaviside_kernel_ts(
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


def _heaviside_impl(x: torch.Tensor, values, out=None):
    if isinstance(values, torch.Tensor) and values.dtype != x.dtype:
        # torch: "heaviside is not yet implemented for tensors with
        # different datatypes".
        raise RuntimeError(
            "heaviside is not yet implemented for tensors with different datatypes"
        )
    if x.dtype == torch.bfloat16 or (
        isinstance(values, torch.Tensor) and values.dtype == torch.bfloat16
    ):
        # tl.where miscompiles on bf16 (issue #119); torch fallback.
        if not isinstance(values, torch.Tensor):
            values = torch.tensor(values, dtype=x.dtype)
        if out is not None:
            return torch.heaviside(x, values, out=out)
        return torch.heaviside(x, values)
    BLOCK_SIZE = 1024
    if isinstance(values, torch.Tensor):
        x, values = torch.broadcast_tensors(x, values)
        x_c = x.contiguous()
        values_c = values.contiguous()
        if out is None:
            out = torch.empty_like(x_c)
        out_c = out if out.is_contiguous() else out.contiguous()
        n_elements = x_c.numel()
        if out.numel() != n_elements:
            raise ValueError(
                "Output tensor must have the same number of elements "
                "as the broadcast input"
            )
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        heaviside_kernel_tt[grid](
            x_c, values_c, out_c, n_elements, BLOCK_SIZE=BLOCK_SIZE
        )
        if out_c.data_ptr() != out.data_ptr():
            out.copy_(out_c)
        return out
    else:
        x_c = x.contiguous()
        if out is None:
            out = torch.empty_like(x_c)
        out_c = out if out.is_contiguous() else out.contiguous()
        n_elements = x_c.numel()
        if out.numel() != n_elements:
            raise ValueError(
                "Output tensor must have the same number of elements "
                "as the broadcast input"
            )
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        heaviside_kernel_ts[grid](x_c, values, out_c, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        if out_c.data_ptr() != out.data_ptr():
            out.copy_(out_c)
        return out


def heaviside(x: torch.Tensor, values):
    return _heaviside_impl(x, values)


def heaviside_out(x: torch.Tensor, values, out: torch.Tensor):
    return _heaviside_impl(x, values, out=out)
