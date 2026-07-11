import torch
import triton
import triton.language as tl


@triton.jit
def soft_margin_loss_elementwise_kernel(
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

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    xf = x.to(tl.float32)
    yf = y.to(tl.float32)
    z = -xf * yf
    absz = tl.abs(z)
    vals = tl.maximum(z, 0.0) + tl.log(1.0 + tl.exp(-absz))

    tl.store(out_ptr + offsets, vals, mask=mask)


@triton.jit
def soft_margin_loss_sum_kernel(
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

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    xf = x.to(tl.float32)
    yf = y.to(tl.float32)
    z = -xf * yf
    absz = tl.abs(z)
    vals = tl.maximum(z, 0.0) + tl.log(1.0 + tl.exp(-absz))
    vals = tl.where(mask, vals, 0.0)

    acc = tl.sum(vals, axis=0)
    tl.store(out_ptr + pid, acc)


@triton.jit
def soft_margin_loss_finalize_kernel(partials_ptr, out_ptr, num_partials):
    acc = tl.full((), 0.0, dtype=tl.float32)
    for offset in range(0, num_partials):
        acc += tl.load(partials_ptr + offset).to(tl.float32)
    tl.store(out_ptr, acc)


def _normalize_reduction(reduction):
    if isinstance(reduction, str):
        r = reduction.lower()
        if r == "none":
            return 0
        if r == "mean":
            return 1
        if r == "sum":
            return 2
        raise ValueError(f"Invalid reduction: {reduction}")
    if isinstance(reduction, int):
        if reduction in (0, 1, 2):
            return reduction
        raise ValueError(f"Invalid reduction int: {reduction}")
    raise ValueError(f"Unsupported reduction type: {type(reduction)}")


def _check_tensors(input: torch.Tensor, target: torch.Tensor):
    if input.numel() != target.numel():
        raise AssertionError(
            "soft_margin_loss: input and target must have the same number of elements."
        )
    if not input.is_contiguous():
        input = input.contiguous()
    if not target.is_contiguous():
        target = target.contiguous()
    return input, target


def _soft_margin_sum(input, target):
    n_elements = input.numel()
    block_size = 1024
    num_partials = triton.cdiv(n_elements, block_size)
    partials = torch.empty((num_partials,), device=input.device, dtype=torch.float32)
    result = torch.empty((), device=input.device, dtype=torch.float32)
    soft_margin_loss_sum_kernel[(num_partials,)](
        input, target, partials, n_elements, BLOCK_SIZE=block_size
    )
    soft_margin_loss_finalize_kernel[(1,)](partials, result, num_partials)
    return result


def soft_margin_loss(input: torch.Tensor, target: torch.Tensor, reduction="mean"):
    input_c, target_c = _check_tensors(input, target)
    red = _normalize_reduction(reduction)
    n_elements = input_c.numel()

    if red == 0:
        out = torch.empty_like(input_c)
        if n_elements == 0:
            return out
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        soft_margin_loss_elementwise_kernel[grid](
            input_c, target_c, out, n_elements, BLOCK_SIZE=BLOCK_SIZE
        )
        return out
    else:
        if n_elements == 0:
            if red == 2:
                return torch.zeros((), device=input.device, dtype=input.dtype)
            else:
                return torch.full(
                    (), float("nan"), device=input.device, dtype=input.dtype
                )
        tmp_sum = _soft_margin_sum(input_c, target_c)
        if red == 2:
            return tmp_sum.to(dtype=input.dtype)
        else:
            return (tmp_sum / float(n_elements)).to(dtype=input.dtype)


def soft_margin_loss_out(input, target, reduction="mean", *, out=None):
    if out is None:
        return soft_margin_loss(input, target, reduction)
    red = _normalize_reduction(reduction)
    n_elements = input.numel()
    if red == 0:
        if n_elements > 0:
            input_c, target_c = _check_tensors(input, target)
            BLOCK_SIZE = 1024
            grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
            soft_margin_loss_elementwise_kernel[grid](
                input_c, target_c, out, n_elements, BLOCK_SIZE=BLOCK_SIZE
            )
        return out
    else:
        if n_elements == 0:
            if red == 2:
                out.fill_(0)
            else:
                out.fill_(float("nan"))
            return out
        input_c, target_c = _check_tensors(input, target)
        tmp_sum = _soft_margin_sum(input_c, target_c)
        if red == 2:
            out.fill_(tmp_sum.to(dtype=input.dtype))
        else:
            out.fill_((tmp_sum / float(n_elements)).to(dtype=input.dtype))
        return out
