import torch
import triton
import triton.language as tl


@triton.jit
def logaddexp2_kernel_tt(
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

    # log2(2**x + 2**y) = m + log2(1 + 2**-|x-y|) with m = max(x, y).
    x_f32 = x.to(tl.float32)
    y_f32 = y.to(tl.float32)
    m = tl.maximum(x_f32, y_f32)
    delta = x_f32 - y_f32
    res = m + tl.log2(1.0 + tl.exp2(-tl.abs(delta)))
    # delta is NaN for equal infinities (inf - inf); the result is then m.
    res = tl.where(delta != delta, m, res)
    # Genuine NaN inputs must still propagate NaN.
    is_nan = (x_f32 != x_f32) | (y_f32 != y_f32)
    out = tl.where(is_nan, float("nan"), res).to(x.dtype)

    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def logaddexp2_kernel_ts(
    x_ptr,
    y,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    x_f32 = x.to(tl.float32)
    # y is a runtime scalar: promote through the tensor operand instead of
    # calling .to on it.
    m = tl.maximum(x_f32, y)
    delta = x_f32 - y
    res = m + tl.log2(1.0 + tl.exp2(-tl.abs(delta)))
    res = tl.where(delta != delta, m, res)
    is_nan = (x_f32 != x_f32) | (y != y)
    out = tl.where(is_nan, float("nan"), res).to(x.dtype)

    tl.store(out_ptr + offsets, out, mask=mask)


def _logaddexp2_impl(A: torch.Tensor, B, out=None):
    if isinstance(B, torch.Tensor) and A.dtype != B.dtype:
        # torch promotes mixed operand dtypes before computing; a fp64
        # result then flows through the existing torch fallback.
        dt = torch.promote_types(A.dtype, B.dtype)
        A = A.to(dt)
        B = B.to(dt)
    if isinstance(B, torch.Tensor):
        A, B = torch.broadcast_tensors(A, B)
    # The kernel computes in fp32 (keep full fp64 precision via torch);
    # integer inputs are out of the formula's scope.
    if A.dtype == torch.float64 or not A.dtype.is_floating_point:
        result = torch.logaddexp2(A, B)
        if out is not None:
            out.copy_(result)
            return out
        return result

    BLOCK_SIZE = 1024
    if isinstance(B, torch.Tensor):
        A_c = A.contiguous()
        B_c = B.contiguous()
        if out is None:
            out = torch.empty_like(A_c)
        out_c = out if out.is_contiguous() else out.contiguous()
        n_elements = A_c.numel()
        if out.numel() != n_elements:
            raise ValueError(
                "Output tensor must have the same number of elements "
                "as the broadcast input"
            )
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        logaddexp2_kernel_tt[grid](A_c, B_c, out_c, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        if out_c.data_ptr() != out.data_ptr():
            out.copy_(out_c)
        return out
    else:
        A_c = A.contiguous()
        if out is None:
            out = torch.empty_like(A_c)
        out_c = out if out.is_contiguous() else out.contiguous()
        n_elements = A_c.numel()
        if out.numel() != n_elements:
            raise ValueError(
                "Output tensor must have the same number of elements "
                "as the broadcast input"
            )
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        logaddexp2_kernel_ts[grid](A_c, B, out_c, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        if out_c.data_ptr() != out.data_ptr():
            out.copy_(out_c)
        return out


def logaddexp2(A: torch.Tensor, B):
    return _logaddexp2_impl(A, B)


def logaddexp2_out(A: torch.Tensor, B, out: torch.Tensor):
    return _logaddexp2_impl(A, B, out=out)
