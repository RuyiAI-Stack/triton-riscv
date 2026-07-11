import torch
import triton
import triton.language as tl

from .all import all


@triton.jit
def isclose_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    rtol,
    atol,
    n_elements,
    equal_nan: tl.constexpr,
    zero_tol: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    cast_x = x if x.dtype.is_fp64() else x.to(tl.float32)
    cast_y = y if x.dtype.is_fp64() else y.to(tl.float32)
    if x.dtype.is_bf16():
        close = cast_x == cast_y
    else:
        close = x == y
    if equal_nan:
        close |= (cast_x != cast_x) & (cast_y != cast_y)
    if not zero_tol:
        allowed = atol + tl.abs(rtol * cast_y)
        actual = tl.abs(cast_x - cast_y)
        actual_finite = actual == actual
        close |= actual_finite & (actual <= allowed)
    tl.store(out_ptr + offsets, close.to(tl.int8), mask=mask)


def isclose(
    A: torch.Tensor,
    B: torch.Tensor,
    rtol=1e-05,
    atol=1e-08,
    equal_nan: bool = False,
) -> torch.Tensor:
    if A.dtype == torch.bool:
        return A == B
    if A.dtype != B.dtype:
        raise RuntimeError(f"{A.dtype} did not match {B.dtype}")
    if A.is_quantized or B.is_quantized:
        raise RuntimeError("isclose is not supported for quantized inputs.")
    if rtol < 0:
        raise RuntimeError(
            f"rtol must be greater than or equal to zero, but got {rtol}"
        )
    if atol < 0:
        raise RuntimeError(
            f"atol must be greater than or equal to zero, but got {atol}"
        )
    A_c = A.contiguous()
    B_c = B.contiguous()
    n_elements = A_c.numel()
    out = torch.empty_like(A_c, dtype=torch.uint8)
    zero_tol = (rtol == 0) and (atol == 0)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    isclose_kernel[grid](
        A_c,
        B_c,
        out,
        rtol,
        atol,
        n_elements,
        equal_nan=equal_nan,
        zero_tol=zero_tol,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out.bool().view_as(A)


def allclose(
    A: torch.Tensor,
    B: torch.Tensor,
    rtol=1e-05,
    atol=1e-08,
    equal_nan: bool = False,
) -> bool:
    return all(isclose(A, B, rtol, atol, equal_nan)).item()
