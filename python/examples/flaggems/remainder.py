import torch
import triton
import triton.language as tl


@triton.jit
def remainder_kernel_tt(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    MODE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0)

    if MODE == 0:
        # Python-style remainder from the C-style trunc remainder: shift by
        # +y when the signs differ.  Triton's `/` is fp32 true division on
        # ints, so trunc division is spelled `//`.  The masked lanes load
        # y = 1 (0/0 would fault), and y == -1 is diverted through division
        # by 1 because the raw INT_MIN // -1 raises SIGFPE on the host; the
        # negated quotient keeps the two's-complement wrap torch shows
        # (remainder(INT_MIN, -1) == 0).
        y = tl.load(y_ptr + offsets, mask=mask, other=1)
        neg_one = y == -1
        q = x // tl.where(neg_one, 1, y)
        q = tl.where(neg_one, -q, q)
        r = x - y * q
        c = (r != 0) & ((x < 0) ^ (y < 0))
        res = tl.where(c, r + y, r)
    elif MODE == 2:
        # fp16/bf16: compute in fp32 (ATen's opmath); frem is exact, so
        # the adjusted result rounds back bit-for-bit like torch.
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
        x_f32 = x.to(tl.float32)
        y_f32 = y.to(tl.float32)
        r = x_f32 % y_f32
        c = (r != 0) & ((x_f32 < 0) ^ (y_f32 < 0))
        res = tl.where(c, r + y_f32, r).to(x.dtype)
    else:
        # fp32/fp64: IEEE frem gives the C-style remainder exactly; the
        # adjustment implements the sign-of-divisor contract.
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
        r = x % y
        c = (r != 0) & ((x < 0) ^ (y < 0))
        res = tl.where(c, r + y, r)

    tl.store(out_ptr + offsets, res, mask=mask)


@triton.jit
def remainder_kernel_ts(
    x_ptr,
    y_val,
    out_ptr,
    n_elements,
    MODE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0)

    if MODE == 0:
        # Same y == -1 guard as the tt kernel (INT_MIN // -1 faults).
        neg_one = y_val == -1
        q = x // tl.where(neg_one, 1, y_val)
        q = tl.where(neg_one, -q, q)
        r = x - y_val * q
        c = (r != 0) & ((x < 0) ^ (y_val < 0))
        res = tl.where(c, r + y_val, r)
    elif MODE == 2:
        x_f32 = x.to(tl.float32)
        y_f32 = y_val.to(tl.float32)
        r = x_f32 % y_f32
        c = (r != 0) & ((x_f32 < 0) ^ (y_f32 < 0))
        res = tl.where(c, r + y_f32, r).to(x.dtype)
    else:
        r = x % y_val
        c = (r != 0) & ((x < 0) ^ (y_val < 0))
        res = tl.where(c, r + y_val, r)

    tl.store(out_ptr + offsets, res, mask=mask)


@triton.jit
def remainder_kernel_st(
    x_val,
    y_ptr,
    out_ptr,
    n_elements,
    MODE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    if MODE == 0:
        # Masked lanes need a safe divisor, and y == -1 needs the same
        # INT_MIN // -1 guard as the tt kernel.
        y = tl.load(y_ptr + offsets, mask=mask, other=1)
        neg_one = y == -1
        q = x_val // tl.where(neg_one, 1, y)
        q = tl.where(neg_one, -q, q)
        r = x_val - y * q
        c = (r != 0) & ((x_val < 0) ^ (y < 0))
        res = tl.where(c, r + y, r)
    elif MODE == 2:
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
        x_f32 = x_val.to(tl.float32)
        y_f32 = y.to(tl.float32)
        r = x_f32 % y_f32
        c = (r != 0) & ((x_f32 < 0) ^ (y_f32 < 0))
        res = tl.where(c, r + y_f32, r).to(y.dtype)
    else:
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
        r = x_val % y
        c = (r != 0) & ((x_val < 0) ^ (y < 0))
        res = tl.where(c, r + y, r)

    tl.store(out_ptr + offsets, res, mask=mask)


def _mode(dtype):
    # 0 = integer trunc-division formula, 1 = native frem (fp32/fp64),
    # 2 = widened frem (fp16/bf16, ATen's fp32 opmath).
    if not dtype.is_floating_point:
        return 0
    if dtype in (torch.float16, torch.bfloat16):
        return 2
    return 1


def _scalar_operand_dtype(scalar, tensor_dtype):
    # torch's weak-scalar promotion: a floating tensor keeps its dtype; an
    # integer tensor next to a float scalar takes the default float dtype;
    # an integer scalar keeps the integer tensor dtype.
    if tensor_dtype.is_floating_point:
        return tensor_dtype
    if isinstance(scalar, float):
        return torch.get_default_dtype()
    return tensor_dtype


def _check_int_divisor(common_dtype, tensor=None, scalar=None):
    # torch raises RuntimeError('ZeroDivisionError') for integer zero
    # divisors; a raw hardware divide would fault the whole process.
    if common_dtype.is_floating_point:
        return
    if scalar is not None:
        if scalar == 0:
            raise RuntimeError("ZeroDivisionError")
    elif bool((tensor == 0).any()):
        raise RuntimeError("ZeroDivisionError")


def _remainder_invoke(A, B):
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        if B.device != A.device:
            B = B.to(A.device)
        A, B = torch.broadcast_tensors(A, B)
        A_c = A.contiguous()
        B_c = B.contiguous()
        common_dtype = torch.promote_types(A.dtype, B.dtype)
        A_c = A_c.to(common_dtype)
        B_c = B_c.to(common_dtype)
        _check_int_divisor(common_dtype, tensor=B_c)

        out = torch.empty_like(A_c)
        n_elements = A_c.numel()
        if n_elements == 0:
            return out.view_as(A)

        grid = (triton.cdiv(n_elements, 1024),)
        remainder_kernel_tt[grid](
            A_c, B_c, out, n_elements, MODE=_mode(common_dtype), BLOCK_SIZE=1024
        )
        return out.view_as(A)

    elif isinstance(A, torch.Tensor):
        A_c = A.contiguous()
        common_dtype = _scalar_operand_dtype(B, A.dtype)
        A_c = A_c.to(common_dtype)
        _check_int_divisor(common_dtype, scalar=B)

        out = torch.empty_like(A_c)
        n_elements = A_c.numel()
        if n_elements == 0:
            return out.view_as(A)

        grid = (triton.cdiv(n_elements, 1024),)
        remainder_kernel_ts[grid](
            A_c, B, out, n_elements, MODE=_mode(common_dtype), BLOCK_SIZE=1024
        )
        return out.view_as(A)

    elif isinstance(B, torch.Tensor):
        B_c = B.contiguous()
        common_dtype = _scalar_operand_dtype(A, B.dtype)
        B_c = B_c.to(common_dtype)
        _check_int_divisor(common_dtype, tensor=B_c)

        out = torch.empty_like(B_c)
        n_elements = B_c.numel()
        if n_elements == 0:
            return out.view_as(B)

        grid = (triton.cdiv(n_elements, 1024),)
        remainder_kernel_st[grid](
            A, B_c, out, n_elements, MODE=_mode(common_dtype), BLOCK_SIZE=1024
        )
        return out.view_as(B)

    else:
        return None


def remainder(A, B):
    res = _remainder_invoke(A, B)
    if res is not None:
        return res
    # torch.remainder rejects two scalars; Python's % carries exactly the
    # divisor-sign semantics (same as upstream's fallback).
    return torch.tensor(A % B)


def remainder_(A, B):
    # No dtype restriction here: upstream serves integer tensors too.
    res = _remainder_invoke(A, B)
    A.copy_(res)
    return A
