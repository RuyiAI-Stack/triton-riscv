import torch
import triton
import triton.language as tl


@triton.jit
def div_rn(x, y):
    """div_rn default - round to nearest"""
    result = x / y
    return tl.math.floor(result + 0.5)


@triton.jit
def div_rz(x, y):
    """div_rz default - round toward zero"""
    result = x / y
    return tl.where(result >= 0, tl.math.floor(result), tl.math.ceil(result))


@triton.jit
def fmod(x, y):
    """Fmod default - floating point modulo"""
    quotient = div_rz(x, y)
    return x - y * quotient


@triton.jit
def trunc(x):
    """Trunc default - truncate to integer"""
    return tl.where(x >= 0, tl.math.floor(x), tl.math.ceil(x))


@triton.jit
def trunc_div(x, y):
    """Truncate division with higher precision to match PyTorch."""
    x_f32 = x.to(tl.float32)
    y_f32 = y.to(tl.float32)
    return trunc(x_f32 / y_f32).to(x.dtype)


@triton.jit
def _int_floordiv(x, y):
    r = x % y
    c1 = r != 0
    c2 = (x < 0) ^ (y < 0)
    return tl.where(c1 & c2, x // y - 1, x // y)


@triton.jit
def _float_floordiv(x, y):
    remainder = fmod(x, y)
    imperfect = remainder != 0.0
    different_sign = (x < 0) ^ (y < 0)

    q = div_rn(x - remainder, y)
    q = tl.where(imperfect & different_sign, q - 1, q)

    floor_q = tl.math.floor(q)
    c = q - floor_q > 0.5
    floor_q = tl.where(c, floor_q + 1.0, floor_q)

    q_is_zeros = q == 0.0
    floor_q = tl.where(q_is_zeros, tl.where(different_sign, -0.0, 0.0), floor_q)

    is_div_by_zero = y == 0.0
    float_division = x / y
    out = tl.where(is_div_by_zero, float_division, floor_q)
    return out


@triton.jit
def _remainder(x, y):
    r = x % y
    c1 = r != 0
    c2 = (x < 0) ^ (y < 0)
    return tl.where(c1 & c2, r + y, r)


# True divide kernels
@triton.jit
def true_div_kernel_tt(
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
    res = x / y
    tl.store(out_ptr + offsets, res, mask=mask)


@triton.jit
def true_div_kernel_ts(
    x_ptr,
    y_val,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    res = x / y_val
    tl.store(out_ptr + offsets, res, mask=mask)


@triton.jit
def true_div_kernel_st(
    x_val,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    y = tl.load(y_ptr + offsets, mask=mask)
    res = x_val / y
    tl.store(out_ptr + offsets, res, mask=mask)


# Trunc divide kernels
@triton.jit
def trunc_div_kernel_tt(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    res = trunc_div(x, y)
    tl.store(out_ptr + offsets, res, mask=mask)


@triton.jit
def trunc_div_kernel_ts(
    x_ptr,
    y_val,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.cast(y_val, x.dtype)
    res = trunc_div(x, y)
    tl.store(out_ptr + offsets, res, mask=mask)


@triton.jit
def trunc_div_kernel_st(
    x_val,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    y = tl.load(y_ptr + offsets, mask=mask)
    x = tl.cast(x_val, y.dtype)
    res = trunc_div(x, y)
    tl.store(out_ptr + offsets, res, mask=mask)


# Int Trunc divide kernels
@triton.jit
def int_trunc_div_kernel_tt(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask, other=1)
    res = x // y
    tl.store(out_ptr + offsets, res, mask=mask)


@triton.jit
def int_trunc_div_kernel_ts(
    x_ptr,
    y_val,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    res = x // y_val
    tl.store(out_ptr + offsets, res, mask=mask)


@triton.jit
def int_trunc_div_kernel_st(
    x_val,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    y = tl.load(y_ptr + offsets, mask=mask, other=1)
    res = x_val // y
    tl.store(out_ptr + offsets, res, mask=mask)


# Floor divide kernels
@triton.jit
def floor_div_kernel_tt(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    IS_INT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    if IS_INT:
        res = _int_floordiv(x, y)
    else:
        res = _float_floordiv(x, y)
    tl.store(out_ptr + offsets, res, mask=mask)


@triton.jit
def floor_div_kernel_ts(
    x_ptr,
    y_val,
    out_ptr,
    n_elements,
    IS_INT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    if IS_INT:
        res = _int_floordiv(x, y_val)
    else:
        res = _float_floordiv(x, y_val)
    tl.store(out_ptr + offsets, res, mask=mask)


@triton.jit
def floor_div_kernel_st(
    x_val,
    y_ptr,
    out_ptr,
    n_elements,
    IS_INT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    y = tl.load(y_ptr + offsets, mask=mask)
    if IS_INT:
        res = _int_floordiv(x_val, y)
    else:
        res = _float_floordiv(x_val, y)
    tl.store(out_ptr + offsets, res, mask=mask)


# Remainder kernels
@triton.jit
def rem_kernel_tt(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    res = _remainder(x, y)
    tl.store(out_ptr + offsets, res, mask=mask)


@triton.jit
def rem_kernel_ts(
    x_ptr,
    y_val,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    res = _remainder(x, y_val)
    tl.store(out_ptr + offsets, res, mask=mask)


@triton.jit
def rem_kernel_st(
    x_val,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    y = tl.load(y_ptr + offsets, mask=mask)
    res = _remainder(x_val, y)
    tl.store(out_ptr + offsets, res, mask=mask)


def _invoke_kernel(kernel_tt, kernel_ts, kernel_st, A, B, out=None, kwargs=None):
    if kwargs is None:
        kwargs = {}
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        if B.device != A.device:
            B = B.to(A.device)
        A, B = torch.broadcast_tensors(A, B)
        A_c = A.contiguous()
        B_c = B.contiguous()
        common_dtype = torch.promote_types(A.dtype, B.dtype)
        A_c = A_c.to(common_dtype)
        B_c = B_c.to(common_dtype)
        if out is None:
            # true_divide always returns float unless specified otherwise in some contexts, but
            # actually PyTorch promotes integer division to float.
            # We'll just let PyTorch decide the out type for true_divide if we want, or manually promote.
            if "true_div" in kernel_tt.__name__ and not common_dtype.is_floating_point:
                common_dtype = torch.get_default_dtype()
                A_c = A_c.to(common_dtype)
                B_c = B_c.to(common_dtype)
            out = torch.empty_like(A_c, dtype=common_dtype)
        n_elements = A_c.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        kernel_tt[grid](A_c, B_c, out, n_elements, BLOCK_SIZE=BLOCK_SIZE, **kwargs)
        return out.view_as(A)
    elif isinstance(A, torch.Tensor):
        A_c = A.contiguous()
        if "true_div" in kernel_ts.__name__ and not A_c.is_floating_point():
            A_c = A_c.to(torch.get_default_dtype())
        if out is None:
            out = torch.empty_like(A_c)
        n_elements = A_c.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        kernel_ts[grid](A_c, B, out, n_elements, BLOCK_SIZE=BLOCK_SIZE, **kwargs)
        return out.view_as(A)
    elif isinstance(B, torch.Tensor):
        B_c = B.contiguous()
        if "true_div" in kernel_st.__name__ and not B_c.is_floating_point():
            B_c = B_c.to(torch.get_default_dtype())
        if out is None:
            out = torch.empty_like(B_c)
        n_elements = B_c.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        kernel_st[grid](A, B_c, out, n_elements, BLOCK_SIZE=BLOCK_SIZE, **kwargs)
        return out.view_as(B)
    else:
        return None


def true_divide(A, B):
    res = _invoke_kernel(
        true_div_kernel_tt, true_div_kernel_ts, true_div_kernel_st, A, B
    )
    if res is not None:
        return res
    return torch.tensor(A / B)


def true_divide_out(A, B, out):
    res = _invoke_kernel(
        true_div_kernel_tt,
        true_div_kernel_ts,
        true_div_kernel_st,
        A,
        B,
        out=out,
    )
    if res is not None:
        return res
    return torch.tensor(A / B) if out is None else out.fill_(A / B)


def true_divide_(A, B):
    _invoke_kernel(
        true_div_kernel_tt, true_div_kernel_ts, true_div_kernel_st, A, B, out=A
    )
    return A


def trunc_divide(A, B):
    is_int_A = (
        isinstance(A, torch.Tensor) and not A.is_floating_point()
    ) or isinstance(A, int)
    is_int_B = (
        isinstance(B, torch.Tensor) and not B.is_floating_point()
    ) or isinstance(B, int)
    if is_int_A and is_int_B:
        res = _invoke_kernel(
            int_trunc_div_kernel_tt,
            int_trunc_div_kernel_ts,
            int_trunc_div_kernel_st,
            A,
            B,
        )
    else:
        res = _invoke_kernel(
            trunc_div_kernel_tt, trunc_div_kernel_ts, trunc_div_kernel_st, A, B
        )
    if res is not None:
        return res
    return torch.tensor(A / B)


def trunc_divide_(A, B):
    is_int_A = (
        isinstance(A, torch.Tensor) and not A.is_floating_point()
    ) or isinstance(A, int)
    is_int_B = (
        isinstance(B, torch.Tensor) and not B.is_floating_point()
    ) or isinstance(B, int)
    if is_int_A and is_int_B:
        _invoke_kernel(
            int_trunc_div_kernel_tt,
            int_trunc_div_kernel_ts,
            int_trunc_div_kernel_st,
            A,
            B,
            out=A,
        )
    else:
        _invoke_kernel(
            trunc_div_kernel_tt,
            trunc_div_kernel_ts,
            trunc_div_kernel_st,
            A,
            B,
            out=A,
        )
    return A


def floor_divide(A, B):
    is_int = False
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        is_int = not A.is_floating_point() and not B.is_floating_point()
    elif isinstance(A, torch.Tensor):
        is_int = not A.is_floating_point() and isinstance(B, int)
    elif isinstance(B, torch.Tensor):
        is_int = not B.is_floating_point() and isinstance(A, int)

    res = _invoke_kernel(
        floor_div_kernel_tt,
        floor_div_kernel_ts,
        floor_div_kernel_st,
        A,
        B,
        kwargs={"IS_INT": is_int},
    )
    if res is not None:
        return res
    return torch.tensor(A // B)


def floor_divide_(A, B):
    is_int = False
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        is_int = not A.is_floating_point() and not B.is_floating_point()
    elif isinstance(A, torch.Tensor):
        is_int = not A.is_floating_point() and isinstance(B, int)
    elif isinstance(B, torch.Tensor):
        is_int = not B.is_floating_point() and isinstance(A, int)

    _invoke_kernel(
        floor_div_kernel_tt,
        floor_div_kernel_ts,
        floor_div_kernel_st,
        A,
        B,
        out=A,
        kwargs={"IS_INT": is_int},
    )
    return A


def div_mode(A, B, rounding_mode=None):
    if rounding_mode is None:
        return true_divide(A, B)
    elif rounding_mode == "trunc":
        return trunc_divide(A, B)
    elif rounding_mode == "floor":
        return floor_divide(A, B)
    else:
        raise ValueError(
            f"div expected rounding_mode to be one of None, 'trunc', or 'floor' but found {rounding_mode}."
        )


def div_mode_(A, B, rounding_mode=None):
    if rounding_mode is None:
        return true_divide_(A, B)
    elif rounding_mode == "trunc":
        return trunc_divide_(A, B)
    elif rounding_mode == "floor":
        return floor_divide_(A, B)
    else:
        raise ValueError(
            f"div expected rounding_mode to be one of None, 'trunc', or 'floor' but found {rounding_mode}."
        )


def remainder(A, B):
    res = _invoke_kernel(rem_kernel_tt, rem_kernel_ts, rem_kernel_st, A, B)
    if res is not None:
        return res
    return torch.tensor(A % B)


def remainder_(A, B):
    _invoke_kernel(rem_kernel_tt, rem_kernel_ts, rem_kernel_st, A, B, out=A)
    return A


# Map functions for export
div = div_mode
div_ = div_mode_
