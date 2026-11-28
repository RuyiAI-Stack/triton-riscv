import torch
import triton
import triton.language as tl


@triton.jit
def xlogy_kernel_tt(
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

    # ATen precedence (probed): NaN when y is NaN -- even for x == 0;
    # then 0 when x == 0 (for any other y, including -inf); otherwise
    # x * log(y) in plain IEEE arithmetic.
    x_f32 = x.to(tl.float32)
    y_f32 = y.to(tl.float32)
    prod = x_f32 * tl.log(y_f32)
    res = tl.where(x_f32 == 0.0, 0.0, prod)
    res = tl.where(y_f32 != y_f32, float("nan"), res)

    tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def xlogy_kernel_ts(
    x_ptr,
    y_val,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)

    x_f32 = x.to(tl.float32)
    y_f32 = y_val.to(tl.float32)
    prod = x_f32 * tl.log(y_f32)
    res = tl.where(x_f32 == 0.0, 0.0, prod)
    res = tl.where(y_f32 != y_f32, float("nan"), res)

    tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def xlogy_kernel_st(
    x_val,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    y = tl.load(y_ptr + offsets, mask=mask)

    x_f32 = x_val.to(tl.float32)
    y_f32 = y.to(tl.float32)
    prod = x_f32 * tl.log(y_f32)
    res = tl.where(x_f32 == 0.0, 0.0, prod)
    res = tl.where(y_f32 != y_f32, float("nan"), res)

    tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


def _result_dtype(dtype_a, dtype_b):
    # Upstream declares the INT_TO_FLOAT promotion: integer inputs come out
    # float32 (matching torch's own promotion), never an integer.
    out_dtype = torch.promote_types(dtype_a, dtype_b)
    if not out_dtype.is_floating_point:
        out_dtype = torch.float32
    return out_dtype


def _scalar_result_dtype(tensor_dtype):
    # Weak scalar next to a tensor (torch's scalar promotion + upstream's
    # INT_TO_FLOAT): a floating tensor keeps its dtype, an integer tensor
    # promotes to float32 whatever the scalar is.
    if tensor_dtype.is_floating_point:
        return tensor_dtype
    return torch.float32


def _xlogy_internal(A, B, out=None):
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        if B.device != A.device:
            B = B.to(A.device)
        A, B = torch.broadcast_tensors(A, B)
        A_c = A.contiguous()
        B_c = B.contiguous()

        if out is None:
            out = torch.empty_like(A_c, dtype=_result_dtype(A.dtype, B.dtype))
            out_c = out
        else:
            out_c = out if out.is_contiguous() else out.contiguous()

        n_elements = A_c.numel()
        if n_elements == 0:
            if out_c.data_ptr() != out.data_ptr():
                out.copy_(out_c)
            return out

        grid = (triton.cdiv(n_elements, 1024),)
        xlogy_kernel_tt[grid](A_c, B_c, out_c, n_elements, BLOCK_SIZE=1024)

        if out_c.data_ptr() != out.data_ptr():
            out.copy_(out_c)
        return out

    elif isinstance(A, torch.Tensor):
        A_c = A.contiguous()
        if out is None:
            out = torch.empty_like(A_c, dtype=_scalar_result_dtype(A.dtype))
            out_c = out
        else:
            out_c = out if out.is_contiguous() else out.contiguous()

        n_elements = A_c.numel()
        if n_elements == 0:
            if out_c.data_ptr() != out.data_ptr():
                out.copy_(out_c)
            return out

        grid = (triton.cdiv(n_elements, 1024),)
        xlogy_kernel_ts[grid](A_c, B, out_c, n_elements, BLOCK_SIZE=1024)

        if out_c.data_ptr() != out.data_ptr():
            out.copy_(out_c)
        return out

    elif isinstance(B, torch.Tensor):
        B_c = B.contiguous()
        if out is None:
            out = torch.empty_like(B_c, dtype=_scalar_result_dtype(B.dtype))
            out_c = out
        else:
            out_c = out if out.is_contiguous() else out.contiguous()

        n_elements = B_c.numel()
        if n_elements == 0:
            if out_c.data_ptr() != out.data_ptr():
                out.copy_(out_c)
            return out

        grid = (triton.cdiv(n_elements, 1024),)
        xlogy_kernel_st[grid](A, B_c, out_c, n_elements, BLOCK_SIZE=1024)

        if out_c.data_ptr() != out.data_ptr():
            out.copy_(out_c)
        return out

    else:
        raise TypeError("xlogy() expects at least one Tensor argument")


def xlogy(self, other):
    return _xlogy_internal(self, other)


def xlogy_out(self, other, out):
    return _xlogy_internal(self, other, out=out)


def xlogy_tensor_scalar(self, other):
    return _xlogy_internal(self, other)


def xlogy_tensor_scalar_out(self, other, out):
    return _xlogy_internal(self, other, out=out)


def xlogy_scalar_tensor(self, other):
    return _xlogy_internal(self, other)


def xlogy_scalar_tensor_out(self, other, out):
    return _xlogy_internal(self, other, out=out)
