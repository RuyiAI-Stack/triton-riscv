import torch
import triton
import triton.language as tl

# clamp_max semantics: out = min(input, max).  max may be a Python scalar
# or a tensor broadcastable to the input shape (torch requires the same
# dtype for the tensor form).  NaN propagates: a NaN input element or a
# NaN bound yields NaN, matching torch.clamp_max; tl.minimum lowers to
# IEEE minnum (it drops NaN), so NaN is restored explicitly.  Integer
# inputs are bit-exact through the Triton integer minimum.


@triton.jit(do_not_specialize=["maxi"])
def clamp_max_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    maxi,
    IS_FLOAT: tl.constexpr,
    COMPUTE_FP32: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)

    if IS_FLOAT:
        if COMPUTE_FP32:
            # fp16/bf16 compute in fp32; minimum is exact, so the single
            # narrowing round-trip loses nothing.
            x_f32 = x.to(tl.float32)
            r = tl.minimum(maxi, x_f32)
            r = tl.where((x_f32 != x_f32) | (maxi != maxi), float("nan"), r)
            result = r.to(x.dtype)
        else:
            r = tl.minimum(maxi, x)
            result = tl.where((x != x) | (maxi != maxi), float("nan"), r)
        tl.store(out_ptr + offsets, result, mask=mask)
    else:
        tl.store(out_ptr + offsets, tl.minimum(maxi, x), mask=mask)


@triton.jit
def clamp_max_kernel_tensor(
    x_ptr,
    max_ptr,
    out_ptr,
    n_elements,
    IS_FLOAT: tl.constexpr,
    COMPUTE_FP32: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    maxi = tl.load(max_ptr + offsets, mask=mask)

    if IS_FLOAT:
        if COMPUTE_FP32:
            x_f32 = x.to(tl.float32)
            maxi_f32 = maxi.to(tl.float32)
            r = tl.minimum(maxi_f32, x_f32)
            r = tl.where((x_f32 != x_f32) | (maxi_f32 != maxi_f32), float("nan"), r)
            result = r.to(x.dtype)
        else:
            r = tl.minimum(maxi, x)
            result = tl.where((x != x) | (maxi != maxi), float("nan"), r)
        tl.store(out_ptr + offsets, result, mask=mask)
    else:
        tl.store(out_ptr + offsets, tl.minimum(maxi, x), mask=mask)


_FLOAT_TYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
)


def _launch(A, kernel, mid_args, tail_args):
    A_c = A.contiguous()
    out = torch.empty_like(A_c)
    n_elements = A_c.numel()
    if n_elements == 0:
        return out.view_as(A)

    IS_FLOAT = A.dtype in _FLOAT_TYPES
    COMPUTE_FP32 = A.dtype in (torch.float16, torch.bfloat16)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    kernel[grid](
        A_c,
        *mid_args,
        out,
        n_elements,
        *tail_args,
        IS_FLOAT=IS_FLOAT,
        COMPUTE_FP32=COMPUTE_FP32,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out.view_as(A)


def _scalar_arg(A, maxi):
    if A.dtype in _FLOAT_TYPES:
        return (float(maxi),)
    # Integer tensors need an integer bound (torch rejects float bounds
    # there as well).
    if isinstance(maxi, int):
        return (maxi,)
    if isinstance(maxi, float) and maxi.is_integer():
        return (int(maxi),)
    raise ValueError(
        f"clamp_max(): float bound {maxi!r} cannot be used with integer input"
    )


def clamp_max(A, max_value):
    if isinstance(max_value, torch.Tensor):
        max_c = max_value.expand_as(A).contiguous()
        return _launch(A, clamp_max_kernel_tensor, (max_c,), ())
    return _launch(A, clamp_max_kernel, (), _scalar_arg(A, max_value))


def clamp_max_(A, max_value):
    result = clamp_max(A, max_value)
    A.copy_(result)
    return A


def clamp_max_out(A, max_value, *, out=None):
    if out is None:
        return clamp_max(A, max_value)
    out.copy_(clamp_max(A, max_value))
    return out
