import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def trunc_kernel_(
    x_ptr,
    n_elements,
    COMPUTE_FP32: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # trunc rounds toward zero (differs from floor on negative inputs) and
    # keeps the sign of -0.0, +-inf and NaN.  libdevice has no fp16
    # signature, so the low-precision dtypes compute in fp32; truncation is
    # exact, so no precision is lost.
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    if COMPUTE_FP32:
        y = libdevice.trunc(x.to(tl.float32)).to(x.dtype)
    else:
        y = libdevice.trunc(x)

    tl.store(x_ptr + offsets, y, mask=mask)


def _find_tensor(*args, **kwargs):
    if len(args) > 0 and isinstance(args[0], torch.Tensor):
        return args[0]
    for key in ("input", "self", "x"):
        val = kwargs.get(key, None)
        if isinstance(val, torch.Tensor):
            return val
    return None


def _trunc_contiguous(t):
    # t must be a contiguous float tensor; mutated in place by the kernel.
    n_elements = t.numel()
    if n_elements == 0:
        return t

    BLOCK_SIZE = 1024
    COMPUTE_FP32 = t.dtype in (torch.float16, torch.bfloat16)
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    trunc_kernel_[grid](t, n_elements, BLOCK_SIZE=BLOCK_SIZE, COMPUTE_FP32=COMPUTE_FP32)
    return t


def trunc_(*args, **kwargs):
    x = _find_tensor(*args, **kwargs)
    if x is None:
        raise ValueError("trunc_: expected a Tensor as the first argument")

    if x.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ):
        # Integer trunc is the identity; aten keeps it bit-exact.
        return torch.ops.aten.trunc_(x)

    if x.is_contiguous():
        _trunc_contiguous(x)
        return x
    else:
        # Rewrite the caller's storage behind the strided view.
        y = _trunc_contiguous(x.contiguous())
        x.copy_(y)
        return x


def trunc(A):
    if A.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ):
        return torch.ops.aten.trunc(A)

    out = A.contiguous()
    if out is A:
        out = A.clone()
    _trunc_contiguous(out)
    return out.view_as(A)


def trunc_out(A, *, out=None):
    if out is None:
        return trunc(A)
    out.copy_(trunc(A))
    return out
