import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def atanh_kernel_(
    x_ptr,
    n_elements,
    COMPUTE_FP32: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Masked lanes load 0.0: atanh(0.0) == 0.0 keeps the math in-domain.
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    if COMPUTE_FP32:
        y = libdevice.atanh(x.to(tl.float32)).to(x.dtype)
    else:
        y = libdevice.atanh(x)

    tl.store(x_ptr + offsets, y, mask=mask)


def atanh_(*args, **kwargs):
    x = None
    if len(args) > 0 and isinstance(args[0], torch.Tensor):
        x = args[0]
    else:
        for key in ("input", "self", "x"):
            val = kwargs.get(key, None)
            if isinstance(val, torch.Tensor):
                x = val
                break
    if x is None:
        raise ValueError("atanh_: expected a Tensor as the first argument")

    if x.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        # fp64 included: libdevice's __nv_atanh double variant is not lowered
        # by the CPU pipeline, so unsupported dtypes go to the aten op.
        return torch.ops.aten.atanh_(x)

    n_elements = x.numel()
    if n_elements == 0:
        return x

    BLOCK_SIZE = 1024
    COMPUTE_FP32 = x.dtype in (torch.float16, torch.bfloat16)

    if x.is_contiguous():
        n_elements = x.numel()

        def grid(meta):
            return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        atanh_kernel_[grid](
            x, n_elements, BLOCK_SIZE=BLOCK_SIZE, COMPUTE_FP32=COMPUTE_FP32
        )
        return x
    else:
        y = x.contiguous()
        n_elements = y.numel()

        def grid(meta):
            return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        atanh_kernel_[grid](
            y, n_elements, BLOCK_SIZE=BLOCK_SIZE, COMPUTE_FP32=COMPUTE_FP32
        )
        x.copy_(y)
        return x
