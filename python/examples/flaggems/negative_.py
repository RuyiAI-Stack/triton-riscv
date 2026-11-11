import torch
import triton
import triton.language as tl


@triton.jit
def negative_kernel_(
    x_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = -x

    tl.store(x_ptr + offsets, y, mask=mask)


def negative_(*args, **kwargs):
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
        raise ValueError("negative_: expected a Tensor as the first argument")

    if x.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ):
        return torch.ops.aten.negative_(x)

    n_elements = x.numel()
    if n_elements == 0:
        return x

    BLOCK_SIZE = 1024

    if x.is_contiguous():
        n_elements = x.numel()

        def grid(meta):
            return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        negative_kernel_[grid](x, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        return x
    else:
        y = x.contiguous()
        n_elements = y.numel()

        def grid(meta):
            return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        negative_kernel_[grid](y, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        x.copy_(y)
        return x
