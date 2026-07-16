import torch
import triton
import triton.language as tl


@triton.jit
def lift_fresh_copy_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x, mask=mask)


def lift_fresh_copy(*args, **kwargs):
    x = None
    if len(args) > 0 and isinstance(args[0], torch.Tensor):
        x = args[0]
    elif "self" in kwargs and isinstance(kwargs["self"], torch.Tensor):
        x = kwargs["self"]
    else:
        for v in list(args) + list(kwargs.values()):
            if isinstance(v, torch.Tensor):
                x = v
                break
    if x is None:
        raise ValueError("lift_fresh_copy expects a Tensor argument")

    x_contig = x.contiguous()
    out = torch.empty_like(x_contig)
    n_elements = x_contig.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    lift_fresh_copy_kernel[grid](x_contig, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out.view_as(x_contig)


def lift_fresh_copy_out(x, out=None):
    if x is None or not isinstance(x, torch.Tensor):
        raise ValueError("lift_fresh_copy_out expects 'x' to be a Tensor")

    x_contig = x.contiguous()

    if out is None:
        out = torch.empty_like(x_contig)
    else:
        if out.dtype != x_contig.dtype:
            raise ValueError("Output tensor 'out' must have the same dtype as input")
        if out.numel() != x_contig.numel() or not out.is_contiguous():
            out.resize_(x_contig.shape)
            if not out.is_contiguous():
                out = out.contiguous()

    n_elements = x_contig.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    lift_fresh_copy_kernel[grid](x_contig, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out.view_as(x_contig)
