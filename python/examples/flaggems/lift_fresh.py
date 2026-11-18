import torch
import triton
import triton.language as tl


@triton.jit
def _copy_kernel(in_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x, mask=mask)


def lift_fresh(*args, **kwargs):
    """Implements aten::lift_fresh(Tensor self) -> Tensor.

    lift_fresh hands a fresh tensor holding the same values as the input to
    autograd. It is materialized here as an on-device Triton copy so the
    returned tensor is an independent allocation rather than an aliasing view.
    """
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
        raise ValueError("lift_fresh expects a Tensor argument")

    x_contig = x.contiguous()

    # Triton cannot canonicalize complex pointer types on some backends;
    # clone() is a functionally equivalent copy that avoids the triton kernel.
    if x_contig.is_complex():
        return x_contig.clone()

    out = torch.empty_like(x_contig, memory_format=torch.contiguous_format)

    n_elements = x_contig.numel()
    if n_elements == 0:
        # A zero-size launch grid aborts some drivers outright; the freshly
        # allocated empty tensor is already the result.
        return out.view_as(x_contig)

    # BLOCK_SIZE=1024 mirrors the sibling lift_fresh_copy kernel; the op is a
    # memory-bandwidth-bound copy, so a fixed block matches its performance.
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _copy_kernel[grid](x_contig, out, n_elements, BLOCK_SIZE=1024)

    return out.view_as(x_contig)
