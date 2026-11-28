from typing import Optional

import torch
import triton
import triton.language as tl

# clone: aten::clone as a Triton flat copy. Non-overlapping dense inputs keep
# their strides (preserve_format); complex tensors copy through view_as_real
# pairs (conj/neg bits resolved first); other memory formats and strided
# inputs fall back to Tensor.copy_. Wrapped in an autograd node so gradients
# keep flowing.


@triton.jit
def _clone_kernel(src_ptr, dst_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    vals = tl.load(src_ptr + offsets, mask=mask)
    tl.store(dst_ptr + offsets, vals, mask=mask)


class _CloneFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, memory_format):
        ctx.memory_format = memory_format
        return _clone_impl(inp, memory_format)

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out, None


def _allocate_preserve_format(inp: torch.Tensor) -> torch.Tensor:
    # Replicating the original strides is only valid for a non-overlapping
    # dense layout; other inputs take PyTorch's suggested layout.
    if torch.ops.aten.is_non_overlapping_and_dense(inp):
        return torch.empty_strided(
            inp.size(), inp.stride(), dtype=inp.dtype, device=inp.device
        )
    return torch.empty_like(inp, memory_format=torch.preserve_format)


def _launch_copy(src: torch.Tensor, dst: torch.Tensor):
    n_elements = src.numel()
    grid = (triton.cdiv(n_elements, 1024),)
    _clone_kernel[grid](src, dst, n_elements, BLOCK_SIZE=1024)


def _clone_impl(inp: torch.Tensor, memory_format: torch.memory_format):
    if inp.numel() == 0:
        if memory_format == torch.preserve_format:
            return _allocate_preserve_format(inp)
        return torch.empty_like(inp, memory_format=memory_format)

    if inp.is_complex():
        if inp.is_conj():
            inp = inp.resolve_conj()
        is_neg = getattr(inp, "is_neg", None)
        if is_neg is not None and is_neg():
            inp = inp.resolve_neg()

        if memory_format == torch.preserve_format:
            out = _allocate_preserve_format(inp)
            if inp.is_contiguous() or torch.ops.aten.is_non_overlapping_and_dense(inp):
                _launch_copy(torch.view_as_real(inp), torch.view_as_real(out))
            else:
                out.copy_(inp)
            return out

        if memory_format == torch.contiguous_format:
            out = torch.empty_like(inp, memory_format=torch.contiguous_format)
            if inp.is_contiguous():
                _launch_copy(torch.view_as_real(inp), torch.view_as_real(out))
            else:
                out.copy_(inp)
            return out

        out = torch.empty(inp.size(), dtype=inp.dtype, device=inp.device)
        out = out.to(memory_format=memory_format)
        out.copy_(inp)
        return out

    if memory_format == torch.preserve_format:
        out = _allocate_preserve_format(inp)
        if inp.is_contiguous() or torch.ops.aten.is_non_overlapping_and_dense(inp):
            _launch_copy(inp, out)
        else:
            out.copy_(inp)
        return out

    if memory_format == torch.contiguous_format:
        out = torch.empty_like(inp, memory_format=torch.contiguous_format)
        if inp.is_contiguous():
            _launch_copy(inp, out)
        else:
            out.copy_(inp)
        return out

    out = torch.empty(inp.size(), dtype=inp.dtype, device=inp.device)
    out = out.to(memory_format=memory_format)
    out.copy_(inp)
    return out


def clone(inp: torch.Tensor, memory_format: Optional[torch.memory_format] = None):
    """Triton copy of ``inp`` (strides preserved by default; complex via real pairs)."""
    if memory_format is None:
        memory_format = torch.preserve_format
    out = _CloneFunction.apply(inp, memory_format)
    return out
