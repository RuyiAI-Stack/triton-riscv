import math

import torch
import triton
import triton.language as tl


@triton.jit
def index_add_gather_kernel(inp, index, src, out, N, J, K, alpha):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    input_offset = (pid_m * N + pid_n) * K + pid_k
    accumulator = tl.load(inp + input_offset)
    for j in range(0, J):
        index_value = tl.load(index + j).to(tl.int32)
        source_value = tl.load(src + (pid_m * J + j) * K + pid_k)
        accumulator += tl.where(index_value == pid_n, source_value * alpha, 0)

    tl.store(out + input_offset, accumulator)


def _validate_index_add(inp, dim, index, src):
    if not (-inp.ndim <= dim < inp.ndim):
        raise IndexError("Dimension out of range")
    dim %= inp.ndim
    if index.ndim != 1:
        raise IndexError("index_add_(): Index is supposed to be a vector")
    if index.numel() != src.size(dim):
        raise RuntimeError(
            "The dimth dimension of source must have the same size as "
            "the length of index"
        )
    if inp.ndim != src.ndim:
        raise RuntimeError(
            "Self and source should have the same number of dimensions"
        )
    for axis in range(inp.ndim):
        if axis != dim and inp.size(axis) != src.size(axis):
            raise RuntimeError(
                "src.size(d) == self.size(d) for all dimensions d != dim"
            )
    if not bool(((index >= 0) & (index < inp.size(dim))).all().item()):
        raise IndexError("0 <= index < self.size(dim)")
    return dim


def _index_add_impl(inp, dim, index, src, alpha, out):
    dim = _validate_index_add(inp, dim, index, src)
    source_inp = inp.contiguous()
    index = index.contiguous()
    src = src.contiguous()

    M = math.prod(inp.shape[:dim])
    N = inp.shape[dim]
    J = index.numel()
    K = math.prod(inp.shape[dim + 1 :])

    kernel_out = (
        out
        if out.is_contiguous()
        else torch.empty_like(inp, memory_format=torch.contiguous_format)
    )
    index_add_gather_kernel[(M, N, K)](
        source_inp, index, src, kernel_out, N, J, K, alpha
    )
    if kernel_out is not out:
        out.copy_(kernel_out)
    return out


def index_add(inp, dim, index, src, alpha=1):
    out = torch.empty_like(inp, memory_format=torch.contiguous_format)
    return _index_add_impl(inp, dim, index, src, alpha, out)


def index_add_(inp, dim, index, src, alpha=1):
    return _index_add_impl(inp, dim, index, src, alpha, inp)
