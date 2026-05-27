import torch
import triton
import triton.language as tl


@triton.jit
def signbit_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)

    if tl.constexpr(x.dtype.is_fp32()):
        xi32 = x.to(tl.int32, bitcast=True)
        result = xi32 < 0
    elif tl.constexpr(x.dtype.is_fp16()):
        xi16 = x.to(tl.int16, bitcast=True)
        result = xi16 < 0
    elif tl.constexpr(x.dtype.is_bf16()):
        xi16 = x.to(tl.int16, bitcast=True)
        result = xi16 < 0
    elif tl.constexpr(x.dtype.is_fp64()):
        xi64 = x.to(tl.int64, bitcast=True)
        result = xi64 < 0
    else:
        result = x < 0

    tl.store(y_ptr + offsets, result, mask=mask)


def signbit(x):
    x_contig = x.contiguous()
    out = torch.empty_like(x_contig, dtype=torch.bool)
    n_elements = out.numel()
    BLOCK_SIZE = 1024

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    signbit_kernel[grid](x_contig, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


def signbit_out(x, *, out=None):
    if out is None:
        return signbit(x)
    x_contig = x.contiguous()
    n_elements = out.numel()
    BLOCK_SIZE = 1024

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    signbit_kernel[grid](x_contig, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out
