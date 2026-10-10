import torch
import triton
import triton.language as tl


@triton.jit
def trunc_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    COMPUTE_FP32: tl.constexpr,
    COMPUTE_FP64: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)

    if COMPUTE_FP64:
        xc = x
        y = tl.where(xc >= 0, tl.floor(xc), tl.ceil(xc))
    elif COMPUTE_FP32:
        xc = x.to(tl.float32)
        y = tl.where(xc >= 0, tl.floor(xc), tl.ceil(xc)).to(x.dtype)
    else:
        y = x

    tl.store(out_ptr + offsets, y, mask=mask)


def _launch_trunc(input_tensor: torch.Tensor, out_tensor: torch.Tensor):
    n_elements = input_tensor.numel()
    if n_elements == 0:
        return

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    trunc_kernel[grid](
        input_tensor,
        out_tensor,
        n_elements,
        COMPUTE_FP32=input_tensor.dtype
        in (torch.float16, torch.bfloat16, torch.float32),
        COMPUTE_FP64=input_tensor.dtype == torch.float64,
        BLOCK_SIZE=1024,
    )


def trunc(A):
    if not isinstance(A, torch.Tensor):
        raise TypeError("trunc expects a torch.Tensor")
    if A.is_complex():
        raise TypeError("trunc is not supported for complex tensors")
    inp = A.contiguous()
    out = torch.empty_like(inp)
    _launch_trunc(inp, out)
    return out.view(A.shape)


def trunc_(A):
    if not isinstance(A, torch.Tensor):
        raise TypeError("trunc_ expects a torch.Tensor")
    if A.is_complex():
        raise TypeError("trunc_ is not supported for complex tensors")
    buf = A if A.is_contiguous() else A.contiguous()
    _launch_trunc(buf, buf)
    if buf is not A:
        A.copy_(buf)
    return A
