import torch
import triton
import triton.language as tl


_DTYPE_NAMES = {
    torch.float16: "Half",
    torch.bfloat16: "BFloat16",
    torch.float32: "Float",
    torch.float64: "Double",
    torch.int8: "Char",
    torch.uint8: "Byte",
    torch.int16: "Short",
    torch.int32: "Int",
    torch.int64: "Long",
    torch.bool: "Bool",
    torch.complex64: "ComplexFloat",
    torch.complex128: "ComplexDouble",
}


@triton.jit
def floor_kernel(
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
        y = tl.floor(x)
    elif COMPUTE_FP32:
        y = tl.floor(x.to(tl.float32)).to(x.dtype)
    else:
        y = x

    tl.store(out_ptr + offsets, y, mask=mask)


def _launch_floor(input_tensor: torch.Tensor, out_tensor: torch.Tensor):
    n_elements = input_tensor.numel()
    if n_elements == 0:
        return

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    floor_kernel[grid](
        input_tensor,
        out_tensor,
        n_elements,
        COMPUTE_FP32=input_tensor.dtype
        in (torch.float16, torch.bfloat16, torch.float32),
        COMPUTE_FP64=input_tensor.dtype == torch.float64,
        BLOCK_SIZE=1024,
    )


def floor(A):
    if not isinstance(A, torch.Tensor):
        raise TypeError("floor expects a torch.Tensor")
    if A.is_complex():
        raise TypeError("floor is not supported for complex tensors")
    inp = A.contiguous()
    out = torch.empty_like(inp)
    _launch_floor(inp, out)
    return out.view(A.shape)


def floor_out(A, *, out):
    if not isinstance(A, torch.Tensor):
        raise TypeError("floor expects a torch.Tensor")
    if A.is_complex():
        raise TypeError("floor is not supported for complex tensors")
    inp = A.contiguous()
    if out.dtype != inp.dtype:
        raise RuntimeError(
            f"Found dtype {_DTYPE_NAMES.get(out.dtype, str(out.dtype))} "
            f"but expected {_DTYPE_NAMES.get(inp.dtype, str(inp.dtype))}"
        )
    out.resize_(A.shape)
    need_copy_back = not out.is_contiguous()
    work_out = torch.empty_like(inp) if need_copy_back else out
    _launch_floor(inp, work_out)
    if need_copy_back:
        out.copy_(work_out)
    return out
