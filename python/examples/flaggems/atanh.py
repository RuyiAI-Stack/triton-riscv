import torch
import triton
import triton.language as tl


@triton.jit
def atanh_kernel(
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
        y = 0.5 * tl.log((1.0 + xc) / (1.0 - xc))
    else:
        xc = x.to(tl.float32) if COMPUTE_FP32 else x
        y = 0.5 * tl.log((1.0 + xc) / (1.0 - xc))
        if COMPUTE_FP32:
            y = y.to(x.dtype)

    tl.store(out_ptr + offsets, y, mask=mask)


@triton.jit
def complex_atanh_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    COMPUTE_FP64: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    base = offsets * 2

    x_real = tl.load(x_ptr + base, mask=mask)
    x_imag = tl.load(x_ptr + base + 1, mask=mask)

    if not COMPUTE_FP64:
        x_real = x_real.to(tl.float32)
        x_imag = x_imag.to(tl.float32)

    num = (1.0 + x_real) * (1.0 + x_real) + x_imag * x_imag
    den = (1.0 - x_real) * (1.0 - x_real) + x_imag * x_imag
    out_real = 0.25 * tl.log(num / den)
    out_imag = 0.5 * tl.math.atan2(
        2.0 * x_imag, 1.0 - x_real * x_real - x_imag * x_imag
    )

    out_real = out_real.to(tl.float64 if COMPUTE_FP64 else tl.float32)
    out_imag = out_imag.to(tl.float64 if COMPUTE_FP64 else tl.float32)
    tl.store(out_ptr + base, out_real, mask=mask)
    tl.store(out_ptr + base + 1, out_imag, mask=mask)


def _prepare_atanh_input(A: torch.Tensor, *, inplace: bool):
    if not isinstance(A, torch.Tensor):
        raise TypeError("atanh expects a torch.Tensor")
    if inplace:
        if not (A.is_floating_point() or A.is_complex()):
            raise RuntimeError(
                "atanh_: result type Float can't be cast to the desired output type"
            )
        return A if A.is_contiguous() else A.contiguous()
    if A.is_floating_point() or A.is_complex():
        return A.contiguous()
    return A.to(torch.get_default_dtype()).contiguous()


def _launch_atanh(input_tensor: torch.Tensor, out_tensor: torch.Tensor):
    n_elements = input_tensor.numel()
    if n_elements == 0:
        return

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    atanh_kernel[grid](
        input_tensor,
        out_tensor,
        n_elements,
        COMPUTE_FP32=input_tensor.dtype
        in (torch.float16, torch.bfloat16, torch.float32),
        COMPUTE_FP64=input_tensor.dtype == torch.float64,
        BLOCK_SIZE=1024,
    )


def _launch_complex_atanh(input_tensor: torch.Tensor, out_tensor: torch.Tensor):
    n_elements = input_tensor.numel()
    if n_elements == 0:
        return

    input_view = torch.view_as_real(input_tensor).reshape(-1)
    out_view = torch.view_as_real(out_tensor).reshape(-1)

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    complex_atanh_kernel[grid](
        input_view,
        out_view,
        n_elements,
        COMPUTE_FP64=input_view.dtype == torch.float64,
        BLOCK_SIZE=1024,
    )


def atanh(A):
    inp = _prepare_atanh_input(A, inplace=False)
    out = torch.empty_like(inp)
    if inp.is_complex():
        _launch_complex_atanh(inp, out)
    else:
        _launch_atanh(inp, out)
    return out.view(A.shape)


def atanh_(A):
    buf = _prepare_atanh_input(A, inplace=True)
    if buf.is_complex():
        _launch_complex_atanh(buf, buf)
    else:
        _launch_atanh(buf, buf)
    if buf is not A:
        A.copy_(buf)
    return A
