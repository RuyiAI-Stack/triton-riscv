import torch
import triton
import triton.language as tl


@triton.jit
def log1p_kernel(
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
        t = 1.0 + x
        u = t - 1.0
        safe_u = tl.where(u == 0.0, 1.0, u)
        y = tl.where(u == 0.0, x, tl.log(t) * (x / safe_u))
        y = tl.where(x == float("inf"), x, y)
    elif COMPUTE_FP32:
        xc = x.to(tl.float32)
        t = 1.0 + xc
        u = t - 1.0
        safe_u = tl.where(u == 0.0, 1.0, u)
        y = tl.where(u == 0.0, xc, tl.log(t) * (xc / safe_u))
        y = tl.where(xc == float("inf"), xc, y).to(x.dtype)
    else:
        t = 1.0 + x
        u = t - 1.0
        safe_u = tl.where(u == 0.0, 1.0, u)
        y = tl.where(u == 0.0, x, tl.log(t) * (x / safe_u))
        y = tl.where(x == float("inf"), x, y)

    tl.store(out_ptr + offsets, y, mask=mask)


@triton.jit
def complex_log1p_kernel(
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

    real_plus_one = x_real + 1.0
    magnitude_sq = real_plus_one * real_plus_one + x_imag * x_imag
    out_real = 0.5 * tl.log(magnitude_sq)
    out_imag = tl.math.atan2(x_imag, real_plus_one)

    out_real = out_real.to(tl.float64 if COMPUTE_FP64 else tl.float32)
    out_imag = out_imag.to(tl.float64 if COMPUTE_FP64 else tl.float32)
    tl.store(out_ptr + base, out_real, mask=mask)
    tl.store(out_ptr + base + 1, out_imag, mask=mask)


def _launch_log1p(input_tensor: torch.Tensor, out_tensor: torch.Tensor):
    n_elements = input_tensor.numel()
    if n_elements == 0:
        return

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    log1p_kernel[grid](
        input_tensor,
        out_tensor,
        n_elements,
        COMPUTE_FP32=input_tensor.dtype
        in (torch.float16, torch.bfloat16, torch.float32),
        COMPUTE_FP64=input_tensor.dtype == torch.float64,
        BLOCK_SIZE=1024,
    )


def _launch_complex_log1p(input_tensor: torch.Tensor, out_tensor: torch.Tensor):
    n_elements = input_tensor.numel()
    if n_elements == 0:
        return

    input_view = torch.view_as_real(input_tensor).reshape(-1)
    out_view = torch.view_as_real(out_tensor).reshape(-1)

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    complex_log1p_kernel[grid](
        input_view,
        out_view,
        n_elements,
        COMPUTE_FP64=input_view.dtype == torch.float64,
        BLOCK_SIZE=1024,
    )


def _prepare_input(A: torch.Tensor) -> torch.Tensor:
    if not isinstance(A, torch.Tensor):
        raise TypeError("log1p expects a torch.Tensor")
    if A.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    ):
        return A.contiguous()
    return A.to(torch.get_default_dtype()).contiguous()


def _result_dtype(A: torch.Tensor) -> torch.dtype:
    if A.is_complex():
        return A.dtype
    if A.is_floating_point():
        return A.dtype
    return torch.get_default_dtype()


def _validate_out_dtype(result_dtype: torch.dtype, out: torch.Tensor):
    if not torch.can_cast(result_dtype, out.dtype):
        raise RuntimeError(
            f"result type {str(result_dtype).split('.')[-1]} can't be cast to the desired output type {str(out.dtype).split('.')[-1]}"
        )


def log1p(A):
    inp = _prepare_input(A)
    out = torch.empty_like(inp)
    if inp.is_complex():
        _launch_complex_log1p(inp, out)
    else:
        _launch_log1p(inp, out)
    return out.view(A.shape)


def log1p_out(A, *, out):
    inp = _prepare_input(A)
    _validate_out_dtype(_result_dtype(inp), out)
    out.resize_(A.shape)
    need_copy_back = (not out.is_contiguous()) or out.dtype != inp.dtype
    work_out = torch.empty_like(inp) if need_copy_back else out
    if inp.is_complex():
        _launch_complex_log1p(inp, work_out)
    else:
        _launch_log1p(inp, work_out)
    if need_copy_back:
        out.copy_(work_out)
    return out
