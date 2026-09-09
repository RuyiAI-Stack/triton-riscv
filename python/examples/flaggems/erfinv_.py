import torch
import triton
import triton.language as tl


@triton.jit
def _erfinv_initial(x):
    w = -tl.log((1.0 - x) * (1.0 + x))
    z = w - 2.5
    p_low = 2.81022636e-08
    p_low = p_low * z + 3.43273939e-07
    p_low = p_low * z - 3.5233877e-06
    p_low = p_low * z - 4.39150654e-06
    p_low = p_low * z + 0.00021858087
    p_low = p_low * z - 0.00125372503
    p_low = p_low * z - 0.00417768164
    p_low = p_low * z + 0.246640727
    p_low = p_low * z + 1.50140941

    u = tl.sqrt(w) - 3.0
    p_high = -2.00214257e-04
    p_high = p_high * u + 1.00950558e-04
    p_high = p_high * u + 1.34934322e-03
    p_high = p_high * u - 3.67342844e-03
    p_high = p_high * u + 5.73950773e-03
    p_high = p_high * u - 7.62246130e-03
    p_high = p_high * u + 9.43887047e-03
    p_high = p_high * u + 1.00167406
    p_high = p_high * u + 2.83297682

    p = tl.where(w < 5.0, p_low, p_high)
    y = p * x
    return y


@triton.jit
def _erfinv_refine(x, y):
    for _ in tl.static_range(3):
        err = tl.math.erf(y) - x
        deriv = 1.1283791670955126 * tl.exp(-(y * y))
        y = y - err / deriv
    return y


@triton.jit
def erfinv_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    COMPUTE_FP32: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    xc = x.to(tl.float32) if COMPUTE_FP32 else x

    initial = _erfinv_initial(xc)
    refined = _erfinv_refine(xc, initial)
    inf = tl.full(xc.shape, float("inf"), dtype=xc.dtype)
    neg_inf = tl.full(xc.shape, float("-inf"), dtype=xc.dtype)
    nan = tl.full(xc.shape, float("nan"), dtype=xc.dtype)
    out = tl.where(xc == 1.0, inf, refined)
    out = tl.where(xc == -1.0, neg_inf, out)
    out = tl.where(tl.abs(xc) > 1.0, nan, out)
    out = tl.where(xc == 0.0, 0.0, out)
    out = out.to(x.dtype) if COMPUTE_FP32 else out
    tl.store(out_ptr + offsets, out, mask=mask)


def erfinv(x):
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise TypeError(f"erfinv only supports floating tensors, got {x.dtype}")

    src = x.contiguous()
    out = torch.empty_like(src)
    n_elements = src.numel()
    if n_elements == 0:
        return out.view_as(x)

    grid = (triton.cdiv(n_elements, 1024),)
    erfinv_kernel[grid](
        src,
        out,
        n_elements,
        BLOCK_SIZE=1024,
        COMPUTE_FP32=x.dtype in (torch.float16, torch.bfloat16),
    )
    return out.view_as(x)


def erfinv_(x):
    out = erfinv(x)
    x.copy_(out)
    return x
