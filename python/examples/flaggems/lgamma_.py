import torch
import triton
import triton.language as tl


@triton.jit
def _lgamma_positive(x):
    z = x - 1.0
    # Keep coefficients literal in the JIT body: this backend deliberately
    # rejects non-constexpr Python globals in Triton functions.
    acc = tl.full(x.shape, 0.9999999999998099, dtype=x.dtype)
    acc += 676.5203681218851 / (z + 1.0)
    acc += -1259.1392167224028 / (z + 2.0)
    acc += 771.3234287776531 / (z + 3.0)
    acc += -176.6150291621406 / (z + 4.0)
    acc += 12.507343278686905 / (z + 5.0)
    acc += -0.13857109526572012 / (z + 6.0)
    acc += 9.984369578019572e-6 / (z + 7.0)
    acc += 1.5056327351493116e-7 / (z + 8.0)
    t = z + 7.5
    return 0.9189385332046727 + (z + 0.5) * tl.log(t) - t + tl.log(acc)


@triton.jit
def _lgamma_value(x):
    reflected = (
        tl.log(3.141592653589793)
        - tl.log(tl.abs(tl.sin(3.141592653589793 * x)))
        - _lgamma_positive(1.0 - x)
    )
    direct = _lgamma_positive(x)
    return tl.where(x < 0.5, reflected, direct)


@triton.jit
def lgamma_kernel(
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
    y = _lgamma_value(xc)
    inf = tl.full(xc.shape, float("inf"), dtype=xc.dtype)
    is_nonpositive_integer = (xc <= 0.0) & (xc == tl.floor(xc))
    y = tl.where(is_nonpositive_integer | (tl.abs(xc) == inf), inf, y)
    out = y.to(x.dtype) if COMPUTE_FP32 else y
    tl.store(out_ptr + offsets, out, mask=mask)


def lgamma(A):
    if A.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise TypeError(f"lgamma only supports floating tensors, got {A.dtype}")

    src = A.contiguous()
    out = torch.empty_like(src)
    n_elements = src.numel()
    if n_elements == 0:
        return out.view_as(A)

    grid = (triton.cdiv(n_elements, 1024),)
    lgamma_kernel[grid](
        src,
        out,
        n_elements,
        BLOCK_SIZE=1024,
        COMPUTE_FP32=A.dtype in (torch.float16, torch.bfloat16),
    )
    return out.view_as(A)


def lgamma_(A):
    out = lgamma(A)
    A.copy_(out)
    return A
