import torch
import triton
import triton.language as tl


@triton.jit
def hamming_window_kernel(
    out_ptr,
    formula_len,
    window_length,
    alpha,
    beta,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < window_length
    n = offsets.to(tl.float32)
    m1 = (formula_len - 1).to(tl.float32)
    val = alpha - beta * tl.cos(6.283185307179586 * n / m1)
    tl.store(out_ptr + offsets, val.to(out_ptr.dtype.element_ty), mask=mask)


def hamming_window(
    window_length,
    periodic=True,
    alpha=0.54,
    beta=0.46,
    *,
    dtype=None,
    layout=None,
    device=None,
    requires_grad=False,
):
    if dtype is not None and not dtype.is_floating_point:
        # torch: "...window expects floating point dtypes"
        raise RuntimeError(
            f"{__name__.rsplit('.', 1)[-1]} expects floating point dtypes, got: {dtype}"
        )
    if layout not in (None, torch.strided):
        raise NotImplementedError("only torch.strided layout is supported")
    if device is not None and str(device) not in ("cpu", "torch:cpu"):
        raise NotImplementedError("only the CPU device is supported")
    if requires_grad:
        raise NotImplementedError("the example kernels do not participate in autograd")
    if dtype is None:
        dtype = torch.get_default_dtype()
    if dtype == torch.float64:
        return torch.hamming_window(
            window_length, periodic=periodic, alpha=alpha, beta=beta, dtype=dtype
        )

    out = torch.empty((window_length,), dtype=dtype)
    if window_length == 0:
        return out
    if window_length == 1:
        # torch special-cases length-1 windows to [1.0]; the formulas are
        # undefined or wrong at a single point.
        out.fill_(1.0)
        return out
    formula_len = window_length + 1 if periodic else window_length

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(window_length, BLOCK_SIZE),)
    hamming_window_kernel[grid](
        out, formula_len, window_length, alpha, beta, BLOCK_SIZE=BLOCK_SIZE
    )
    return out
