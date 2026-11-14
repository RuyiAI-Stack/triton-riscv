import torch
import triton
import triton.language as tl


@triton.jit
def log_sigmoid_forward_kernel(
    x_ptr,
    out_ptr,
    buffer_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Stable form: log_sigmoid(x) = min(x, 0) - log(1 + exp(-|x|)); the
    # backward buffer is exp(-|x|).
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    xf = x.to(tl.float32)
    abs_x = tl.abs(xf)
    y = tl.minimum(xf, 0.0) - tl.log(1.0 + tl.exp(-abs_x))

    tl.store(out_ptr + offsets, y.to(x.dtype), mask=mask)
    tl.store(buffer_ptr + offsets, tl.exp(-abs_x).to(x.dtype), mask=mask)


def log_sigmoid_forward(x: torch.Tensor):
    if not x.dtype.is_floating_point:
        # torch: '"log_sigmoid_cpu" not implemented for integer inputs'.
        raise NotImplementedError(
            '"log_sigmoid_cpu" not implemented for integer inputs'
        )
    if x.dtype == torch.float64:
        # The kernel computes in fp32; keep full fp64 precision via aten.
        return torch.ops.aten.log_sigmoid_forward(x)

    x_c = x.contiguous()
    out = torch.empty_like(x_c)
    buffer = torch.empty_like(x_c)

    n_elements = x_c.numel()
    if n_elements == 0:
        return out, buffer

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    log_sigmoid_forward_kernel[grid](
        x_c, out, buffer, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )

    return out, buffer
