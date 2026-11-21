import torch
import triton
import triton.language as tl

# log_sigmoid backward: the forward saves exp(-|x|) as its buffer, and the
# backward contract (re-probed on torch CPU, see the log_sigmoid_backward
# behavior row of the porting spec) is
#     z = buffer  ->  grad * where(x < 0, 1, z) / (1 + z)
# With the contract buffer this equals the mathematical derivative
# grad * (1 - sigmoid(x)) to 1.5e-16 in fp64. A non-contract buffer is NOT
# validated: buffer = 0 degenerates to the step function x < 0 -> 1, x >= 0 -> 0.
# When no buffer is supplied the same value is rebuilt from self.
#
# Non-fp64 inputs compute in fp32 (libdevice has no fp16/bf16 signatures).


@triton.jit
def _log_sigmoid_backward_kernel(
    g_ptr, x_ptr, buf_ptr, out_ptr, n_elements, IS64: tl.constexpr, BLOCK: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    g = tl.load(g_ptr + offsets, mask=mask, other=0.0)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    z = tl.load(buf_ptr + offsets, mask=mask, other=0.0)
    if IS64:
        zc = z.to(tl.float64)
    else:
        zc = z.to(tl.float32)
    deriv = tl.where(x < 0.0, 1.0, zc) / (1.0 + zc)
    tl.store(out_ptr + offsets, (g * deriv).to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _log_sigmoid_backward_self_kernel(
    g_ptr, x_ptr, out_ptr, n_elements, IS64: tl.constexpr, BLOCK: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    g = tl.load(g_ptr + offsets, mask=mask, other=0.0)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    if IS64:
        xc = x.to(tl.float64)
    else:
        xc = x.to(tl.float32)
    # Rebuild the same z = exp(-|x|) the forward would have saved.
    z = tl.exp(-tl.abs(xc))
    deriv = tl.where(xc < 0.0, 1.0, z) / (1.0 + z)
    tl.store(out_ptr + offsets, (g * deriv).to(out_ptr.dtype.element_ty), mask=mask)


def log_sigmoid_backward(grad_output, self_t, buffer=None):
    shape = torch.broadcast_shapes(grad_output.shape, self_t.shape)
    g = grad_output.expand(shape).contiguous()
    x = self_t.expand(shape).contiguous()
    out = torch.empty(shape, dtype=g.dtype, device=g.device)
    n = out.numel()
    if n == 0:
        return out
    use_buffer = (
        buffer is not None and buffer.numel() == x.numel() and buffer.dtype == x.dtype
    )
    is64 = g.dtype == torch.float64
    if use_buffer:
        buf = buffer.reshape(x.shape).contiguous()
        _log_sigmoid_backward_kernel[(triton.cdiv(n, 1024),)](
            g, x, buf, out, n, IS64=is64, BLOCK=1024
        )
    else:
        _log_sigmoid_backward_self_kernel[(triton.cdiv(n, 1024),)](
            g, x, out, n, IS64=is64, BLOCK=1024
        )
    return out
