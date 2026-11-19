import torch
import triton
import triton.language as tl


@triton.jit
def _masked_softmax_backward_kernel(
    grad_output_ptr,
    output_ptr,
    mask_ptr,
    grad_input_ptr,
    N,
    inner,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    o = pid // inner
    i = pid % inner
    base = o * (N * inner) + i

    r = tl.arange(0, BLOCK_SIZE)
    idx = base + r * inner
    valid = r < N

    m = tl.load(mask_ptr + idx, mask=valid, other=0)
    mbool = m != 0
    out = tl.load(output_ptr + idx, mask=valid, other=0.0).to(tl.float32)
    go = tl.load(grad_output_ptr + idx, mask=valid, other=0.0).to(tl.float32)

    # Masked-out positions carry no gradient; their output is zeroed before
    # use, so NaN lanes of a raw (non-prezeroed) fully-masked forward row
    # cannot poison the dot product -- those rows backprop to exactly 0.
    out = tl.where(mbool, 0.0, out)
    tmp = out * go
    tmp = tl.where(valid, tmp, 0.0)
    scale = tl.sum(tmp, axis=0)

    grad_input = out * (go - scale)
    grad_input = tl.where(mbool, 0.0, grad_input)

    tl.store(
        grad_input_ptr + idx,
        grad_input.to(grad_input_ptr.dtype.element_ty),
        mask=valid,
    )


def _masked_softmax_backward(grad_output, output, mask, dim=None):
    # grad of _masked_softmax w.r.t. its input, elementwise (mask_type-2)
    # mask: grad = where(mask, 0, out' * (grad_output - sum(out' *
    # grad_output))) with out' = where(mask, 0, output).  dim defaults to
    # the last dim; the mask is compared against 0, so bool and integer
    # masks both work (no bool assert here, unlike the forward).
    output = output.contiguous()
    grad_output = grad_output.contiguous()
    mask = mask.expand(output.shape).contiguous()

    ndim = output.dim()
    if dim is None:
        dim = ndim - 1
    if dim < 0:
        dim += ndim
    assert 0 <= dim < ndim, "Invalid dim for masked softmax backward"

    shape = output.shape
    N = shape[dim]
    inner = 1
    for d in range(dim + 1, ndim):
        inner *= shape[d]
    outer = 1
    for d in range(dim):
        outer *= shape[d]
    M = outer * inner

    grad_input = torch.empty_like(output)

    if M == 0 or N == 0:
        return grad_input

    BLOCK_SIZE = triton.next_power_of_2(N)
    if BLOCK_SIZE <= 128:
        num_warps = 2
    elif BLOCK_SIZE <= 512:
        num_warps = 4
    elif BLOCK_SIZE <= 2048:
        num_warps = 8
    else:
        num_warps = 16

    grid = (M,)
    _masked_softmax_backward_kernel[grid](
        grad_output,
        output,
        mask,
        grad_input,
        N,
        inner,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return grad_input
