import torch
import triton
import triton.language as tl

# narrow_copy: aten::narrow as a value copy with real index arithmetic in
# the Triton kernel (each output element decodes its (pre, dim, post)
# coordinates and gathers from the input).
# Upstream behavior kept verbatim: dim is asserted valid, negative start
# wraps modulo the dim size, and length is silently clamped to the
# remaining space (aten narrow would raise on start + length overflow).


@triton.jit
def narrow_copy_kernel(
    out_ptr,
    inp_ptr,
    total_elements,
    inp_total_elements,
    dim_size,
    dim_prod_post,
    start,
    length,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = block_start + offsets < total_elements

    idx = block_start + offsets
    pre_idx = idx // (length * dim_prod_post)
    dim_idx = (idx // dim_prod_post) % length
    post_idx = idx % dim_prod_post

    inp_dim_idx = dim_idx + start
    inp_idx = (
        pre_idx * dim_size * dim_prod_post + inp_dim_idx * dim_prod_post + post_idx
    )

    inp_data = tl.load(inp_ptr + inp_idx, mask=mask)
    tl.store(out_ptr + idx, inp_data, mask=mask)


def narrow_copy(inp, dim, start, length):
    """Copy of ``inp.narrow(dim, start, length)``; length clamps to the available range."""
    assert dim >= -inp.ndim and dim < inp.ndim, (
        f"Invalid dim: {dim} for tensor with {inp.ndim} dimensions"
    )
    dim = dim % inp.ndim

    if start < 0:
        start = start % inp.size(dim)
    length = min(length, inp.size(dim) - start)

    out_shape = list(inp.shape)
    out_shape[dim] = length
    out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)

    if out.numel() == 0:
        return out

    total_elements = out.numel()
    dim_size = inp.size(dim)
    inp_total_elements = inp.numel()

    dim_prod_post = 1
    for d in range(dim + 1, inp.ndim):
        dim_prod_post *= inp.size(d)

    inp = inp.contiguous()
    grid = (triton.cdiv(total_elements, 1024),)
    narrow_copy_kernel[grid](
        out,
        inp,
        total_elements,
        inp_total_elements,
        dim_size,
        dim_prod_post,
        start,
        length,
        BLOCK_SIZE=1024,
    )
    return out
