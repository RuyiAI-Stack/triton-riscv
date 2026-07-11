import torch
import triton
import triton.language as tl


@triton.jit
def nonzero_single_pass_kernel(
    inp,
    out,
    n_elements,
    shape,
    ndim: tl.constexpr,
):
    target_index = tl.program_id(0)
    selected_count = tl.full((), 0, dtype=tl.int32)
    flat_index = tl.full((), 0, dtype=tl.int32)

    for offset in range(0, n_elements):
        is_nonzero = tl.load(inp + offset).to(tl.int32) != 0
        take = is_nonzero & (selected_count == target_index)
        flat_index = tl.where(take, offset, flat_index)
        selected_count += is_nonzero.to(tl.int32)

    remaining = flat_index
    for dim in range(ndim - 1, -1, -1):
        dim_size = tl.load(shape + dim)
        coordinate = remaining % dim_size
        remaining //= dim_size
        tl.store(out + target_index * ndim + dim, coordinate)


def nonzero(inp, *, as_tuple=False):
    inp = inp.contiguous()
    n_elements = inp.numel()
    inp_bytes = (inp.reshape(-1) != 0).to(torch.uint8)
    num_nonzeros = inp_bytes.sum().item()

    shape = torch.tensor(inp.shape, dtype=torch.int32, device=inp.device)
    out_i32 = torch.empty(
        (num_nonzeros, inp.ndim), dtype=torch.int32, device=inp.device
    )
    if num_nonzeros > 0:
        nonzero_single_pass_kernel[(num_nonzeros,)](
            inp_bytes,
            out_i32,
            n_elements,
            shape,
            inp.ndim,
            num_warps=1,
        )
    out = out_i32.to(torch.int64)

    if as_tuple:
        return torch.unbind(out, dim=1)
    return out
