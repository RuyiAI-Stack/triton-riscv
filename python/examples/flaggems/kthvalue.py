import torch

from .sort import sort


def kthvalue(inp, k, dim=-1, keepdim=False):
    ndim = inp.ndim
    if dim < -ndim or dim >= ndim:
        raise IndexError(
            f"Dimension out of range (expected to be in range of [{-ndim}, {ndim - 1}], but got {dim})"
        )

    if dim < 0:
        dim += ndim

    dim_size = inp.shape[dim]
    if dim_size == 0:
        raise IndexError(
            f"kthvalue(): Expected reduction dim {dim} to have non-zero size."
        )

    if k < 1 or k > dim_size:
        raise RuntimeError(
            f"kthvalue(): selected number k out of range for dimension {dim}"
        )

    if inp.numel() == 0:
        out_shape = list(inp.shape)
        if keepdim:
            out_shape[dim] = 1
        else:
            del out_shape[dim]
        return (
            torch.empty(out_shape, dtype=inp.dtype, device=inp.device),
            torch.empty(out_shape, dtype=torch.int64, device=inp.device),
        )

    work = (
        inp.contiguous()
        if dim == ndim - 1
        else torch.movedim(inp, dim, -1).contiguous()
    )
    sorted_values, sorted_indices = sort(work, dim=-1, descending=False)
    kth_values = sorted_values[..., k - 1 : k]
    kth_indices = sorted_indices[..., k - 1 : k]

    if dim != ndim - 1:
        kth_values = torch.movedim(kth_values, -1, dim)
        kth_indices = torch.movedim(kth_indices, -1, dim)

    if keepdim:
        return kth_values, kth_indices
    return kth_values.squeeze(dim), kth_indices.squeeze(dim)
