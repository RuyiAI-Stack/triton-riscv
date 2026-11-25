import torch

from .detach_copy import _copy_values


def _unique_rows(rows, sorted_rows=True):
    n, cols = rows.shape
    if n == 0:
        return (
            rows,
            torch.empty(0, dtype=torch.int64),
            torch.empty(0, dtype=torch.int64),
        )
    order = torch.arange(n)
    if sorted_rows:
        keys = rows.detach().cpu()
        order = sorted(range(n), key=lambda i: tuple(keys[i].tolist()))
        order = torch.tensor(order, dtype=torch.long, device=rows.device)
    ordered = _copy_values(rows.index_select(0, order))
    keep = [0]
    for i in range(1, n):
        if not torch.equal(ordered[i], ordered[i - 1]):
            keep.append(i)
    keep_t = torch.tensor(keep, device=rows.device)
    unique = _copy_values(ordered.index_select(0, keep_t))
    inverse = torch.empty(n, dtype=torch.int64, device=rows.device)
    counts = torch.zeros(unique.shape[0], dtype=torch.int64, device=rows.device)
    cursor = 0
    for pos in range(n):
        if pos in keep[1:]:
            cursor += 1
        inverse[order[pos]] = cursor
        counts[cursor] += 1
    return unique, inverse, counts


def unique_dim(input, dim=0, sorted=True, return_inverse=False, return_counts=False):
    moved = torch.movedim(input, dim, 0).contiguous()
    rows = moved.reshape(moved.shape[0], -1)
    unique_rows, inverse, counts = _unique_rows(rows, sorted_rows=sorted)
    shape = (unique_rows.shape[0],) + moved.shape[1:]
    values = torch.movedim(unique_rows.view(shape), 0, dim)
    if return_inverse and return_counts:
        return values, inverse, counts
    if return_inverse:
        return values, inverse
    if return_counts:
        return values, counts
    return values
