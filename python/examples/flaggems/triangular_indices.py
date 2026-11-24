import torch


def triangular_indices(
    row, col, offset=0, *, upper=False, dtype=torch.int64, device="cpu"
):
    if upper:
        return torch.triu_indices(row, col, offset, dtype=dtype, device=device)
    rows = []
    cols = []
    for i in range(row):
        for j in range(col):
            if j - i <= offset:
                rows.append(i)
                cols.append(j)
    if not rows:
        return torch.empty((2, 0), dtype=dtype, device=device)
    return torch.stack(
        (
            torch.tensor(rows, dtype=dtype, device=device),
            torch.tensor(cols, dtype=dtype, device=device),
        )
    )
