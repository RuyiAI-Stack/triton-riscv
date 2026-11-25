import torch


def _sparse_semi_structured_tile(dense, block=4):
    _ = block
    mask = torch.zeros_like(dense, dtype=torch.bool)
    flat = dense.abs().reshape(dense.shape[0], -1, 4)
    top = torch.topk(flat, 2, dim=-1).indices
    rows = torch.arange(dense.shape[0]).unsqueeze(-1).unsqueeze(-1)
    groups = torch.arange(flat.shape[1]).view(1, -1, 1)
    cols = groups * 4 + top
    mask.view(dense.shape[0], -1)[rows, cols] = True
    return dense * mask, mask
