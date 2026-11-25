"""Sparse gather gradient with coordinate construction in Triton."""

import torch
import triton
import triton.language as tl


@triton.jit
def _gather_coordinates(
    INDEX,
    COORDS,
    N,
    DIM: tl.constexpr,
    RANK: tl.constexpr,
    SHAPE: tl.constexpr,
    INDEX_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pos = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = pos < N
    index = tl.load(INDEX + pos * INDEX_STRIDE, mask, other=0).to(tl.int64)
    rest = pos
    for d in tl.static_range(len(SHAPE) - 1, -1, -1):
        coordinate = rest % SHAPE[d]
        rest = rest // SHAPE[d]
        if d < RANK:
            value = index if d == DIM else coordinate
            tl.store(COORDS + d * N + pos, value, mask)


def _gather_sparse_backward(self, dim, index, grad):
    rank = self.ndim
    n = grad.numel()
    if rank == 0:
        coords = torch.empty((0, n), dtype=torch.int64, device=index.device)
    elif grad.ndim == 0:
        coords = index.reshape(1, 1).to(torch.int64)
    else:
        coords = torch.empty((rank, n), dtype=torch.int64, device=index.device)
        if n:
            if grad.ndim < rank:
                raise IndexError("gradient rank must be at least the input rank")
            if index.numel() not in (1, n):
                raise IndexError(
                    "index must contain one element or one per gradient element"
                )
            flat = index.reshape(-1).to(torch.int64).contiguous()
            dim = dim + rank if dim < 0 else dim
            _gather_coordinates[(triton.cdiv(n, 256),)](
                flat,
                coords,
                n,
                dim,
                rank,
                tuple(grad.shape),
                0 if index.numel() == 1 else 1,
                BLOCK=256,
            )
    return torch.sparse_coo_tensor(
        coords, grad.reshape(-1), self.shape, device=self.device
    )
