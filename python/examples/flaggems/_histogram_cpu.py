"""Shared multidimensional histogram kernels with packed edge storage."""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _count(
    INPUT,
    WEIGHT,
    EDGES,
    OUT,
    N,
    D: tl.constexpr,
    BINS: tl.constexpr,
    STARTS: tl.constexpr,
    STRIDES: tl.constexpr,
    STEPS: tl.constexpr,
    WEIGHTED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = row < N
    out_index = tl.full((BLOCK,), 0, tl.int64)
    for d in tl.static_range(D):
        x = tl.load(INPUT + row * D + d, row < N, other=0)
        first = tl.load(EDGES + STARTS[d])
        last = tl.load(EDGES + STARTS[d] + BINS[d])
        valid &= (x >= first) & (x <= last)
        low = tl.full((BLOCK,), 0, tl.int32)
        high = tl.full((BLOCK,), BINS[d] + 1, tl.int32)
        for _ in range(STEPS[d]):
            mid = (low + high) // 2
            edge = tl.load(EDGES + STARTS[d] + mid, mid <= BINS[d], other=float("inf"))
            right = x >= edge
            active = low < high
            low = tl.where(active & right, mid + 1, low)
            high = tl.where(active & ~right, mid, high)
        b = tl.minimum(low - 1, BINS[d] - 1)
        out_index += b.to(tl.int64) * STRIDES[d]
    if WEIGHTED:
        value = tl.load(WEIGHT + row, row < N, other=0)
    else:
        value = tl.full((BLOCK,), 1, OUT.dtype.element_ty)
    tl.atomic_add(OUT + out_index, value, valid)


@triton.jit
def _total(HIST, TOTAL, N, BLOCK: tl.constexpr):
    acc = tl.full((BLOCK,), 0, HIST.dtype.element_ty)
    for base in range(0, N, BLOCK):
        i = base + tl.arange(0, BLOCK)
        acc += tl.load(HIST + i, i < N, other=0)
    tl.store(TOTAL, tl.sum(acc))


@triton.jit
def _density(
    HIST,
    TOTAL,
    EDGES,
    N,
    D: tl.constexpr,
    BINS: tl.constexpr,
    STARTS: tl.constexpr,
    STRIDES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < N
    volume = tl.full((BLOCK,), 1, HIST.dtype.element_ty)
    for d in tl.static_range(D):
        b = i // STRIDES[d] % BINS[d]
        left = tl.load(EDGES + STARTS[d] + b, mask, other=0)
        right = tl.load(EDGES + STARTS[d] + b + 1, mask, other=1)
        volume *= right - left
    value = tl.load(HIST + i, mask, other=0)
    total = tl.load(TOTAL)
    tl.store(HIST + i, value / total / volume, mask)


def histogram_edges(input, edges, weight=None, density=False):
    d = input.shape[-1]
    if input.ndim < 2 or len(edges) != d:
        raise ValueError(
            "input must have shape (..., dimensions), with one edge tensor per dimension"
        )
    if input.dtype not in (torch.float32, torch.float64):
        raise TypeError("histogram supports float32 and float64")
    for edge in edges:
        if edge.ndim != 1 or edge.numel() < 2 or edge.dtype != input.dtype:
            raise ValueError(
                "edges must be 1D, have at least two entries, and match input dtype"
            )
        if not bool(torch.all(edge[1:] >= edge[:-1])):
            raise ValueError("bin edges must be sorted")
    x = input.reshape(-1, d).contiguous()
    if weight is not None and (
        weight.numel() != x.shape[0] or weight.dtype != input.dtype
    ):
        raise ValueError("weights must match input points and dtype")
    bins = tuple(e.numel() - 1 for e in edges)
    starts = tuple(sum(e.numel() for e in edges[:i]) for i in range(d))
    strides = tuple(math.prod(bins[i + 1 :]) for i in range(d))
    packed = torch.cat([e.contiguous() for e in edges])
    hist = torch.zeros(bins, dtype=x.dtype, device=x.device)
    w = weight.reshape(-1).contiguous() if weight is not None else x
    if x.shape[0]:
        _count[(triton.cdiv(x.shape[0], 256),)](
            x,
            w,
            packed,
            hist,
            x.shape[0],
            d,
            bins,
            starts,
            strides,
            tuple((b + 1).bit_length() for b in bins),
            weight is not None,
            BLOCK=256,
        )
    if density:
        total = torch.empty((), dtype=hist.dtype, device=hist.device)
        _total[(1,)](hist, total, hist.numel(), BLOCK=256)
        _density[(triton.cdiv(hist.numel(), 256),)](
            hist, total, packed, hist.numel(), d, bins, starts, strides, BLOCK=256
        )
    return hist


def uniform_edges(input, bins, range=None):
    d = input.shape[-1]
    if isinstance(bins, int):
        bins = [bins] * d
    if len(bins) != d or any(int(b) <= 0 for b in bins):
        raise ValueError("bin counts must be positive and match input dimensions")
    if range is not None and len(range) != 2 * d:
        raise ValueError("range must have a lower and upper bound per dimension")
    edges = []
    for i, b in enumerate(bins):
        if range is None:
            from ._aminmax import _aminmax

            column = input[..., i].contiguous()
            lo, hi = map(float, _aminmax(column)) if column.numel() else (0.0, 1.0)
        else:
            lo, hi = map(float, range[2 * i : 2 * i + 2])
        if not math.isfinite(lo) or not math.isfinite(hi) or lo > hi:
            raise ValueError("histogram range must be finite and increasing")
        if lo == hi:
            lo -= 0.5
            hi += 0.5
        edges.append(
            torch.linspace(lo, hi, int(b) + 1, dtype=input.dtype, device=input.device)
        )
    return edges
