import torch
import triton
import triton.language as tl


@triton.jit
def var_scalar_rows_kernel(X, Var, M, N, correction):
    row = tl.program_id(0)
    base = row * N

    mean = tl.full((), 0.0, dtype=tl.float32)
    for col in range(0, N):
        mean += tl.load(X + base + col).to(tl.float32)
    mean /= N

    squared_deviation = tl.full((), 0.0, dtype=tl.float32)
    for col in range(0, N):
        value = tl.load(X + base + col).to(tl.float32)
        delta = value - mean
        squared_deviation += delta * delta

    tl.store(Var + row, squared_deviation / (N - correction))


@triton.jit
def welford_func(mean_x, count_x, M_x, mean_y, count_y, M_y):
    count = count_x + count_y
    _count = tl.maximum(count, 1)
    mc_x = mean_x * count_x
    mc_y = mean_y * count_y
    mean = (mc_x + mc_y) / _count
    M = M_x + mc_x * mean_x + M_y + mc_y * mean_y - count * mean * mean
    return mean, count, M


@triton.jit
def var_welford_kernel(
    X,
    Var,
    M,
    N,
    correction,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    X = X + pid * N
    Var = Var + pid
    row_mask = pid < M

    _mean = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    _acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    _count = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        x = tl.load(X + cols, mask, other=0.0).to(tl.float32)

        count = _count + mask
        cnt = tl.maximum(count, 1)
        cur_mean = (_mean * _count + x) / cnt
        _acc += (x - cur_mean) * (x - _mean) * mask
        _mean = cur_mean
        _count = count

    _, _, acc = tl.reduce(
        (_mean, _count, _acc), axis=1, combine_fn=welford_func
    )
    var = acc / (N - correction)
    var = var[:, None]
    tl.store(Var, var, row_mask)


@triton.jit
def var_kernel_1(
    X,
    Acc,
    Average,
    Count,
    N,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_N + tl.arange(0, BLOCK_N)

    X = X + offset
    Acc = Acc + pid
    Average = Average + pid
    Count = Count + pid
    mask = offset < N

    x = tl.load(X, mask, other=0.0).to(tl.float32)

    count = tl.sum(mask.to(tl.float32))
    average = tl.sum(x) / count
    acc = tl.sum(x * x) - count * average * average

    tl.store(Average, average)
    tl.store(Acc, acc)
    tl.store(Count, count)


@triton.jit
def var_kernel_2(
    Acc,
    Average,
    Count,
    Var,
    N,
    correction,
    BLOCK_NUM,
    BLOCK_N: tl.constexpr,
):
    offset = tl.arange(0, BLOCK_N)
    mask = offset < BLOCK_NUM
    Acc = Acc + offset
    Average = Average + offset
    Count = Count + offset
    acc = tl.load(Acc, mask, other=0.0).to(tl.float32)
    average = tl.load(Average, mask, other=0.0).to(tl.float32)
    count = tl.load(Count, mask, other=0.0).to(tl.float32)

    _, _, nvar = tl.reduce(
        (average, count, acc), axis=0, combine_fn=welford_func
    )

    var = nvar / (N - correction)
    tl.store(Var, var)


def _prepare_reduction_rows(x, dim):
    if dim is None:
        dims = list(range(x.ndim))
    elif isinstance(dim, int):
        dims = [dim % x.ndim]
    else:
        dims = [d % x.ndim for d in dim]
    dims = sorted(set(dims))
    remaining = [d for d in range(x.ndim) if d not in dims]

    permuted = x.permute(remaining + dims).contiguous()
    reduction_size = 1
    for d in dims:
        reduction_size *= x.shape[d]
    row_count = x.numel() // reduction_size
    rows = permuted.reshape(row_count, reduction_size)
    compact_shape = [x.shape[d] for d in remaining]
    keepdim_shape = [1 if d in dims else x.shape[d] for d in range(x.ndim)]
    return rows, row_count, reduction_size, compact_shape, keepdim_shape


def var(x, dim=None, *, correction=None, keepdim=False):
    correction = 1.0 if correction is None else correction
    rows, M, N, compact_shape, keepdim_shape = _prepare_reduction_rows(x, dim)
    var_out = torch.empty((M,), dtype=x.dtype, device=x.device)
    var_scalar_rows_kernel[(M,)](
        rows, var_out, M, N, correction, num_warps=1
    )
    return var_out.reshape(keepdim_shape if keepdim else compact_shape)


def var_dim(x, dim=None, *, correction=None, keepdim=False):
    return var(x, dim=dim, correction=correction, keepdim=keepdim)


def var_correction(x, dim=None, *, correction=None, keepdim=False):
    return var(x, dim=dim, correction=correction, keepdim=keepdim)
