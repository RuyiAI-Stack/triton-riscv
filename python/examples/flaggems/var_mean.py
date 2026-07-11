import torch
import triton
import triton.language as tl

from .var import _prepare_reduction_rows


@triton.jit
def var_mean_scalar_rows_kernel(X, Var, Mean, M, N, correction):
    row = tl.program_id(0)
    base = row * N

    acc_dtype = tl.float64 if X.dtype.element_ty is tl.float64 else tl.float32
    mean = tl.full((), 0.0, dtype=acc_dtype)
    squared_deviation = tl.full((), 0.0, dtype=acc_dtype)
    count = tl.full((), 0.0, dtype=acc_dtype)
    for col in range(0, N):
        value = tl.load(X + base + col).to(acc_dtype)
        count += 1.0
        delta = value - mean
        mean += delta / count
        squared_deviation += delta * (value - mean)

    tl.store(Mean + row, mean)
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
def var_mean_welford_kernel(
    X,
    Var,
    Mean,
    M,
    N,
    correction,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    X = X + pid * N
    Var = Var + pid
    Mean = Mean + pid
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

    mean, _, acc = tl.reduce((_mean, _count, _acc), axis=1, combine_fn=welford_func)
    var = acc / (N - correction)
    mean = mean[:, None]
    var = var[:, None]
    tl.store(Mean, mean, row_mask)
    tl.store(Var, var, row_mask)


@triton.jit
def var_mean_kernel_1(
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
def var_mean_kernel_2(
    Acc,
    Average,
    Count,
    Var,
    Mean,
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

    mean, _, nvar = tl.reduce((average, count, acc), axis=0, combine_fn=welford_func)

    var = nvar / (N - correction)
    tl.store(Mean, mean)
    tl.store(Var, var)


def var_mean(x, dim=None, *, correction=None, keepdim=False):
    correction = 1.0 if correction is None else correction
    rows, M, N, compact_shape, keepdim_shape = _prepare_reduction_rows(x, dim)
    var_out = torch.empty((M,), dtype=x.dtype, device=x.device)
    mean_out = torch.empty((M,), dtype=x.dtype, device=x.device)
    var_mean_scalar_rows_kernel[(M,)](
        rows, var_out, mean_out, M, N, correction, num_warps=1
    )
    output_shape = keepdim_shape if keepdim else compact_shape
    return var_out.reshape(output_shape), mean_out.reshape(output_shape)
