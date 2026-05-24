import torch
import triton
import triton.language as tl


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


def var(x, dim=None, *, correction=None, keepdim=False):
    if correction is None:
        correction = 1.0

    if dim is None or len(dim) == x.ndim:
        dim = list(range(x.ndim))
        shape = [1] * x.ndim
        N = x.numel()
        var_out = torch.empty(shape, dtype=x.dtype, device=x.device)
        BLOCK_N = 1024
        BLOCK_NUM = triton.cdiv(N, BLOCK_N)
        acc = torch.empty([BLOCK_NUM], dtype=x.dtype, device=x.device)
        average = torch.empty([BLOCK_NUM], dtype=x.dtype, device=x.device)
        count = torch.empty([BLOCK_NUM], dtype=x.dtype, device=x.device)

        var_kernel_1[(BLOCK_NUM,)](x, acc, average, count, N, BLOCK_N=BLOCK_N)
        var_kernel_2[(1,)](
            acc,
            average,
            count,
            var_out,
            N,
            correction,
            BLOCK_NUM,
            BLOCK_N=1024,
        )
    else:
        shape = list(x.shape)
        dim = [d % x.ndim for d in dim]
        # Create a view with compressed dims
        x_reshaped = x
        N = 1
        for d in sorted(dim, reverse=True):
            N *= shape[d]
        M = x.numel() // N

        # Reshape to 2D for the kernel
        x_2d = x_reshaped.reshape(M, N)
        var_out = torch.empty(M, 1, device=x.device, dtype=x.dtype)

        def grid(META):
            return (triton.cdiv(M, META["BLOCK_M"]),)

        var_welford_kernel[grid](
            x_2d, var_out, M, N, correction, BLOCK_M=1, BLOCK_N=1024
        )

        # Reshape back
        out_shape = []
        for d in range(x.ndim):
            if d in dim:
                out_shape.append(1)
            else:
                out_shape.append(x.shape[d])
        var_out = var_out.reshape(out_shape)

    if not keepdim:
        if dim is not None:
            var_out = var_out.squeeze(dim=dim)
        else:
            var_out = var_out.squeeze()
    return var_out


def var_dim(x, dim=None, *, correction=None, keepdim=False):
    return var(x, dim=dim, correction=correction, keepdim=keepdim)


def var_correction(x, dim=None, *, correction=None, keepdim=False):
    return var(x, dim=dim, correction=correction, keepdim=keepdim)
