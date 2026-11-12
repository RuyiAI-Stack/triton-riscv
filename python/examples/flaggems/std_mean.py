import torch
import triton
import triton.language as tl


@triton.jit
def std_mean_pass1_kernel(
    inp,
    anchor_ptr,
    out_mean_shift,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp_ptrs = inp + rows * N
    anchor_ptrs = anchor_ptr + rows
    mean_ptrs = out_mean_shift + rows
    row_mask = rows < M

    # The anchor is the row's first element; an infinite first element
    # makes every shifted value inf/NaN, so this kernel assumes finite
    # anchors (matching the stable-variance formula's domain).
    # Subtracting a per-row anchor keeps the sums small when the mean
    # dominates the values, avoiding fp32 cancellation.
    anchor = tl.load(anchor_ptrs, mask=row_mask, other=0.0)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        mask = row_mask & (cols < N)
        a = tl.load(inp_ptrs + cols, mask=mask, other=0.0)
        # Padding lanes load 0.0, which must not subtract the anchor.
        acc += tl.where(mask, a - anchor, 0.0)

    mean_shift = tl.sum(acc, axis=1)[:, None] / N
    tl.store(mean_ptrs, mean_shift, mask=row_mask)


@triton.jit
def std_mean_pass2_kernel(
    inp,
    anchor_ptr,
    in_mean_shift,
    out_std,
    out_mean,
    M,
    N,
    correction,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp_ptrs = inp + rows * N
    anchor_ptrs = anchor_ptr + rows
    mean_shift_ptrs = in_mean_shift + rows
    std_ptrs = out_std + rows
    mean_ptrs = out_mean + rows
    row_mask = rows < M

    anchor = tl.load(anchor_ptrs, mask=row_mask, other=0.0)
    mean_shift = tl.load(mean_shift_ptrs, mask=row_mask, other=0.0)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        mask = row_mask & (cols < N)
        a = tl.load(inp_ptrs + cols, mask=mask, other=0.0)
        d = (a - anchor) - mean_shift
        # Padding lanes load 0.0; their deviation from the mean must not
        # enter the variance.
        acc += tl.where(mask, d * d, 0.0)

    denom = N - correction
    # torch yields NaN when correction >= N.
    var = tl.where(denom <= 0, float("nan"), tl.sum(acc, axis=1)[:, None] / denom)
    tl.store(std_ptrs, tl.sqrt(var), mask=row_mask)
    tl.store(mean_ptrs, anchor + mean_shift, mask=row_mask)


def _dim_compress(inp, dims):
    if not isinstance(dims, (list, tuple)):
        dims = [dims]

    dims = [d % inp.ndim for d in dims]
    other_dims = [i for i in range(inp.ndim) if i not in dims]
    perm = other_dims + dims

    inp_perm = inp.permute(perm).contiguous()

    N = 1
    for d in dims:
        N *= inp.shape[d]
    M = inp.numel() // N

    return inp_perm.view(M, N)


def std_mean(inp, dim=None, *, correction=1, keepdim=False):
    if inp.numel() == 0:
        # torch returns (nan, nan) for empty reductions
        return torch.std_mean(inp, dim=dim, correction=correction, keepdim=keepdim)
    if inp.dtype != torch.float32:
        return torch.std_mean(inp, dim=dim, correction=correction, keepdim=keepdim)

    if dim is None:
        ndim = inp.dim()
        inp = inp.reshape(1, -1)
        M, N = 1, inp.numel()
        if keepdim:
            out_shape = [1] * ndim
        else:
            out_shape = []
        dim = None
    else:
        if isinstance(dim, int):
            dim = [dim]
        shape = list(inp.shape)
        dim = [d % inp.ndim for d in dim]
        inp = _dim_compress(inp, dim)

        N = 1
        for i in dim:
            N *= shape[i]
            shape[i] = 1
        M = inp.numel() // N
        out_shape = shape

    anchor = inp[:, 0].contiguous()
    mean_shift = torch.empty((M,), dtype=torch.float32, device=inp.device)
    mean = torch.empty((M,), dtype=torch.float32, device=inp.device)
    std = torch.empty((M,), dtype=torch.float32, device=inp.device)

    BLOCK_M = 32
    BLOCK_N = 32
    grid = (triton.cdiv(M, BLOCK_M),)
    std_mean_pass1_kernel[grid](
        inp, anchor, mean_shift, M, N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
    )
    std_mean_pass2_kernel[grid](
        inp,
        anchor,
        mean_shift,
        std,
        mean,
        M,
        N,
        correction,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )

    mean = mean.view(out_shape)
    std = std.view(out_shape)
    if not keepdim and dim is not None:
        mean = mean.squeeze(dim=dim)
        std = std.squeeze(dim=dim)
    return std, mean
