import torch
import triton
import triton.language as tl


@triton.jit
def _pdist_locate_pair(pid, N):
    """Map a flat pdist output index to its (i, j) row pair, i < j.

    pdist output ordering for N=4: pairs (0,1), (0,2), (0,3), (1,2), (1,3),
    (2,3). cum_before_i = i*N - i*(i+1)//2 counts the pairs before row i;
    the largest i with cum_before_i <= pid is the first row of the pair.
    Linear scan (upstream algorithm) with N as a runtime value so each N does
    not trigger a recompile.
    """
    row_i = tl.full([], 0, tl.int32)
    for i in range(0, N):
        cum_before_i = i * N - i * (i + 1) // 2
        row_i = tl.where(pid >= cum_before_i, i, row_i)

    cum_before_row_i = row_i * N - row_i * (row_i + 1) // 2
    col_offset = pid - cum_before_row_i
    row_j = row_i + 1 + col_offset
    return row_i, row_j


@triton.jit
def _pdist_forward_p2_kernel(
    x_ptr,
    out_ptr,
    N,
    M,
    stride_x,
    BLOCK_M: tl.constexpr,
):
    """Forward kernel for p=2 case (Euclidean distance)."""
    pid = tl.program_id(0)

    row_i, row_j = _pdist_locate_pair(pid, N)

    i_offset = row_i * stride_x
    j_offset = row_j * stride_x

    # Accumulate in the input's precision: float64 in / float64 acc (torch
    # computes pdist natively per dtype; upstream hard-codes fp32 math).
    acc_dtype = tl.float64 if x_ptr.type.element_ty == tl.float64 else tl.float32

    sum_sq = tl.full([], 0.0, acc_dtype)
    for m in range(0, M, BLOCK_M):
        m_offsets = m + tl.arange(0, BLOCK_M)
        m_mask = m_offsets < M

        xi = tl.load(x_ptr + i_offset + m_offsets, mask=m_mask, other=0.0).to(
            acc_dtype
        )
        xj = tl.load(x_ptr + j_offset + m_offsets, mask=m_mask, other=0.0).to(
            acc_dtype
        )

        diff = xi - xj
        sum_sq += tl.sum(diff * diff)

    dist = tl.sqrt(sum_sq)
    tl.store(out_ptr + pid, dist)


@triton.jit
def _pdist_forward_p1_kernel(
    x_ptr,
    out_ptr,
    N,
    M,
    stride_x,
    BLOCK_M: tl.constexpr,
):
    """Forward kernel for p=1 case (Manhattan distance)."""
    pid = tl.program_id(0)

    row_i, row_j = _pdist_locate_pair(pid, N)

    i_offset = row_i * stride_x
    j_offset = row_j * stride_x

    acc_dtype = tl.float64 if x_ptr.type.element_ty == tl.float64 else tl.float32

    sum_abs = tl.full([], 0.0, acc_dtype)
    for m in range(0, M, BLOCK_M):
        m_offsets = m + tl.arange(0, BLOCK_M)
        m_mask = m_offsets < M

        xi = tl.load(x_ptr + i_offset + m_offsets, mask=m_mask, other=0.0).to(
            acc_dtype
        )
        xj = tl.load(x_ptr + j_offset + m_offsets, mask=m_mask, other=0.0).to(
            acc_dtype
        )

        diff = xi - xj
        sum_abs += tl.sum(tl.abs(diff))

    tl.store(out_ptr + pid, sum_abs)


@triton.jit
def _pdist_forward_p0_kernel(
    x_ptr,
    out_ptr,
    N,
    M,
    stride_x,
    BLOCK_M: tl.constexpr,
):
    """Forward kernel for p=0 case: count of non-matching coordinates.

    torch.pdist(x, 0) counts the coordinates where the two rows differ
    (probed); upstream would return exp(log(count)/0) = inf through its
    general kernel, so this branch is explicit.
    """
    pid = tl.program_id(0)

    row_i, row_j = _pdist_locate_pair(pid, N)

    i_offset = row_i * stride_x
    j_offset = row_j * stride_x

    acc_dtype = tl.float64 if x_ptr.type.element_ty == tl.float64 else tl.float32

    count = tl.full([], 0.0, acc_dtype)
    for m in range(0, M, BLOCK_M):
        m_offsets = m + tl.arange(0, BLOCK_M)
        m_mask = m_offsets < M

        xi = tl.load(x_ptr + i_offset + m_offsets, mask=m_mask, other=0.0).to(
            acc_dtype
        )
        xj = tl.load(x_ptr + j_offset + m_offsets, mask=m_mask, other=0.0).to(
            acc_dtype
        )

        diff = xi - xj
        count += tl.sum((diff != 0.0).to(acc_dtype))

    tl.store(out_ptr + pid, count)


@triton.jit
def _pdist_forward_inf_kernel(
    x_ptr,
    out_ptr,
    N,
    M,
    stride_x,
    BLOCK_M: tl.constexpr,
):
    """Forward kernel for p=inf case (Chebyshev distance)."""
    pid = tl.program_id(0)

    row_i, row_j = _pdist_locate_pair(pid, N)

    i_offset = row_i * stride_x
    j_offset = row_j * stride_x

    acc_dtype = tl.float64 if x_ptr.type.element_ty == tl.float64 else tl.float32

    max_abs = tl.full([], 0.0, acc_dtype)
    for m in range(0, M, BLOCK_M):
        m_offsets = m + tl.arange(0, BLOCK_M)
        m_mask = m_offsets < M

        xi = tl.load(x_ptr + i_offset + m_offsets, mask=m_mask, other=0.0).to(
            acc_dtype
        )
        xj = tl.load(x_ptr + j_offset + m_offsets, mask=m_mask, other=0.0).to(
            acc_dtype
        )

        diff = tl.abs(xi - xj)
        max_abs = tl.maximum(max_abs, tl.max(diff))

    tl.store(out_ptr + pid, max_abs)


@triton.jit
def _pdist_forward_general_kernel(
    x_ptr,
    out_ptr,
    N,
    M,
    p_val,
    stride_x,
    BLOCK_M: tl.constexpr,
):
    """Forward kernel for general p value."""
    pid = tl.program_id(0)

    row_i, row_j = _pdist_locate_pair(pid, N)

    i_offset = row_i * stride_x
    j_offset = row_j * stride_x

    acc_dtype = tl.float64 if x_ptr.type.element_ty == tl.float64 else tl.float32

    sum_pow = tl.full([], 0.0, acc_dtype)
    for m in range(0, M, BLOCK_M):
        m_offsets = m + tl.arange(0, BLOCK_M)
        m_mask = m_offsets < M

        xi = tl.load(x_ptr + i_offset + m_offsets, mask=m_mask, other=0.0).to(
            acc_dtype
        )
        xj = tl.load(x_ptr + j_offset + m_offsets, mask=m_mask, other=0.0).to(
            acc_dtype
        )

        diff = tl.abs(xi - xj)
        powered = tl.where(diff > 0.0, tl.exp(p_val * tl.log(diff)), 0.0)
        sum_pow += tl.sum(powered)

    dist = tl.where(sum_pow > 0.0, tl.exp(tl.log(sum_pow) / p_val), 0.0)

    tl.store(out_ptr + pid, dist)


def _pdist_forward(x, p=2.0):
    """Compute pairwise distances between rows of x.

    Args:
        x: Input tensor of shape (N, M)
        p: The norm degree. Default is 2.0 (Euclidean distance).

    Returns:
        Output tensor of shape (N*(N-1)//2,) containing pairwise distances.
    """
    assert x.dtype in (
        torch.float32,
        torch.float64,
    ), f"pdist only supports float32 and float64, got {x.dtype}"
    assert x.ndim == 2, "pdist only supports 2D input"
    N = x.shape[0]
    M = x.shape[1]

    # torch contract (probed): negative p is rejected before anything else.
    if p < 0:
        raise RuntimeError("pdist only supports non-negative p values")

    if N < 2:
        return torch.empty((0,), dtype=x.dtype, device=x.device)

    x = x.contiguous()
    n_pairs = N * (N - 1) // 2

    # torch contract (probed): empty feature dim -> zeros for every p,
    # including inf. Also avoids triton.next_power_of_2(0) == 0 below.
    if M == 0:
        return torch.zeros((n_pairs,), dtype=x.dtype, device=x.device)

    out = torch.empty((n_pairs,), dtype=x.dtype, device=x.device)

    BLOCK_M = min(triton.next_power_of_2(M), 1024)
    grid = (n_pairs,)

    if p == 2.0:
        _pdist_forward_p2_kernel[grid](
            x,
            out,
            N,
            M,
            x.stride(0),
            BLOCK_M=BLOCK_M,
        )
    elif p == 1.0:
        _pdist_forward_p1_kernel[grid](
            x,
            out,
            N,
            M,
            x.stride(0),
            BLOCK_M=BLOCK_M,
        )
    elif p == 0.0:
        _pdist_forward_p0_kernel[grid](
            x,
            out,
            N,
            M,
            x.stride(0),
            BLOCK_M=BLOCK_M,
        )
    elif abs(p) == float("inf"):
        _pdist_forward_inf_kernel[grid](
            x,
            out,
            N,
            M,
            x.stride(0),
            BLOCK_M=BLOCK_M,
        )
    else:
        _pdist_forward_general_kernel[grid](
            x,
            out,
            N,
            M,
            p,
            x.stride(0),
            BLOCK_M=BLOCK_M,
        )

    return out
