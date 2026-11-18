import torch
import triton
import triton.language as tl


@triton.jit
def small_n_log_softmax_kernel(
    output_ptr,
    input_ptr,
    M,
    N,
    stride_m,
    BLOCK_N: tl.constexpr,
):
    """Kernel for small N (N <= 1024): single warp, single pass scan."""
    row_id = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_N)
    mask = col_offsets < N

    # Load the entire row in one go
    row = tl.load(
        input_ptr + row_id * stride_m + col_offsets,
        mask=mask,
        other=-float("inf"),
    ).to(tl.float32)

    # Single pass: compute max, then exp and sum
    row_max = tl.max(row, axis=0)
    row_shifted = row - row_max
    exp_row = tl.exp(row_shifted)
    sum_exp = tl.sum(exp_row, axis=0)
    log_sum = tl.log(sum_exp)
    out = row_shifted - log_sum

    # Store result
    tl.store(output_ptr + row_id * stride_m + col_offsets, out, mask=mask)


@triton.jit
def large_n_log_softmax_kernel(
    output_ptr,
    input_ptr,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr = 8,
    BLOCK_N: tl.constexpr = 256,
):
    """Kernel for large N (N > 1024): two-pass online softmax algorithm."""
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    # First pass: compute max and sum of exp
    m = tl.full([BLOCK_M, BLOCK_N], value=float("-inf"), dtype=tl.float32)
    z = tl.full([BLOCK_M, BLOCK_N], value=0.0, dtype=tl.float32)
    for start_n in range(0, N, BLOCK_N):
        n_offset = start_n + tl.arange(0, BLOCK_N)
        offset = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
        mask = (m_offset[:, None] < M) & (n_offset[None, :] < N)
        input_ptrs = input_ptr + offset
        inp = tl.load(input_ptrs, mask=mask, other=-float("inf")).to(tl.float32)
        m_new = tl.maximum(inp, m)
        all_neg_inf = m_new == float("-inf")
        z = tl.where(all_neg_inf, z, z * tl.exp(m - m_new) + tl.exp(inp - m_new))
        m = m_new

    m_reduced = tl.max(m, 1)
    z = tl.sum(z * tl.exp(m - m_reduced[:, None]), 1)
    m = m_reduced

    # Second pass: compute and store output
    for start_n in range(0, N, BLOCK_N):
        n_offset = start_n + tl.arange(0, BLOCK_N)
        offset = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
        mask = (m_offset[:, None] < M) & (n_offset[None, :] < N)
        input_ptrs = input_ptr + offset
        inp = tl.load(input_ptrs, mask=mask, other=-float("inf")).to(tl.float32)
        o = inp - m[:, None] - tl.log(z[:, None])
        tl.store(output_ptr + offset, o, mask=mask)


def special_log_softmax(self, dim, dtype=None):
    assert dim >= -self.ndim and dim < self.ndim, "Invalid dim"
    dim = dim % self.ndim
    N = self.shape[dim]
    permuted = self.movedim(dim, -1).contiguous()
    M = permuted.numel() // N
    inp = permuted.reshape(M, N)

    # Handle dtype conversion
    if dtype is not None and dtype != inp.dtype:
        inp = inp.to(dtype)

    dtype = inp.dtype
    out = torch.empty_like(inp, dtype=dtype)
    K = 1

    # Dual-path: small N uses single-warp kernel, large N uses two-pass kernel
    if N <= 1024:
        BLOCK_N = triton.next_power_of_2(N)
        grid = (M,)
        small_n_log_softmax_kernel[grid](
            out,
            inp,
            M,
            N,
            inp.stride(0),
            BLOCK_N=BLOCK_N,
        )
    else:
        # Large N: two-pass online softmax kernel with autotune
        def grid(meta):
            return (
                triton.cdiv(M, meta["BLOCK_M"]),
                K,
            )

        large_n_log_softmax_kernel[grid](
            out,
            inp,
            M,
            N,
            K,
        )

    return out.reshape(permuted.shape).movedim(-1, dim)
