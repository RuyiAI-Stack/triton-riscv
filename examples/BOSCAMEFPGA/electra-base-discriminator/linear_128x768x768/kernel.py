"""Standalone Triton source for linear_128x768x768."""

import triton
import triton.language as tl


@triton.jit
def linear(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    """C[M,N] += A[M,K] @ B[N,K]^T, where B is stored as [N,K]."""
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0.0, tl.float32)
    for block in range(tl.cdiv(K, BK)):
        kk = block * BK + k
        a = tl.load(
            A + rows[:, None] * K + kk[None, :],
            (rows[:, None] < M) & (kk[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            B + cols[None, :] * K + kk[:, None],
            (cols[None, :] < N) & (kk[:, None] < K),
            other=0.0,
        )
        acc = tl.dot(a, b, acc, input_precision="ieee")
    valid = (rows[:, None] < M) & (cols[None, :] < N)
    previous = tl.load(C + rows[:, None] * N + cols[None, :], valid, other=0.0)
    tl.store(C + rows[:, None] * N + cols[None, :], acc + previous, valid)

# Fixed specialization; cases.py reads this literal without importing Triton.
KERNEL_CONFIG = {'family': 'linear',
 'kernel': 'linear',
 'arguments': [{'name': 'A', 'rank': 2, 'dtype': 'f32'},
               {'name': 'B', 'rank': 2, 'dtype': 'f32'},
               {'name': 'C', 'rank': 2, 'dtype': 'f32'}],
 'shape': [128, 768, 768],
 'constexprs': {'M': 128, 'N': 768, 'K': 768, 'BM': 16, 'BN': 16, 'BK': 64},
 'grid': [8, 48, 1],
 'roles': [],
 'metadata': {'dtype': 'f32',
              'model': 'google/electra-base-discriminator',
              'seq_len': 128,
              'note': 'Unique encoder shapes at seq_len 128. Twelve layers share these '
                      'bins.'}}
