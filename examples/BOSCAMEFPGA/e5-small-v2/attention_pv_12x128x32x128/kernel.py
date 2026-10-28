"""Standalone Triton source for attention_pv_12x128x32x128."""

import triton
import triton.language as tl


@triton.jit
def attention_dot(
    A,
    B,
    C,
    HEADS: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SCALE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    """Batched head matmul for QK^T and attention-probability V."""
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    head = tl.program_id(2)
    k = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0.0, tl.float32)
    for block in range(tl.cdiv(K, BK)):
        kk = block * BK + k
        a = tl.load(
            A + head * M * K + rows[:, None] * K + kk[None, :],
            (rows[:, None] < M) & (kk[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            B + head * K * N + kk[:, None] * N + cols[None, :],
            (kk[:, None] < K) & (cols[None, :] < N),
            other=0.0,
        )
        acc = tl.dot(a, b, acc, input_precision="ieee")
    valid = (rows[:, None] < M) & (cols[None, :] < N)
    tl.store(
        C + head * M * N + rows[:, None] * N + cols[None, :], acc * SCALE, valid
    )

# Fixed specialization; cases.py reads this literal without importing Triton.
KERNEL_CONFIG = {'family': 'attention_pv',
 'kernel': 'attention_dot',
 'arguments': [{'name': 'A', 'rank': 3, 'dtype': 'f32'},
               {'name': 'B', 'rank': 3, 'dtype': 'f32'},
               {'name': 'C', 'rank': 3, 'dtype': 'f32'}],
 'shape': [12, 128, 32, 128],
 'constexprs': {'HEADS': 12,
                'M': 128,
                'N': 32,
                'K': 128,
                'BM': 16,
                'BN': 16,
                'BK': 64,
                'SCALE': 1.0},
 'grid': [8, 2, 12],
 'roles': [],
 'metadata': {'dtype': 'f32',
              'model': 'intfloat/e5-small-v2',
              'seq_len': 128,
              'note': 'Unique encoder shapes at seq_len 128. Twelve layers share these '
                      'bins.'}}
