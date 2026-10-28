"""Standalone Triton source for linear_m1_n1000_k1024."""

import triton
import triton.language as tl

@triton.jit
def linear(X, Weight, Bias, Out, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    """Out[1,N] = X[1,K] @ Weight[N,K]^T + Bias[N]. One program owns one output."""
    tl.static_assert(M == 1)
    col = tl.program_id(0)
    acc = 0.0
    for q in range(K):
        acc += tl.load(X + q) * tl.load(Weight + col * K + q)
    tl.store(Out + col, acc + tl.load(Bias + col))

# Fixed specialization; cases.py reads this literal without importing Triton.
KERNEL_CONFIG = {'family': 'linear',
 'kernel': 'linear',
 'arguments': [{'name': 'X', 'rank': 2, 'dtype': 'f32'},
               {'name': 'Weight', 'rank': 2, 'dtype': 'f32'},
               {'name': 'Bias', 'rank': 1, 'dtype': 'f32'},
               {'name': 'Out', 'rank': 2, 'dtype': 'f32'}],
 'shape': [1, 1024],
 'constexprs': {'M': 1, 'N': 1000, 'K': 1024},
 'grid': [1000, 1, 1],
 'roles': []}
