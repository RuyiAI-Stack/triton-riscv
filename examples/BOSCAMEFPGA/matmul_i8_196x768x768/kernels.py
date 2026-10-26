"""Triton kernel for the standalone signed INT8 AME matmul example.

``matmul_i8`` is the kernel used by the ViT-B/16 W8A8 path
(``examples/FPGA-BOSCAME/vit-base-patch16-224/triton/kernels.py``).  The
operational shape ``[196, 768] @ [768, 768]^T`` is fixed in this module
together with the AME tile contract ``BM=32``, ``BN=128``, ``BK=128``; the
launch grid is derived from them by :mod:`build`.

The AME register file caps one tile dimension at 64, so ``BN=128`` is a
*logical* program tile: the BOSCAME planner lowers it as two 64-column AME
groups, and ``BK=128`` as two 64-wide AME K blocks.  M=196 is not a multiple of
BM=32 and N/K need not match the tiles exactly, so the kernel keeps its
boundary masks (rows past ``M``, K lanes past ``K``) and iterates over the
``K/BK`` reduction blocks.

The external triton-riscv AOT exporter normalizes and bufferizes this source.
Buddy verifies its standard Contract, plans storage, verifies the plan and
emits BOSCAME before LLVM lowering, as in ``../matmul_i8_128x1024x128``.
"""

import triton
import triton.language as tl

M = tl.constexpr(196)
N = tl.constexpr(768)
K = tl.constexpr(768)
BM = tl.constexpr(32)
BN = tl.constexpr(128)
BK = tl.constexpr(128)


@triton.jit
def matmul_i8(A, B, C):
    """Signed INT8 matrix multiply for the patch6 AME path.

    ``A`` is row-major physical ``[M,K]`` and ``B`` is the physical model
    layout ``[N,K]``.  ``C`` is the accumulating ``int32`` ``[M,N]`` buffer:
    its current contents are loaded into the accumulator *before* the reduction
    (AME's ``mqma.b.mm`` has ``+=`` semantics) and stored back once afterwards,
    so C is read and written exactly once per program.  Masked rows/lanes are
    initialized from ``other=0``, so the partial boundary tile contributes
    zeros exactly where no logical element exists and the masked store leaves
    excluded C elements untouched.
    """
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    valid = (rows[:, None] < M) & (cols[None, :] < N)
    address = C + rows[:, None] * N + cols[None, :]
    acc = tl.load(address, valid, other=0)
    for block in range(tl.cdiv(K, BK)):
        kk = block * BK + k
        valid_a = (rows[:, None] < M) & (kk[None, :] < K)
        valid_b = (cols[None, :] < N) & (kk[:, None] < K)
        a = tl.load(A + rows[:, None] * K + kk[None, :], valid_a, other=0)
        b = tl.load(B + cols[None, :] * K + kk[:, None], valid_b, other=0)
        acc = tl.dot(a, b, acc, input_precision="ieee")
    tl.store(address, acc, valid)
