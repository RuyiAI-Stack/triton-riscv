"""Triton kernel for the standalone signed INT8 AME matmul example.

``matmul_i8`` is the kernel used by the ViT-B/16 W8A8 path; the same body is
in ``../matmul_i8_196x768x768/kernels.py`` (only the signature formatting is
per ruff here).  The operational shape ``[128, 128] @ [1024, 128]^T`` is
fixed in this module and lowered through the BOSCAME NR pipeline.
"""

import triton
import triton.language as tl

# This example is intentionally specialized. Keep its complete shape and tile
# contract here instead of passing a second configuration object from build.py.
M = tl.constexpr(128)
N = tl.constexpr(1024)
K = tl.constexpr(128)
BM = tl.constexpr(64)
BN = tl.constexpr(64)
BK = tl.constexpr(128)


@triton.jit
def matmul_i8(A, B, C):
    """Signed INT8 matrix multiply for the patch6 AME path.

    ``A`` is row-major physical ``[M,K]`` and ``B`` is the physical model
    layout ``[N,K]``. The result is an
    accumulating ``int32`` ``[M,N]`` buffer;
    callers clear it before launch because AME's ``mqma.b.mm`` has +=
    semantics.  Keeping this as a separate Triton entry point prevents an
    accidental FP32 dot from reaching the integer lowering pass.

    The fixed 128-wide K block is lowered into two hardware-sized 64-wide
    AME blocks.
    """
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.int32)
    a = tl.load(A + rows[:, None] * K + k[None, :])
    b = tl.load(B + cols[None, :] * K + k[:, None])
    acc = tl.dot(a, b, acc, input_precision="ieee")
    address = C + rows[:, None] * N + cols[None, :]
    previous = tl.load(address)
    tl.store(address, acc + previous)
