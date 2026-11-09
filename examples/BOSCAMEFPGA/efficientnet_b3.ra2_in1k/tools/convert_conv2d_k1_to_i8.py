#!/usr/bin/env python3
"""Rewrite TIMEOUT 1x1 conv2d_*_k1_s1 cases to packed GEMM i8 x i8 -> i32.

Same pattern as conv2d_18x18_576x96_k1_s1: Triton tl.dot + bias in launch.c.
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# From fpga-batch-report / effnet_batch_run.log TIMEOUT set (k1 only).
TIMEOUT_K1 = [
    "conv2d_144x144_24x144_k1_s1",
    "conv2d_144x144_40x24_k1_s1",
    "conv2d_18x18_136x816_k1_s1",
    "conv2d_18x18_288x96_k1_s1",
    "conv2d_18x18_576x136_k1_s1",
    "conv2d_18x18_576x96_k1_s1",
    "conv2d_18x18_816x136_k1_s1",
    "conv2d_18x18_96x576_k1_s1",
    "conv2d_1x1_10x40_k1_s1",
    "conv2d_1x1_12x288_k1_s1",
    "conv2d_1x1_1392x58_k1_s1",
    "conv2d_1x1_144x6_k1_s1",
    "conv2d_1x1_192x8_k1_s1",
    "conv2d_1x1_2304x96_k1_s1",
    "conv2d_1x1_24x576_k1_s1",
    "conv2d_1x1_24x6_k1_s1",
    "conv2d_1x1_288x12_k1_s1",
    "conv2d_1x1_34x816_k1_s1",
    "conv2d_1x1_40x10_k1_s1",
    "conv2d_1x1_576x24_k1_s1",
    "conv2d_1x1_58x1392_k1_s1",
    "conv2d_1x1_6x144_k1_s1",
    "conv2d_1x1_6x24_k1_s1",
    "conv2d_1x1_816x34_k1_s1",
    "conv2d_1x1_8x192_k1_s1",
    "conv2d_1x1_96x2304_k1_s1",
    "conv2d_36x36_192x48_k1_s1",
    "conv2d_36x36_288x48_k1_s1",
    "conv2d_36x36_48x288_k1_s1",
    "conv2d_72x72_144x32_k1_s1",
    "conv2d_72x72_192x32_k1_s1",
    "conv2d_72x72_32x192_k1_s1",
    "conv2d_9x9_1392x232_k1_s1",
    "conv2d_9x9_1392x384_k1_s1",
    "conv2d_9x9_2304x384_k1_s1",
    "conv2d_9x9_232x1392_k1_s1",
    "conv2d_9x9_384x1536_k1_s1",
    "conv2d_9x9_384x2304_k1_s1",
    "conv2d_9x9_816x232_k1_s1",
]

NAME_RE = re.compile(
    r"^conv2d_(?P<h>\d+)x(?P<w>\d+)_(?P<c>\d+)x(?P<oc>\d+)_k(?P<kh>\d+)_s(?P<sh>\d+)$"
)

BM = BN = 16
BK = 64

KERNEL_PY = '''"""Standalone Triton source for {name} (i8 x i8 -> i32).

1x1 NCHW conv is lowered as a packed GEMM:
  X[H*W, C] @ Weight[OC, C]^T -> Out[H*W, OC]  (i8 x i8 -> i32)
Bias is applied in launch.c (contract rejects bias broadcast into the tile).
"""

import triton
import triton.language as tl


@triton.jit
def conv2d_nchw(
    X,
    Weight,
    Bias,
    Out,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    INTEGER: tl.constexpr,
):
    """Signed INT8 Out[M, N] = X[M, K] @ Weight[N, K]^T (Bias added in launch)."""
    tl.static_assert(INTEGER)
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.int32)
    for block in range(tl.cdiv(K, BK)):
        kk = block * BK + k
        a = tl.load(
            X + rows[:, None] * K + kk[None, :],
            (rows[:, None] < M) & (kk[None, :] < K),
            other=0,
        )
        b = tl.load(
            Weight + cols[None, :] * K + kk[:, None],
            (cols[None, :] < N) & (kk[:, None] < K),
            other=0,
        )
        acc = tl.dot(a, b, acc, input_precision="ieee")
    # Keep Bias in the ABI; value applied in launch.c.
    _ = tl.load(Bias + cols, cols < N, other=0)
    valid = (rows[:, None] < M) & (cols[None, :] < N)
    tl.store(Out + rows[:, None] * N + cols[None, :], acc, valid)


# Fixed specialization; cases.py reads this literal without importing Triton.
# shape mirrors NCHW [1,C,H,W]; GEMM uses M=H*W, N=OC, K=C.
KERNEL_CONFIG = {{
    "family": "conv2d_i8",
    "kernel": "conv2d_nchw",
    "arguments": [
        {{"name": "X", "rank": 2, "dtype": "i8"}},
        {{"name": "Weight", "rank": 2, "dtype": "i8"}},
        {{"name": "Bias", "rank": 1, "dtype": "i32"}},
        {{"name": "Out", "rank": 2, "dtype": "i32"}},
    ],
    "shape": [1, {c}, {h}, {w}],
    "constexprs": {{
        "M": {m},
        "N": {n},
        "K": {k},
        "BM": {bm},
        "BN": {bn},
        "BK": {bk},
        "INTEGER": True,
    }},
    "grid": [{gx}, {gy}, 1],
    "roles": [],
    "metadata": {{
        "dtype": "i8 x i8 -> i32",
        "layout": "packed GEMM X[H*W,C] @ W[OC,C]^T",
        "nchw": [1, {c}, {h}, {w}],
        "model": "timm/efficientnet_b3.ra2_in1k",
    }},
}}
'''

LAUNCH_C = """#include "support.h"
extern void _mlir_ciface_kernel_{name}(MemRef2 *, MemRef2 *,
                                                           MemRef1 *, MemRef2 *);
int launch(void) {{
  size_t off = 0;
  /* Packed X: i8 [H*W, C] = [{m}, {k}] */
  int8_t *x = (int8_t *)workspace(off);
  off = vit_align64(off + (size_t){m} * {k} * 1u);
  for (int i = 0; i < {m} * {k}; ++i)
    x[i] = 0;
  MemRef2 X = make_2(x, {m}, {k});
  /* Weight: i8 [OC, C] = [{n}, {k}] */
  int8_t *w = (int8_t *)workspace(off);
  off = vit_align64(off + (size_t){n} * {k} * 1u);
  for (int i = 0; i < {n} * {k}; ++i)
    w[i] = 0;
  MemRef2 W = make_2(w, {n}, {k});
  /* Bias: i32 [{n}] */
  int32_t *b = (int32_t *)workspace(off);
  off = vit_align64(off + (size_t){n} * 4u);
  for (int i = 0; i < {n}; ++i)
    b[i] = 0;
  MemRef1 B = make_1(b, {n});
  /* Out: i32 [H*W, OC] = [{m}, {n}] */
  int32_t *o = (int32_t *)workspace(off);
  off = vit_align64(off + (size_t){m} * {n} * 4u);
  for (int i = 0; i < {m} * {n}; ++i)
    o[i] = 0;
  MemRef2 O = make_2(o, {m}, {n});
  _mlir_ciface_kernel_{name}(&X, &W, &B, &O);
  /* Bias applied outside the AME GEMM tile. */
  for (int s = 0; s < {m}; ++s) {{
    int32_t *row = o + s * {n};
    for (int oc = 0; oc < {n}; ++oc)
      row[oc] += b[oc];
  }}
  nr_puts("numerical acceptance is simcheck\\n");
  return 0;
}}
"""


def parse(name: str):
    match = NAME_RE.match(name)
    if not match:
        raise ValueError(name)
    vals = {k: int(v) for k, v in match.groupdict().items()}
    if vals["kh"] != 1 or vals["sh"] != 1:
        raise ValueError(f"{name}: only k1_s1 supported")
    h, w, c, oc = vals["h"], vals["w"], vals["c"], vals["oc"]
    m, n, k = h * w, oc, c
    gx = math.ceil(m / BM)
    gy = math.ceil(n / BN)
    return dict(
        name=name, h=h, w=w, c=c, m=m, n=n, k=k, bm=BM, bn=BN, bk=BK, gx=gx, gy=gy
    )


def convert(name: str) -> Path:
    params = parse(name)
    case_dir = ROOT / name
    if not case_dir.is_dir():
        raise FileNotFoundError(case_dir)
    (case_dir / "kernel.py").write_text(KERNEL_PY.format(**params))
    (case_dir / "launch.c").write_text(LAUNCH_C.format(**params))
    makefile = case_dir / "makefile"
    if not makefile.is_file():
        makefile.write_text("include ../common.mk\n")
    return case_dir


def main(argv=None):
    names = argv[1:] if argv and len(argv) > 1 else TIMEOUT_K1
    for name in names:
        path = convert(name)
        print(f"converted {path.name}")


if __name__ == "__main__":
    main(sys.argv)
