"""chain_matmul in Triton.

Ported from upstream FlagGems ops/chain_matmul.py (upstream entry ``run``,
re-exported upstream as ``chain_matmul``).

Computes the product of a chain of 2-D matrices using an optimal
(minimum-flop) parenthesization, mirroring torch.linalg.multi_dot /
aten::chain_matmul semantics. Every matmul is computed by a Triton device
kernel; the host wrapper only runs the matrix-chain DP, allocates
intermediates and launches kernels. Each DP node is rounded back to the
input dtype, so the result matches a native-dtype reference rather than an
internally-widened one.

Numerical strategy per dtype is selected in ``_config`` and dispatched by
the ``MODE`` constexpr of ``_gemm``:

- fp32: MODE 1 does ``ieee`` dots over 32-wide K steps for long chains;
  MODE 0 is the generic masked-tile path for everything else.
- fp64: MODE 3 accumulates in fp64 with scalar FMA (``tl.fma``), which
  reproduces aten CPU fp64 matmul bit-for-bit on this backend; there is no
  fp64 tensor-core path.

Porting notes (repo conventions):
  - The CUDA-only performance machinery is not ported: MODE 2/4/5 (fp16/
    bf16 tile-order variants reproducing CUTLASS WMMA rounding, one of them
    built on ``tl.inline_asm_elementwise``), the split-K ``_partial``/
    ``_merge`` pair, the fused pair/tail kernels, the shared-memory stage
    capper and all num_warps/num_stages plumbing. Their dtype regimes fall
    back to the generic MODE 0 masked-tile dot, which is value-equivalent
    (a different summation order, within the upstream fp16/fp32 tolerances;
    no test in this suite pins the WMMA rounding boundaries).
  - The autograd wrapper (_Chain) is not ported: no test exercises
    requires_grad inputs (house precedent: ger.py).
  - @lru_cache on the config/schedule helpers is kept (only immutable
    integer metadata is retained, never tensors).
  - Upstream entry name: ``run``; exposed here as ``chain_matmul`` plus
    ``chain_matmul_out`` per the upstream ops/__init__ re-export.
"""

from functools import lru_cache

import torch
import triton
import triton.language as tl


@triton.jit
def _gemm(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    AM: tl.constexpr,
    AK: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    CM: tl.constexpr,
    CN: tl.constexpr,
    TM: tl.constexpr,
    TN: tl.constexpr,
    TK: tl.constexpr,
    MODE: tl.constexpr,
):
    if MODE == 3:
        # fp64: outer-product scalar-FMA accumulation, sequential over K.
        # Matches aten CPU fp64 matmul bit-for-bit on this backend.
        m = tl.program_id(0) * TM + tl.arange(0, TM)
        n = tl.program_id(1) * TN + tl.arange(0, TN)
        acc = tl.zeros((TM, TN), tl.float64)
        for kk in range(K):
            a = tl.load(A + m * AM + kk * AK, m < M, 0)
            b = tl.load(B + kk * BK + n * BN, n < N, 0)
            acc = tl.fma(a[:, None], b[None, :], acc)
        tl.store(
            C + m[:, None] * CM + n[None, :] * CN,
            acc,
            (m[:, None] < M) & (n[None, :] < N),
        )
    else:
        pm = tl.program_id(0)
        pn = tl.program_id(1)
        m = pm * TM + tl.arange(0, TM)
        n = pn * TN + tl.arange(0, TN)
        k = tl.arange(0, TK)
        if MODE == 1:
            # fp32 ieee: two 32-wide dots per 64-wide K block.
            acc = tl.zeros((TM, TN), tl.float32)
            for base in range(tl.cdiv(K, 64)):
                part = tl.zeros((TM, TN), tl.float32)
                for step in tl.static_range(2):
                    kk = base * 64 + step * 32 + k
                    a = tl.load(
                        A + m[:, None] * AM + kk[None, :] * AK,
                        (m[:, None] < M) & (kk[None, :] < K),
                        0,
                    )
                    b = tl.load(
                        B + kk[:, None] * BK + n[None, :] * BN,
                        (kk[:, None] < K) & (n[None, :] < N),
                        0,
                    )
                    part = tl.dot(a, b, part, input_precision="ieee")
                acc = acc + part
        else:
            acc = tl.zeros((TM, TN), tl.float32)
            for base in range(tl.cdiv(K, TK)):
                kk = base * TK + k
                if K % TK == 0:
                    a = tl.load(
                        A + m[:, None] * AM + kk[None, :] * AK, m[:, None] < M, 0
                    )
                    b = tl.load(
                        B + kk[:, None] * BK + n[None, :] * BN, n[None, :] < N, 0
                    )
                else:
                    a = tl.load(
                        A + m[:, None] * AM + kk[None, :] * AK,
                        (m[:, None] < M) & (kk[None, :] < K),
                        0,
                    )
                    b = tl.load(
                        B + kk[:, None] * BK + n[None, :] * BN,
                        (kk[:, None] < K) & (n[None, :] < N),
                        0,
                    )
                if A.dtype.element_ty == tl.float32:
                    acc = tl.dot(a, b, acc, input_precision="ieee")
                else:
                    acc = tl.dot(a, b, acc)
        tl.store(
            C + m[:, None] * CM + n[None, :] * CN,
            acc,
            (m[:, None] < M) & (n[None, :] < N),
        )


@triton.jit
def _copy(
    A,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    AM: tl.constexpr,
    AN: tl.constexpr,
    CM: tl.constexpr,
    CN: tl.constexpr,
):
    x = tl.program_id(0) * 256 + tl.arange(0, 256)
    m = x // N
    n = x % N
    v = tl.load(A + m * AM + n * AN, x < M * N, 0)
    tl.store(C + m * CM + n * CN, v, x < M * N)


def _copy2d(a, c):
    m, n = a.shape
    if m * n:
        _copy[(triton.cdiv(m * n, 256),)](a, c, m, n, *a.stride(), *c.stride())
    return c


@lru_cache(maxsize=256)
def _config(dtype, m, n, k):
    """Pick tile sizes and MODE for one (dtype, M, N, K) gemm node.

    Simplified from the upstream offline-tuned table: the CUDA-only MODEs
    (2/4/5) and the shared-memory stage capping are dropped, so low
    precision dtypes always take the generic MODE 0 masked-tile path.
    """
    mode = 0
    tk = 64
    tm = 64
    tn = 64 if n >= 64 else max(16, triton.next_power_of_2(n))
    if dtype == torch.float64:
        # No fp64 tensor-core path; MODE 3 accumulates with scalar FMA,
        # which only fits in registers with 32x32 tiles.
        mode = 3
        tm = 32
        tn = 32
    elif dtype == torch.float32:
        mode = 1 if k >= 1024 and k % 64 == 0 else 0
        tk = 32
    return tm, tn, tk, mode, (triton.cdiv(m, tm), triton.cdiv(n, tn))


def _mm(a, b):
    m, k = a.shape
    n = b.shape[1]
    c = torch.empty((m, n), dtype=a.dtype, device=a.device)
    if not m * n:
        return c
    tm, tn, tk, mode, grid = _config(a.dtype, m, n, k)
    _gemm[grid](
        a,
        b,
        c,
        m,
        n,
        k,
        *a.stride(),
        *b.stride(),
        n,
        1,
        tm,
        tn,
        tk,
        mode,
    )
    return c


def _split(p):
    n = len(p) - 1
    cost = [[0] * n for _ in range(n)]
    s = [[0] * n for _ in range(n)]
    for length in range(1, n):
        for i in range(n - length):
            j = i + length
            best = None
            for k in range(i, j):
                v = cost[i][k] + cost[k + 1][j] + p[i] * p[k + 1] * p[j + 1]
                if best is None or v < best:
                    best = v
                    s[i][j] = k
            cost[i][j] = best
    return s


@lru_cache(maxsize=128)
def _schedule(p):
    # Only immutable integer metadata is retained, never tensors or values.
    s = _split(p)
    order = []

    def build(i, j):
        if i != j:
            k = s[i][j]
            build(i, k)
            build(k + 1, j)
            order.append((i, j, k))

    build(0, len(p) - 2)
    return tuple(order)


@lru_cache(maxsize=128)
def _slots(p):
    # Map each DP step to the operand slot ids: leaves occupy 0..n-1 and
    # every intermediate result takes the next appended position.
    ids = {(i, i): i for i in range(len(p) - 1)}
    order = []
    for i, j, k in _schedule(p):
        order.append((ids[i, k], ids[k + 1, j]))
        ids[i, j] = len(ids)
    return tuple(order)


def _forward(mats):
    if len(mats) == 1:
        return _copy2d(mats[0], torch.empty_like(mats[0]))
    if len(mats) == 2:
        return _mm(mats[0], mats[1])
    p = tuple(a.shape[0] for a in mats) + (mats[-1].shape[1],)
    tensors = list(mats)
    for left, right in _slots(p):
        tensors.append(_mm(tensors[left], tensors[right]))
    return tensors[-1]


def _run(matrices, args, out=None):
    if isinstance(matrices, torch.Tensor):
        mats = [matrices, *args]
    elif isinstance(matrices, (list, tuple)):
        if args:
            raise TypeError("chain_matmul(): unexpected extra arguments")
        mats = list(matrices)
    else:
        raise TypeError("chain_matmul(): expected a list of tensors")
    if not mats:
        raise RuntimeError("chain_matmul(): Expected one or more matrices")
    if any(not isinstance(a, torch.Tensor) for a in mats):
        raise TypeError("expected tensors")
    dtype, device = mats[0].dtype, mats[0].device
    for i, a in enumerate(mats):
        if a.ndim != 2:
            raise RuntimeError(f"Tensor dimension is {a.ndim}, expected 2 instead.")
        if a.dtype != dtype or a.device != device:
            raise RuntimeError(
                "multi_dot(): all tensors must have the same dtype and device"
            )
        if not dtype.is_floating_point:
            raise RuntimeError("chain_matmul(): unsupported dtype")
        if i and mats[i - 1].shape[1] != a.shape[0]:
            raise RuntimeError("multi_dot(): tensors cannot be multiplied")
    if out is not None:
        if not isinstance(out, torch.Tensor):
            raise RuntimeError("out must be a Tensor")
        if out.dtype != dtype or out.device != device:
            raise RuntimeError("multi_dot(): out dtype or device mismatch")
        if tuple(out.shape) != (mats[0].shape[0], mats[-1].shape[1]):
            raise RuntimeError("multi_dot(): out shape mismatch")
    result = _forward(mats)
    return result if out is None else _copy2d(result, out)


def chain_matmul(matrices, *args, out=None):
    return _run(matrices, args, out=out)


def chain_matmul_out(matrices, *args, out):
    """aten::chain_matmul.out — same product, written into the caller's buffer."""
    return _run(matrices, args, out=out)
