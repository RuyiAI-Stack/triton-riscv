"""Triton implementation of aten::tensordot.

Ported from upstream FlagGems ops/tensordot.py.

Contracts the axes ``dims_self`` of ``self`` against ``dims_other`` of
``other`` (aten semantics: a size-1 contracted axis on either side
broadcasts to the other side's extent), permutes the free + contracted
axes into a (M, K) x (K, N) pair and multiplies them with the tiled
``tensordot_mm_kernel``.

Porting notes (repo conventions):
  - @libentry/@libtuner stripped (plain @triton.jit); the shared "mm" tune
    table is replaced by a fixed output tile table (16/16 for small sides,
    64/64 otherwise; the FMA accumulation needs no K tile); the unused
    SPLIT_K placeholder the tuner needed is dropped; no torch device
    context manager; ``ext.`` -> ``tl.``; grid is a plain tuple.
  - Accumulation is a sequential scalar-FMA loop over K for every dtype
    (fp64 natively; everything else cast to fp32 first): probe
    /tmp/b15probe/sb1/probe3.py shows that order reproduces aten CPU fp32
    and fp64 matmul bit-for-bit on the pinned exactness data, while tl.dot
    rounds differently by 1 ulp and the backend's fp64 tl.dot carries
    ~fp32 accumulator precision (house precedent: inner.py).
  - ``get_higher_dtype`` is not ported: dtype equality is validated before
    the launch, so the output dtype is the input dtype.
  - The autograd wrapper (TensordotFunction) is not ported: no test
    exercises requires_grad inputs (house precedent: ger.py, addmm.py).
  - fp64 is in the supported dtype set: the CPU backend supports fp64
    through the FMA path above (upstream queried runtime.device).
"""

import math
import warnings

import torch
import triton
import triton.language as tl


@triton.jit
def tensordot_mm_kernel(
    A,
    B,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
    IS_FP64: tl.constexpr,
):
    # Tiled matrix multiplication C = A @ B, where A is (M, K) and B is (K, N).
    # The contracted axes of tensordot are collapsed into K before launch.
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    # re-order program ID for better L2 performance
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m[:, None] < M
    mask_n = offs_n[None, :] < N

    # Accumulate with scalar FMA over K, sequential like aten CPU matmul:
    # probe (/tmp/b15probe/sb1/probe3.py) shows that order reproduces aten's
    # fp32 and fp64 mm bit-for-bit on the pinned exactness data, while tl.dot
    # rounds differently by 1 ulp.
    acc_dtype = tl.float64 if IS_FP64 else tl.float32
    a_vec = A + offs_m.to(tl.int64) * stride_am
    b_vec = B + offs_n.to(tl.int64) * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype)
    for kk in range(K):
        a = tl.load(a_vec + kk * stride_ak, mask=offs_m < M, other=0.0).to(acc_dtype)
        b = tl.load(b_vec + kk * stride_bk, mask=offs_n < N, other=0.0).to(acc_dtype)
        acc = tl.fma(a[:, None], b[None, :], acc)

    offs_m64 = offs_m.to(tl.int64)
    offs_n64 = offs_n.to(tl.int64)
    c_ptrs = C + (offs_m64[:, None] * stride_cm + offs_n64[None, :] * stride_cn)
    tl.store(c_ptrs, acc.to(C.dtype.element_ty), mask=mask_m & mask_n)


def _select_block_config(M, N):
    """Pick output tile sizes (the fixed table)."""
    if M < 16 or N < 16:
        return 16, 16
    return 64, 64


def _matmul_2d(a, b, out=None):
    # Contract two 2D tensors (M, K) x (K, N) -> (M, N) with the Triton kernel.
    a = a.contiguous()
    b = b.contiguous()
    M, K = a.shape
    _, N = b.shape
    c_dtype = a.dtype
    if out is None:
        out = torch.empty((M, N), device=a.device, dtype=c_dtype)

    block_m, block_n = _select_block_config(M, N)
    grid = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n),)
    tensordot_mm_kernel[grid](
        a,
        b,
        out,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        GROUP_M=8,
        IS_FP64=a.dtype == torch.float64,
    )
    return out


def _normalize_dims(dims, ndim):
    # Map (possibly negative) dim indices into the range [0, ndim).
    # Raise error for out-of-range dimensions.
    dims = [dims] if isinstance(dims, int) else list(dims)
    normalized = []
    for d in dims:
        if d < -ndim or d >= ndim:
            raise IndexError(
                f"Dimension out of range (expected to be in range of [{-ndim}, {ndim - 1}], but got {d})"
            )
        normalized.append(d if d >= 0 else d + ndim)
    return normalized


def _resolve_contraction(self, other, dims_self, dims_other):
    # Normalize contracted dims and reconcile their sizes following the ATen
    # tensordot contract: a contracted axis may be size 1 on either side, in
    # which case it is broadcast (and later reduced) rather than rejected. Only
    # unequal, non-1 sizes are an error. Returns normalized dim lists together
    # with the (possibly expanded) tensors that share matching contracted sizes.
    dims_self = _normalize_dims(dims_self, self.ndim)
    dims_other = _normalize_dims(dims_other, other.ndim)

    if len(dims_self) != len(dims_other):
        raise RuntimeError(
            "tensordot: number of contracted dims must match, but got "
            f"{len(dims_self)} and {len(dims_other)}"
        )

    self_shape = list(self.shape)
    other_shape = list(other.shape)
    for ds, do in zip(dims_self, dims_other):
        s1 = self.shape[ds]
        s2 = other.shape[do]
        if s1 == s2:
            continue
        if s1 == 1:
            self_shape[ds] = s2
        elif s2 == 1:
            other_shape[do] = s1
        else:
            raise RuntimeError(
                "tensordot: contracted dimensions need to match, but got "
                f"size {s1} in dim {ds} of self and size {s2} in dim {do} of "
                "other"
            )

    self_e = self.expand(self_shape) if self_shape != list(self.shape) else self
    other_e = other.expand(other_shape) if other_shape != list(other.shape) else other
    return dims_self, dims_other, self_e, other_e


def _copy_tensordot_out(result: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Validate, resize, and copy a computed tensordot result into an out tensor."""
    if out.device != result.device:
        raise RuntimeError(
            "tensordot: Expected the output tensor to be on the same device as "
            f"the result ({result.device}), but got {out.device}"
        )
    if out.dtype != result.dtype:
        raise RuntimeError(
            f"tensordot: Expected the output tensor to have dtype {result.dtype}, "
            f"but got {out.dtype}"
        )
    if tuple(out.shape) != tuple(result.shape):
        if out.numel() != 0:
            warnings.warn(
                "An output with one or more elements was resized since it had "
                f"shape {list(out.shape)}, which does not match the required "
                f"output shape {list(result.shape)}. This behavior is deprecated, "
                "and in a future PyTorch release outputs will not be resized "
                "unless they have zero elements. You can explicitly reuse an out "
                "tensor t by resizing it, inplace, to zero elements with "
                "t.resize_(0).",
                UserWarning,
                stacklevel=3,
            )
        out.resize_(result.shape)
    out.copy_(result)
    return out


def _tensordot_impl(self, other, dims_self, dims_other, out=None):
    # Validate supported dtypes - only floating point types
    supported_dtypes = {torch.float16, torch.float32, torch.bfloat16, torch.float64}

    if self.dtype not in supported_dtypes:
        raise RuntimeError(
            f"tensordot: unsupported dtype {self.dtype}. "
            f"Only floating point dtypes are supported."
        )

    if other.dtype not in supported_dtypes:
        raise RuntimeError(
            f"tensordot: unsupported dtype {other.dtype}. "
            f"Only floating point dtypes are supported."
        )

    # Validate dtype and device match
    if self.dtype != other.dtype:
        raise RuntimeError(
            f"tensordot: expected self and other to have the same dtype, "
            f"but got {self.dtype} and {other.dtype}"
        )
    if self.device != other.device:
        raise RuntimeError(
            f"tensordot: expected self and other to be on the same device, "
            f"but got {self.device} and {other.device}"
        )

    dims_self, dims_other, self, other = _resolve_contraction(
        self, other, dims_self, dims_other
    )

    # Free (non-contracted) dims, preserving their original order.
    free_self = [d for d in range(self.ndim) if d not in dims_self]
    free_other = [d for d in range(other.ndim) if d not in dims_other]

    free_self_sizes = [self.shape[d] for d in free_self]
    free_other_sizes = [other.shape[d] for d in free_other]

    M = math.prod(free_self_sizes) if free_self_sizes else 1
    N = math.prod(free_other_sizes) if free_other_sizes else 1
    K = math.prod([self.shape[d] for d in dims_self]) if dims_self else 1

    # Reorder so contracted dims are collapsed into a single axis:
    #   self  -> (free_self..., contracted...)  reshaped to (M, K)
    #   other -> (contracted..., free_other...) reshaped to (K, N)
    a2d = self.permute(free_self + dims_self).contiguous().reshape(M, K)
    b2d = other.permute(dims_other + free_other).contiguous().reshape(K, N)

    result_shape = free_self_sizes + free_other_sizes

    if out is not None:
        # Compute result and copy it into out without reshaping out itself, so a
        # non-contiguous out never creates a temporary tensor. resize_ + copy_
        # matches the aten out= contract (resize when the shape differs, raise on
        # dtype/device mismatch).
        res = _matmul_2d(a2d, b2d).reshape(result_shape)
        return _copy_tensordot_out(res, out)

    res = _matmul_2d(a2d, b2d)
    return res.reshape(result_shape)


def tensordot(self, other, dims_self, dims_other):
    return _tensordot_impl(self, other, dims_self, dims_other)


def tensordot_out(self, other, dims_self, dims_other, *, out):
    _tensordot_impl(self, other, dims_self, dims_other, out=out)
    return out
