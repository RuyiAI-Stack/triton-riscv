import torch
import triton
import triton.language as tl

GROUP_M = 8

# Decoding recipes accepted in ``recipe_a`` / ``recipe_b``.
RECIPE_TENSOR_WISE = 0
RECIPE_ROW_WISE = 1

# Swizzling patterns accepted in ``swizzle_a`` / ``swizzle_b``.
SWIZZLE_NONE = 0

# Scale granularity as seen by the kernel.
SCALE_TENSOR_WISE = 0
SCALE_ROW_WISE = 1

# The same modes, spelled as Triton compile-time constants.  A global read
# inside a ``@triton.jit`` body has to be an instantiated ``tl.constexpr``.
_SCALE_TENSOR_WISE = tl.constexpr(SCALE_TENSOR_WISE)

_MAT_B_TRANSPOSED_ERROR = "Expected mat2 to be transposed"
_OUT_DTYPE_ERROR = "Unsupported out_dtype for _scaled_mm_v2"
_CONTRACTION_DIM_ERROR = "contraction_dim is only supported for batched operands"


def _heur_even_k(args):
    return args["K"] % args["BLOCK_K"] == 0


@triton.heuristics({"EVEN_K": _heur_even_k})
@triton.jit
def _scaled_mm_v2_kernel(
    A,
    B,
    ScaleA,
    ScaleB,
    Bias,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    SCALE_A_MODE: tl.constexpr,
    SCALE_B_MODE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    """Tiled GEMM with the decoding scales folded into the accumulation.

    ``ScaleA``/``ScaleB`` are read according to ``SCALE_A_MODE``/``SCALE_B_MODE``
    (a single broadcast element for ``TensorWise``, one element per row/column
    for ``RowWise``) and multiply the accumulator before it is written out.
    """
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = offs_m.to(tl.int64)
    offs_n = offs_n.to(tl.int64)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            a = tl.load(a_ptrs, mask=offs_m[:, None] < M, other=0.0)
            b = tl.load(b_ptrs, mask=offs_n[None, :] < N, other=0.0)
        else:
            k_remaining = K - k * BLOCK_K
            a = tl.load(
                a_ptrs,
                mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_remaining),
                other=0.0,
            )
            b = tl.load(
                b_ptrs,
                mask=(offs_k[:, None] < k_remaining) & (offs_n[None, :] < N),
                other=0.0,
            )
        acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if SCALE_A_MODE == _SCALE_TENSOR_WISE:
        scale_a = tl.full((BLOCK_M,), tl.load(ScaleA), dtype=tl.float32)
    else:
        scale_a = tl.load(ScaleA + offs_m, mask=offs_m < M, other=0.0)

    if SCALE_B_MODE == _SCALE_TENSOR_WISE:
        scale_b = tl.full((BLOCK_N,), tl.load(ScaleB), dtype=tl.float32)
    else:
        scale_b = tl.load(ScaleB + offs_n, mask=offs_n < N, other=0.0)

    acc = acc * scale_a[:, None] * scale_b[None, :]

    if HAS_BIAS:
        bias = tl.load(Bias + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        acc += bias[None, :]

    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(C.dtype.element_ty), mask=c_mask)


def _float8_dtypes():
    return tuple(
        getattr(torch, name)
        for name in (
            "float8_e4m3fn",
            "float8_e5m2",
            "float8_e4m3fnuz",
            "float8_e5m2fnuz",
        )
        if hasattr(torch, name)
    )


def _is_float8_dtype(dtype):
    return dtype in _float8_dtypes()


def _parse_recipe(recipe, name):
    """Normalise ``recipe_a``/``recipe_b`` into a single recipe constant."""
    if recipe is None or len(recipe) == 0:
        return RECIPE_ROW_WISE
    if len(recipe) != 1:
        raise ValueError(
            f"{name} must describe exactly one scaling recipe, got {len(recipe)}"
        )
    value = int(recipe[0])
    if value not in (RECIPE_TENSOR_WISE, RECIPE_ROW_WISE):
        raise ValueError(f"unsupported scaling recipe {value} for {name}")
    return value


def _parse_swizzle(swizzle, name):
    """Reject swizzled scale buffers; only a plain layout is supported."""
    if swizzle is None or len(swizzle) == 0:
        return SWIZZLE_NONE
    if len(swizzle) != 1:
        raise ValueError(
            f"{name} must describe exactly one swizzle pattern, got {len(swizzle)}"
        )
    value = int(swizzle[0])
    if value != SWIZZLE_NONE:
        raise ValueError(f"unsupported swizzle pattern {value} for {name}")
    return value


def _parse_scale_list(scales, name):
    if scales is None:
        raise ValueError(f"{name} must be a list of tensors")
    if len(scales) != 1:
        raise ValueError(f"{name} must contain exactly one tensor, got {len(scales)}")
    return scales[0]


def _prepare_scale(scale, *, recipe, length, name):
    """Validate a decoding scale and flatten it into the layout the kernel reads.

    ``TensorWise`` scales hold a single element; ``RowWise`` scales hold one
    factor per row of ``mat_a`` (or column of ``mat_b``) and may be handed over
    either 1D or in the ``(length, 1)`` / ``(1, length)`` shapes the operator
    documents.
    """
    if scale.dtype != torch.float32:
        raise ValueError(f"{name} must be float32, got {scale.dtype}")

    if recipe == RECIPE_TENSOR_WISE:
        if scale.numel() != 1:
            raise ValueError(f"{name} must hold one element, got {scale.numel()}")
        return scale.reshape(1).contiguous()

    if scale.numel() != length:
        raise ValueError(
            f"{name} must have the same length as mat, "
            f"got {scale.numel()} instead of {length}"
        )
    return scale.reshape(length).contiguous()


def _resolve_out_dtype(mat_a, out_dtype):
    if out_dtype is None:
        return mat_a.dtype
    supported = (torch.float32, torch.float16, torch.bfloat16) + _float8_dtypes()
    if out_dtype not in supported:
        raise ValueError(f"{_OUT_DTYPE_ERROR}: {out_dtype}")
    return out_dtype


def _scaled_mm_v2(
    mat_a,
    mat_b,
    scale_a,
    recipe_a,
    swizzle_a,
    scale_b,
    recipe_b,
    swizzle_b,
    bias=None,
    out_dtype=None,
    contraction_dim=None,
    use_fast_accum=False,
):
    if contraction_dim:
        raise ValueError(_CONTRACTION_DIM_ERROR)

    if mat_a.dim() != 2 or mat_b.dim() != 2:
        raise ValueError("_scaled_mm_v2 expects 2D operands")
    if mat_a.dtype != mat_b.dtype:
        raise ValueError(
            f"mat_a and mat_b must have the same dtype, "
            f"got {mat_a.dtype} and {mat_b.dtype}"
        )
    if not _is_float8_dtype(mat_a.dtype):
        raise ValueError(f"expected mat_a to be a float8 matrix, got {mat_a.dtype}")

    m_size, k_size = mat_a.shape
    k_b, n_size = mat_b.shape
    # ``mat_b`` is contracted along its leading dimension, so it has to be
    # handed over in the transposed (column major) layout. This check runs
    # before the K-shape check so a wrongly-oriented mat2 reports the layout
    # error (the wrapper contract), not a spurious shape mismatch.
    if mat_b.stride(0) != 1:
        raise ValueError(_MAT_B_TRANSPOSED_ERROR)
    if k_size != k_b:
        raise ValueError(
            f"mat_a and mat_b shapes cannot be multiplied ({k_size} and {k_b})"
        )
    if mat_a.stride(1) != 1:
        mat_a = mat_a.contiguous()

    recipe_a = _parse_recipe(recipe_a, "recipe_a")
    recipe_b = _parse_recipe(recipe_b, "recipe_b")
    _parse_swizzle(swizzle_a, "swizzle_a")
    _parse_swizzle(swizzle_b, "swizzle_b")

    scale_a = _prepare_scale(
        _parse_scale_list(scale_a, "scale_a"),
        recipe=recipe_a,
        length=m_size,
        name="scale_a",
    )
    scale_b = _prepare_scale(
        _parse_scale_list(scale_b, "scale_b"),
        recipe=recipe_b,
        length=n_size,
        name="scale_b",
    )

    if bias is not None:
        if bias.numel() != n_size:
            raise ValueError(f"bias must have {n_size} elements, got {bias.numel()}")
        bias = bias.reshape(n_size).contiguous()

    out_dtype = _resolve_out_dtype(mat_a, out_dtype)
    out = torch.empty((m_size, n_size), dtype=out_dtype, device=mat_a.device)
    if out.numel() == 0:
        return out

    grid = (triton.cdiv(m_size, 64) * triton.cdiv(n_size, 64),)
    _scaled_mm_v2_kernel[grid](
        mat_a,
        mat_b,
        scale_a,
        scale_b,
        bias,
        out,
        m_size,
        n_size,
        k_size,
        mat_a.stride(0),
        mat_a.stride(1),
        mat_b.stride(0),
        mat_b.stride(1),
        out.stride(0),
        out.stride(1),
        SCALE_A_MODE=(
            SCALE_TENSOR_WISE if recipe_a == RECIPE_TENSOR_WISE else SCALE_ROW_WISE
        ),
        SCALE_B_MODE=(
            SCALE_TENSOR_WISE if recipe_b == RECIPE_TENSOR_WISE else SCALE_ROW_WISE
        ),
        HAS_BIAS=bias is not None,
        GROUP_M=GROUP_M,
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K=64,
    )
    return out
