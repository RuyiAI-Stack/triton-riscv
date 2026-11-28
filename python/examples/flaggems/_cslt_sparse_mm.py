import torch
import triton
import triton.language as tl


@triton.jit
def _decompress_kernel(
    values_ptr,  # fp16/bf16 [M, K//2] row-major nonzeros
    meta_ptr,  # int32 view of metadata nibbles packed as uint16 words
    out_ptr,  # dense output [M, K], pre-zeroed
    M,
    K,
    N_KT,  # number of 8-group K-tiles = ceil((K/4)/8)
    stride_vm,
    stride_vk,
    stride_om,
    stride_ok,
    BLOCK_G: tl.constexpr,  # groups processed per program along K
):
    # One program handles one row and a block of column-groups.
    row = tl.program_id(0)
    g0 = tl.program_id(1) * BLOCK_G
    goff = g0 + tl.arange(0, BLOCK_G)
    gpr = K // 4
    gmask = goff < gpr

    # Swizzle: nibble_index for (row, group). See module docstring.
    rt = row // 32
    ri = row % 32
    gt = goff // 8
    gi = goff % 8
    col = (gi % 4) * 4 + tl.where(gi >= 4, 2, 0)
    within = (ri % 8) * 16 + ((ri // 8) % 2) + ((ri // 8) // 2) * 128 + col
    tile_base = (rt % 2) * 256 + gt * 512 + (rt // 2) * (N_KT * 512)
    nib_idx = tile_base + within

    # metadata is a flat uint16 array; 4 nibbles per uint16 word.
    word_idx = nib_idx // 4
    nib_in_word = nib_idx % 4
    word = tl.load(meta_ptr + word_idx, mask=gmask, other=0)
    nib = (word >> (nib_in_word * 4)) & 0xF
    idx0 = nib & 0x3
    idx1 = (nib >> 2) & 0x3

    # Two nonzero values per group, consumed row-major from the values section.
    v0 = tl.load(
        values_ptr + row * stride_vm + (goff * 2) * stride_vk, mask=gmask, other=0.0
    )
    v1 = tl.load(
        values_ptr + row * stride_vm + (goff * 2 + 1) * stride_vk,
        mask=gmask,
        other=0.0,
    )

    # Backend constraint: store addresses must stay purely affine; an
    # arange-derived term summed with a data-dependent index (base_col + idx)
    # is dropped during lowering, so the column is selected by equality masks.
    base = out_ptr + row * stride_om + goff * 4 * stride_ok
    for k in tl.static_range(4):
        tl.store(base + k * stride_ok, v0, mask=gmask & (idx0 == k))
        tl.store(base + k * stride_ok, v1, mask=gmask & (idx1 == k))


def _decompress(compressed_A, M, K):
    """Recover the dense M x K matrix from a cuSPARSELt compressed blob."""
    dev = compressed_A.device
    dtype = compressed_A.dtype
    nnz = M * (K // 2)
    flat = compressed_A.reshape(-1)
    values = flat[:nnz].reshape(M, K // 2)
    meta = flat[nnz:].view(torch.uint16)

    n_kt = (K // 4 + 7) // 8
    dense = torch.zeros(M, K, dtype=dtype, device=dev)

    BLOCK_G = 64
    grid = (M, triton.cdiv(K // 4, BLOCK_G))
    _decompress_kernel[grid](
        values,
        meta,
        dense,
        M,
        K,
        n_kt,
        values.stride(0),
        values.stride(1),
        dense.stride(0),
        dense.stride(1),
        BLOCK_G=BLOCK_G,
    )
    return dense


@triton.jit
def _mm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    bias_ptr,
    alpha,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_rem = K - k * BLOCK_K
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_rem),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < k_rem) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    acc = acc * alpha
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        acc += bias[None, :]

    c = acc.to(c_ptr.dtype.element_ty)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def _cslt_sparse_mm(
    compressed_A,
    dense_B,
    bias=None,
    alpha=None,
    out_dtype=None,
    transpose_result=False,
    alg_id=0,
    split_k=1,
    split_k_mode=-1,
):
    """Sparse (2:4) x dense matmul matching ``aten._cslt_sparse_mm``.

    ``compressed_A`` is the cuSPARSELt-compressed representation of an ``M x K``
    matrix; ``dense_B`` is ``K x N``. Returns ``M x N`` (or ``N x M`` when
    ``transpose_result`` is set).

    Only correct on architectures whose cuSPARSELt metadata swizzle this
    decoder implements. See the module docstring.
    """
    K, N = dense_B.shape
    # ``torch._cslt_compress`` returns a 2-D blob whose leading dimension is the
    # original row count M; the trailing dimension packs the K/2 nonzero values
    # plus the (architecture-padded) metadata. Inferring M arithmetically from
    # the total element count is unreliable because cuSPARSELt pads the metadata
    # section, so read M directly from the compressed tensor's row dimension.
    if compressed_A.dim() != 2:
        raise RuntimeError(
            "expected a 2-D cuSPARSELt compressed input to infer the row count"
        )
    M = compressed_A.size(0)

    dense_A = _decompress(compressed_A, M, K)

    out_dt = out_dtype if out_dtype is not None else dense_B.dtype
    C = torch.empty(M, N, dtype=out_dt, device=dense_B.device)

    a_alpha = 1.0 if alpha is None else float(alpha)
    has_bias = bias is not None
    bias_arg = bias if has_bias else C  # placeholder ptr when unused

    grid = (triton.cdiv(M, 64) * triton.cdiv(N, 64),)
    _mm_kernel[grid](
        dense_A,
        dense_B,
        C,
        bias_arg,
        a_alpha,
        M,
        N,
        K,
        dense_A.stride(0),
        dense_A.stride(1),
        dense_B.stride(0),
        dense_B.stride(1),
        C.stride(0),
        C.stride(1),
        HAS_BIAS=has_bias,
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K=32,
        GROUP_M=8,
    )

    if transpose_result:
        return C.t().contiguous()
    return C
