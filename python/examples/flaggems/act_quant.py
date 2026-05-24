import torch
import triton
import triton.language as tl


@triton.jit
def fast_log2_ceil(x):
    # bits_x = T.reinterpret("uint32", x)
    bits_x = x.cast(tl.uint32, bitcast=True)
    exp_x = (bits_x >> 23) & 0xFF
    man_bits = bits_x & ((1 << 23) - 1)
    # return T.Cast("int32", exp_x - 127 + T.if_then_else(man_bits != 0, 1, 0))
    return (exp_x - 127 + tl.where(man_bits != 0, 1, 0)).cast(tl.int32)


@triton.jit
def fast_pow2(x):
    bits_x = (x + 127) << 23
    # return T.reinterpret("float32", bits_x)
    return bits_x.cast(tl.float32, bitcast=True)


@triton.jit
def fast_round_scale(amax, fp8_max_inv):
    return fast_pow2(fast_log2_ceil(amax * fp8_max_inv))


@triton.jit(do_not_specialize=["M"])
def act_quant_triton_kernel(
    X_ptr,
    Y_ptr,
    S_ptr,
    M,
    N,
    stride_xm,
    stride_ym,
    stride_sm,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ROUND_SCALE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    row_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_row = row_offset < M
    mask_col = col_offsets < N
    mask = mask_row[:, None] & mask_col[None, :]

    x = tl.load(
        X_ptr + row_offset[:, None] * stride_xm + col_offsets[None, :],
        mask=mask,
        other=0.0,
    )

    amax = tl.max(tl.abs(x), axis=1)
    amax = tl.maximum(amax, 1e-4)

    FP8_MAX: tl.constexpr = 448.0
    FP8_MAX_INV: tl.constexpr = 1.0 / 448.0

    if ROUND_SCALE:
        # Round scale to power of 2: scale = 2^ceil(log2(amax / 448))
        # scale_raw = amax * FP8_MAX_INV
        # log2_scale = tl.math.log2(scale_raw)
        # log2_ceil = tl.math.ceil(log2_scale)
        # scale = tl.math.exp2(log2_ceil)
        scale = fast_round_scale(amax, FP8_MAX_INV)
    else:
        scale = amax * FP8_MAX_INV

    y = x / scale[:, None]
    y = tl.clamp(y, -FP8_MAX, FP8_MAX)

    y_offset = row_offset
    tl.store(
        Y_ptr + y_offset[:, None] * stride_ym + col_offsets[None, :],
        y.to(tl.float8e4nv),
        mask=mask,
    )

    s_offset = row_offset
    tl.store(S_ptr + s_offset * stride_sm + pid_n, scale, mask=mask_row)


def act_quant_triton(
    x: torch.Tensor, block_size: int = 128, scale_fmt: str | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantizes the input tensor `x` using block-wise quantization.

    Args:
        x (torch.Tensor): The input tensor to be quantized. Must be contiguous and
                          its last dimension size must be divisible by `block_size`.
        block_size (int, optional): The size of the blocks for quantization. Default is 128.
        scale_fmt (Optional[str], optional): If not None, rounds scale to power of 2.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The quantized tensor with dtype `torch.float8_e4m3fn`.
            - A tensor of scaling factors with dtype `torch.float32`.
    """
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert x.size(-1) % block_size == 0, (
        f"Last dimension size must be divisible by block_size (block_size={block_size})"
    )

    N = x.size(-1)
    # original_shape = x.shape
    x_2d = x.view(-1, N)
    M = x_2d.size(0)

    BLOCK_M = 32
    # if M <= 32:
    #     BLOCK_M = M
    # elif M <= 512:
    #     BLOCK_M = 16
    # else:
    #     BLOCK_M = 32

    BLOCK_N = block_size
    m_blocks = triton.cdiv(M, BLOCK_M)
    n_blocks = N // BLOCK_N

    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s = x.new_empty(*x.size()[:-1], n_blocks, dtype=torch.float32)
    y_view = y.view(-1, N)
    s_view = s.view(-1, n_blocks)

    grid = (m_blocks, n_blocks)
    act_quant_triton_kernel[grid](
        x_2d,
        y_view,
        s_view,
        M,
        N,
        x_2d.stride(0),
        y_view.stride(0),
        s_view.stride(0),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        ROUND_SCALE=(scale_fmt is not None),
    )

    return y, s
