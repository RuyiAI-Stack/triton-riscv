import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def reflection_pad3d_kernel(
    in_ptr,
    out_ptr,
    B,
    D_in,
    H_in,
    W_in,
    pad_l,
    pad_t,
    pad_f,
    pD,
    pH,
    pW,
    D_out,
    H_out,
    W_out,
    BLOCK_DHW: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Flatten 3D index to 1D for block processing
    offs_n = pid_n * BLOCK_DHW + tl.arange(0, BLOCK_DHW)
    # Decode to (d, h, w) coordinates
    d_idx = offs_n // (H_out * W_out)
    h_idx = (offs_n % (H_out * W_out)) // W_out
    w_idx = offs_n % W_out

    mask = (offs_n < D_out * H_out * W_out) & (pid_b < B)

    base_in = pid_b * (D_in * H_in * W_in)
    base_out = pid_b * (D_out * H_out * W_out)

    # Compute reflected indices for depth. The reflection period
    # (2 * (n - 1), floored to 1 for size-1 dims with zero padding -- aten
    # allows those) is computed on the host because a size-1 dim would make
    # the in-kernel 2 * (D_in - 1) a division by zero.
    z = d_idx.to(tl.int32) - pad_f
    t_d = tl.abs(z)
    m_d = t_d % pD
    id_ = tl.where(m_d < D_in, m_d, pD - m_d)

    # Compute reflected indices for height
    y = h_idx.to(tl.int32) - pad_t
    t_h = tl.abs(y)
    m_h = t_h % pH
    ih = tl.where(m_h < H_in, m_h, pH - m_h)

    # Compute reflected indices for width
    x = w_idx.to(tl.int32) - pad_l
    t_w = tl.abs(x)
    m_w = t_w % pW
    iw = tl.where(m_w < W_in, m_w, pW - m_w)

    # Load from input and store to output
    in_offs = id_ * (H_in * W_in) + ih * W_in + iw
    vals = tl.load(in_ptr + base_in + in_offs, mask=mask, other=0)
    tl.store(out_ptr + base_out + offs_n, vals, mask=mask)


@triton.jit
def copy_tensor_kernel(in_ptr, out_ptr, B, D, H, W, BLOCK_DHW: tl.constexpr):
    pid_b = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_n = pid_n * BLOCK_DHW + tl.arange(0, BLOCK_DHW)
    mask = (offs_n < D * H * W) & (pid_b < B)

    base = pid_b * (D * H * W)
    vals = tl.load(in_ptr + base + offs_n, mask=mask, other=0)
    tl.store(out_ptr + base + offs_n, vals, mask=mask)


def launch_reflection_pad3d(input: torch.Tensor, padding, out: torch.Tensor = None):
    """reflection_pad3d core; ``padding`` is aten-ordered
    ``(pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)``
    (last dim first), which is exactly what upstream unpacks here too.

    Deviations from upstream, both to match the aten oracle: a size-1 spatial
    dim with zero padding on it is accepted (aten only rejects
    ``pad >= dim``), and the per-dim reflection periods are computed on the
    host so those size-1 dims are safe in the kernel. Upstream's blanket
    ``D_in < 2`` rejection is dropped. Upstream's flag_gems device checks are
    dropped with the flag_gems machinery.
    """
    # Validate padding format
    if not isinstance(padding, (list, tuple)):
        raise ValueError("padding must be a sequence")
    if len(padding) != 6:
        raise ValueError(
            "padding must be a sequence of length 6: (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)"
        )
    pad_l, pad_r, pad_t, pad_b, pad_f, pad_ba = [int(p) for p in padding]

    # Validate padding values
    if pad_l < 0 or pad_r < 0 or pad_t < 0 or pad_b < 0 or pad_f < 0 or pad_ba < 0:
        raise ValueError("padding values must be >= 0")

    # Handle 4D input (C, D, H, W) -> (1, C, D, H, W) by unsqueezing
    is_4d = input.dim() == 4
    if is_4d:
        input = input.unsqueeze(0)

    # Validate input
    if input.dim() < 5:
        raise ValueError("input must have at least 5 dimensions")

    x = input.contiguous()
    N, C, D_in, H_in, W_in = x.shape

    # Validate reflection padding constraints (aten's rule: every pad < dim)
    if (
        pad_l >= W_in
        or pad_r >= W_in
        or pad_t >= H_in
        or pad_b >= H_in
        or pad_f >= D_in
        or pad_ba >= D_in
    ):
        raise ValueError(
            "padding values must be less than the input spatial dimensions for reflection padding"
        )

    D_out = D_in + pad_f + pad_ba
    H_out = H_in + pad_t + pad_b
    W_out = W_in + pad_l + pad_r

    # Handle output tensor
    if out is None:
        out = torch.empty((N, C, D_out, H_out, W_out), device=x.device, dtype=x.dtype)
    else:
        expected_shape = (N, C, D_out, H_out, W_out)
        if tuple(out.shape) != expected_shape:
            raise ValueError(
                f"out tensor has shape {tuple(out.shape)}, expected {expected_shape}"
            )
        if out.dtype != x.dtype:
            raise ValueError(
                f"out dtype {out.dtype} does not match input dtype {x.dtype}"
            )
        if out.device != x.device:
            raise ValueError("out must be on the same device as input")
        out = out.contiguous()

    # No padding: just copy
    if (
        pad_l == 0
        and pad_r == 0
        and pad_t == 0
        and pad_b == 0
        and pad_f == 0
        and pad_ba == 0
    ):
        BLOCK_DHW = 256
        grid = (N * C, triton.cdiv(D_in * H_in * W_in, BLOCK_DHW))
        copy_tensor_kernel[grid](x, out, N * C, D_in, H_in, W_in, BLOCK_DHW=BLOCK_DHW)
        return out.squeeze(0) if is_4d else out

    # Reflection periods; a size-1 dim (only possible with zero pad on it)
    # gets period 1 so the modulo never divides by zero.
    pD = max(2 * D_in - 2, 1)
    pH = max(2 * H_in - 2, 1)
    pW = max(2 * W_in - 2, 1)

    BLOCK_DHW = 256
    grid = (N * C, triton.cdiv(D_out * H_out * W_out, BLOCK_DHW))
    reflection_pad3d_kernel[grid](
        x,
        out,
        N * C,
        D_in,
        H_in,
        W_in,
        pad_l,
        pad_t,
        pad_f,
        pD,
        pH,
        pW,
        D_out,
        H_out,
        W_out,
        BLOCK_DHW=BLOCK_DHW,
    )
    return out.squeeze(0) if is_4d else out


def reflection_pad3d(input: torch.Tensor, padding, *, out: torch.Tensor = None):
    """aten::reflection_pad3d (reflect, period 2*(n-1)); padding is aten-ordered
    (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)."""
    logger.debug("GEMS REFLECTION_PAD3D")
    return launch_reflection_pad3d(input, padding, out=out)


def reflection_pad3d_out(input: torch.Tensor, padding, out: torch.Tensor):
    """``reflection_pad3d`` writing into ``out`` (shape/dtype/device checked)."""
    logger.debug("GEMS REFLECTION_PAD3D_OUT")
    return reflection_pad3d(input, padding, out=out)
