import torch
import triton
import triton.language as tl


@triton.jit
def replicationpad3d_kernel(
    x_ptr,
    out_ptr,
    D_in,
    H_in,
    W_in,
    D_out,
    H_out,
    W_out,
    pad_l,
    pad_t,
    pad_f,
    stride_xn,
    stride_xc,
    stride_xd,
    stride_xh,
    stride_xw,
    stride_on,
    stride_oc,
    stride_od,
    stride_oh,
    stride_ow,
    C,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_w = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_ncd = tl.program_id(2)

    d_idx = pid_ncd % D_out
    nc_idx = pid_ncd // D_out
    c_idx = nc_idx % C
    n_idx = nc_idx // C

    iz = d_idx - pad_f
    iz = tl.where(iz < 0, 0, iz)
    iz = tl.where(iz > D_in - 1, D_in - 1, iz)

    x_base_ptr = x_ptr + n_idx * stride_xn + c_idx * stride_xc + iz * stride_xd
    out_base_ptr = out_ptr + n_idx * stride_on + c_idx * stride_oc + d_idx * stride_od

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)

    iy = offs_h - pad_t
    iy = tl.where(iy < 0, 0, iy)
    iy = tl.where(iy > H_in - 1, H_in - 1, iy)

    ix = offs_w - pad_l
    ix = tl.where(ix < 0, 0, ix)
    ix = tl.where(ix > W_in - 1, W_in - 1, ix)

    x_offset = iy[:, None] * stride_xh + ix[None, :] * stride_xw
    out_offset = offs_h[:, None] * stride_oh + offs_w[None, :] * stride_ow

    mask = (offs_h[:, None] < H_out) & (offs_w[None, :] < W_out)

    vals = tl.load(x_base_ptr + x_offset, mask=mask)
    tl.store(out_base_ptr + out_offset, vals, mask=mask)


def replication_pad3d(x, padding):
    if isinstance(padding, int):
        pad_l = pad_r = pad_t = pad_b = pad_f = pad_ba = padding
    else:
        pad_l, pad_r, pad_t, pad_b, pad_f, pad_ba = padding

    is_4d = x.ndim == 4
    if is_4d:
        x = x.unsqueeze(0)

    N, C, D_in, H_in, W_in = x.shape
    D_out = D_in + pad_f + pad_ba
    H_out = H_in + pad_t + pad_b
    W_out = W_in + pad_l + pad_r

    out = torch.empty((N, C, D_out, H_out, W_out), device=x.device, dtype=x.dtype)

    BLOCK_H = 8
    BLOCK_W = 32

    def grid(meta):
        return (
            triton.cdiv(W_out, BLOCK_W),
            triton.cdiv(H_out, BLOCK_H),
            N * C * D_out,
        )

    replicationpad3d_kernel[grid](
        x,
        out,
        D_in,
        H_in,
        W_in,
        D_out,
        H_out,
        W_out,
        pad_l,
        pad_t,
        pad_f,
        *x.stride(),
        *out.stride(),
        C,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
    )

    return out.squeeze(0) if is_4d else out


def replication_pad3d_out(x, padding, out):
    if isinstance(padding, int):
        pad_l = pad_r = pad_t = pad_b = pad_f = pad_ba = padding
    else:
        pad_l, pad_r, pad_t, pad_b, pad_f, pad_ba = padding

    is_4d = x.ndim == 4
    if is_4d:
        x = x.unsqueeze(0)

    N, C, D_in, H_in, W_in = x.shape
    D_out = D_in + pad_f + pad_ba
    H_out = H_in + pad_t + pad_b
    W_out = W_in + pad_l + pad_r

    assert out.shape == (N, C, D_out, H_out, W_out) or (
        is_4d and out.shape == (C, D_out, H_out, W_out)
    ), "out shape mismatch"

    BLOCK_H = 8
    BLOCK_W = 32

    def grid(meta):
        return (
            triton.cdiv(W_out, BLOCK_W),
            triton.cdiv(H_out, BLOCK_H),
            N * C * D_out,
        )

    replicationpad3d_kernel[grid](
        x,
        out,
        D_in,
        H_in,
        W_in,
        D_out,
        H_out,
        W_out,
        pad_l,
        pad_t,
        pad_f,
        *x.stride(),
        *out.stride(),
        C,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
    )

    return out
