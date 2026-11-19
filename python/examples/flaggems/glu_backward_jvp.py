import torch
import triton
import triton.language as tl

# glu_backward_jvp: forward-mode JVP helper for glu backward. Upstream
# docstring contract, S = grad_glu.size(dim):
#     a = x.narrow(dim, 0, S);   b = x.narrow(dim, S, S)
#     da = dx.narrow(dim, 0, S); db = dx.narrow(dim, S, S)
#     gxa = grad_x.narrow(dim, 0, S)   # grad_x is a read-only operand;
#                                      # its second half is never read
#     sig_b = sigmoid(b); glu = a * sig_b; db_neg = db - db * sig_b
#     dgrad_x_a = dgrad_glu * sig_b + gxa * db_neg
#     dgrad_x_b = dgrad_x_a * (a - glu) + gxa * (da - da * sig_b - glu * db_neg)
#     out = cat([dgrad_x_a, dgrad_x_b], dim)
# grad_glu only supplies the split extent S: it is never loaded, so its
# values cannot change the result. grad_x is never written. The narrowed
# operands broadcast over every axis except dim (host-side, per the
# contract's broadcast rule).
#
# Non-fp64 inputs compute in fp32 (libdevice has no fp16/bf16 signatures).

_FLOAT8_DTYPES = (
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2fnuz,
)


@triton.jit
def _glu_bwd_jvp_kernel(
    a_ptr,
    b_ptr,
    da_ptr,
    db_ptr,
    gxa_ptr,
    dg_ptr,
    out_ptr,
    n_lanes,
    s,
    IS64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_lanes
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    da = tl.load(da_ptr + offsets, mask=mask, other=0.0)
    db = tl.load(db_ptr + offsets, mask=mask, other=0.0)
    gxa = tl.load(gxa_ptr + offsets, mask=mask, other=0.0)
    dg = tl.load(dg_ptr + offsets, mask=mask, other=0.0)
    if IS64:
        ac = a.to(tl.float64)
        bc = b.to(tl.float64)
        dac = da.to(tl.float64)
        dbc = db.to(tl.float64)
        gxac = gxa.to(tl.float64)
        dgc = dg.to(tl.float64)
    else:
        ac = a.to(tl.float32)
        bc = b.to(tl.float32)
        dac = da.to(tl.float32)
        dbc = db.to(tl.float32)
        gxac = gxa.to(tl.float32)
        dgc = dg.to(tl.float32)
    sig_b = 1.0 / (1.0 + tl.exp(-bc))
    glu = ac * sig_b
    db_neg = dbc - dbc * sig_b
    out_a = dgc * sig_b + gxac * db_neg
    out_b = out_a * (ac - glu) + gxac * (dac - dac * sig_b - glu * db_neg)
    row = offsets // s
    col = offsets - row * s
    base = row * (2 * s) + col
    tl.store(out_ptr + base, out_a.to(out_ptr.dtype.element_ty), mask=mask)
    tl.store(out_ptr + base + s, out_b.to(out_ptr.dtype.element_ty), mask=mask)


def glu_backward_jvp(grad_x, grad_glu, x, dgrad_glu, dx, dim, out=None):
    tensors = (grad_x, grad_glu, x, dgrad_glu, dx)
    for t in tensors:
        if t.dtype == torch.bool:
            raise RuntimeError("glu_backward_jvp: bool tensors are not supported")
        if t.dtype in _FLOAT8_DTYPES:
            raise RuntimeError("glu_backward_jvp: float8 tensors are not supported")
    nd = x.dim()
    d = dim + nd if dim < 0 else dim
    if d < 0 or d >= nd:
        raise IndexError(
            f"Dimension out of range (expected to be in range of "
            f"[-{nd}, {nd - 1}], but got {dim})"
        )
    s = grad_glu.size(d)
    if 2 * s > x.size(d) or 2 * s > dx.size(d):
        raise RuntimeError(
            f"glu_backward_jvp: invalid grad_glu extent {s} along dim {dim}"
        )
    if grad_x.size(d) < s:
        raise RuntimeError(
            f"glu_backward_jvp: grad_glu extent {s} exceeds the grad_x extent "
            f"{grad_x.size(d)} along dim {dim}"
        )
    if dgrad_glu.size(d) not in (1, s):
        raise RuntimeError(
            f"glu_backward_jvp: grad_glu extent {s} and dgrad_glu extent "
            f"{dgrad_glu.size(d)} are not broadcastable along dim {dim}"
        )

    x_l = x.movedim(d, -1)
    dx_l = dx.movedim(d, -1)
    gx_l = grad_x.movedim(d, -1)
    dg_l = dgrad_glu.movedim(d, -1)
    a = x_l.narrow(-1, 0, s)
    b = x_l.narrow(-1, s, s)
    da = dx_l.narrow(-1, 0, s)
    db = dx_l.narrow(-1, s, s)
    gxa = gx_l.narrow(-1, 0, s)
    dg = dg_l.narrow(-1, 0, s)

    shape = torch.broadcast_shapes(
        a.shape, b.shape, da.shape, db.shape, gxa.shape, dg.shape
    )
    a_c = a.expand(shape).contiguous()
    b_c = b.expand(shape).contiguous()
    da_c = da.expand(shape).contiguous()
    db_c = db.expand(shape).contiguous()
    gxa_c = gxa.expand(shape).contiguous()
    dg_c = dg.expand(shape).contiguous()

    out_moved = torch.empty(shape[:-1] + (2 * s,), dtype=x.dtype, device=x.device)
    lanes = gxa_c.numel()
    if lanes:
        _glu_bwd_jvp_kernel[(triton.cdiv(lanes, 1024),)](
            a_c.reshape(-1),
            b_c.reshape(-1),
            da_c.reshape(-1),
            db_c.reshape(-1),
            gxa_c.reshape(-1),
            dg_c.reshape(-1),
            out_moved.reshape(-1),
            lanes,
            s,
            IS64=x.dtype == torch.float64,
            BLOCK=1024,
        )
    res = out_moved.movedim(-1, d)
    if out is None:
        return res
    if tuple(out.shape) != tuple(res.shape):
        raise RuntimeError(
            "glu_backward_jvp: the out buffer does not match the expected output shape"
        )
    if out.dtype != res.dtype:
        raise RuntimeError(
            "glu_backward_jvp: the out buffer dtype must match the input"
        )
    out.copy_(res)
    return out
