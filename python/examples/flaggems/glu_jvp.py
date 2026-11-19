import torch
import triton
import triton.language as tl

# glu_jvp: forward-mode JVP of aten::glu. Upstream docstring contract:
# with d the (normalized) dim, N = glu.size(d), a = x[..., 0:N, ...] and
# b = x[..., N:2N, ...] (narrowed along d),
#     glu = a * sigmoid(b)
#     out = da * sigmoid(b) + glu * (1 - sigmoid(b)) * db
# where da/db are the matching halves of dx. The first half of x is never
# read; the output has N on d and broadcasts every other axis -- the four
# narrowed operands are broadcast to the common shape on the host.
#
# Non-fp64 inputs compute in fp32 (libdevice has no fp16/bf16 signatures).

_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


@triton.jit
def _glu_jvp_kernel(
    glu_ptr,
    b_ptr,
    da_ptr,
    db_ptr,
    out_ptr,
    n_elements,
    IS64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    glu = tl.load(glu_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    da = tl.load(da_ptr + offsets, mask=mask, other=0.0)
    db = tl.load(db_ptr + offsets, mask=mask, other=0.0)
    if IS64:
        bc = b.to(tl.float64)
    else:
        bc = b.to(tl.float32)
    sig_b = 1.0 / (1.0 + tl.exp(-bc))
    res = da * sig_b + glu * (1.0 - sig_b) * db
    tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


def glu_jvp(glu, x, dx, dim, out=None):
    for name, t in (("glu", glu), ("x", x), ("dx", dx)):
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"glu_jvp: {name} must be a tensor")
        if t.dtype not in _FLOAT_DTYPES:
            raise TypeError(
                "glu_jvp: only float16/float32/bfloat16/float64 tensors are supported"
            )
    nd = x.dim()
    d = dim + nd if dim < 0 else dim
    if d < 0 or d >= nd:
        raise IndexError(
            f"glu_jvp: dimension out of range (expected to be in range of "
            f"[{-nd}, {nd - 1}], but got {dim})"
        )
    for t in (glu, dx):
        if d >= t.dim():
            raise IndexError(
                f"glu_jvp: dimension out of range (expected to be in range of "
                f"[{-t.dim()}, {t.dim() - 1}], but got {dim})"
            )

    n = glu.size(d)
    if 2 * n > x.size(d) or 2 * n > dx.size(d):
        raise RuntimeError(
            "glu_jvp: the narrowed extent does not fit within the operand"
        )

    glu_l = glu.movedim(d, -1)
    x_l = x.movedim(d, -1)
    dx_l = dx.movedim(d, -1)
    b = x_l.narrow(-1, n, n)
    da = dx_l.narrow(-1, 0, n)
    db = dx_l.narrow(-1, n, n)

    shape = torch.broadcast_shapes(glu_l.shape, b.shape, da.shape, db.shape)
    glu_c = glu_l.expand(shape).contiguous()
    b_c = b.expand(shape).contiguous()
    da_c = da.expand(shape).contiguous()
    db_c = db.expand(shape).contiguous()

    out_moved = torch.empty(shape, dtype=glu.dtype, device=glu.device)
    total = out_moved.numel()
    if total:
        _glu_jvp_kernel[(triton.cdiv(total, 1024),)](
            glu_c,
            b_c,
            da_c,
            db_c,
            out_moved.reshape(-1),
            total,
            IS64=glu.dtype == torch.float64,
            BLOCK=1024,
        )
    res = out_moved.movedim(-1, d)
    if out is None:
        return res
    if tuple(out.shape) != tuple(res.shape):
        raise RuntimeError(
            "glu_jvp: the out buffer does not match the expected output shape"
        )
    if out.dtype != glu.dtype:
        raise RuntimeError("glu_jvp: the out buffer dtype must match glu")
    out.copy_(res)
    return out
