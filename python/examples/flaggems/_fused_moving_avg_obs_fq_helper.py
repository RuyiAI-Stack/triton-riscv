import torch
import triton
import triton.language as tl


@triton.jit
def _round_half_even(x):
    f = tl.floor(x)
    d = x - f
    odd = (f - 2.0 * tl.floor(f * 0.5)) > 0.5
    return tl.where(d > 0.5, f + 1.0, tl.where(d < 0.5, f, tl.where(odd, f + 1.0, f)))


@triton.jit
def _fused_kernel(
    x_ptr,
    out_ptr,
    mask_ptr,
    rmin_ptr,
    rmax_ptr,
    scale_ptr,
    zp_ptr,
    INNER,
    avg,
    qmin,
    qmax,
    dnegf,
    dposf,
    zsymf,
    OBS: tl.constexpr,
    FQ: tl.constexpr,
    SYM: tl.constexpr,
    FLOOR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    c = tl.program_id(0)
    base = c * INNER
    one = tl.arange(0, 1)
    zf = tl.zeros((1,), tl.float32)
    avgv = avg.to(tl.float32) + zf
    qmnf = qmin.to(tl.float32)
    qmxf = qmax.to(tl.float32)
    rmin = tl.load(rmin_ptr + c + one).to(tl.float32)
    rmax = tl.load(rmax_ptr + c + one).to(tl.float32)
    if OBS:
        acc_min = tl.full((BLOCK,), float("inf"), tl.float32)
        acc_max = tl.full((BLOCK,), float("-inf"), tl.float32)
        for i in range(0, INNER, BLOCK):
            idx = i + tl.arange(0, BLOCK)
            m = idx < INNER
            v = tl.load(x_ptr + base + idx, mask=m, other=0.0).to(tl.float32)
            acc_min = tl.minimum(acc_min, tl.where(m, v, float("inf")))
            acc_max = tl.maximum(acc_max, tl.where(m, v, float("-inf")))
        cmn = tl.min(acc_min, axis=0) + zf
        cmx = tl.max(acc_max, axis=0) + zf
        nmin = rmin + avgv * (cmn - rmin)
        nmax = rmax + avgv * (cmx - rmax)
        rmin = tl.where(tl.abs(rmin) == float("inf"), cmn, nmin)
        rmax = tl.where(tl.abs(rmax) == float("inf"), cmx, nmax)
        tl.store(rmin_ptr + c + one, rmin.to(rmin_ptr.dtype.element_ty))
        tl.store(rmax_ptr + c + one, rmax.to(rmax_ptr.dtype.element_ty))
    if FQ:
        qmnv = qmnf + zf
        qmxv = qmxf + zf
        mn = tl.minimum(rmin, zf)
        mx = tl.maximum(rmax, zf)
        sc_a = (mx - mn) / (qmxv - qmnv)
        deg = sc_a == 0.0
        sc_a = tl.where(deg, 0.1 + zf, sc_a)
        z_a = _round_half_even(qmnv - (mn / sc_a))
        z_a = tl.minimum(tl.maximum(z_a, qmnv), qmxv)
        z_a = tl.where(deg, tl.where(qmnv < 0.0, qmxv, qmnv), z_a)
        sc_v = sc_a
        z_v = z_a
        if SYM:
            dneg = dnegf + zf
            dpos = dposf + zf
            both = (mn < 0.0) & (mx > 0.0)
            sc_s = tl.maximum((-mn) / dneg, mx / dpos)
            sc_s = tl.where(sc_s == 0.0, 0.1 + zf, sc_s)
            z_s = zsymf + zf
            sc_v = tl.where(both, sc_s, sc_a)
            z_v = tl.where(both, z_s, z_a)
        z = tl.max(z_v, axis=0)
        sc_v = tl.maximum(sc_v, FLOOR + zf)
        sc = tl.max(sc_v, axis=0)
        tl.store(scale_ptr + c + one, sc_v.to(scale_ptr.dtype.element_ty))
        tl.store(zp_ptr + c + one, z_v.to(zp_ptr.dtype.element_ty))
        for i in range(0, INNER, BLOCK):
            idx = i + tl.arange(0, BLOCK)
            m = idx < INNER
            v = tl.load(x_ptr + base + idx, mask=m, other=0.0).to(tl.float32)
            q = _round_half_even(v / sc) + z
            valid = (q >= qmnf) & (q <= qmxf)
            qc = tl.minimum(tl.maximum(q, qmnf), qmxf)
            res = (qc - z) * sc
            tl.store(out_ptr + base + idx, res.to(out_ptr.dtype.element_ty), mask=m)
            tl.store(mask_ptr + base + idx, valid.to(tl.uint8), mask=m)
    else:
        for i in range(0, INNER, BLOCK):
            idx = i + tl.arange(0, BLOCK)
            m = idx < INNER
            v = tl.load(x_ptr + base + idx, mask=m, other=0.0).to(tl.float32)
            tl.store(out_ptr + base + idx, v.to(out_ptr.dtype.element_ty), mask=m)
            tl.store(mask_ptr + base + idx, tl.full((BLOCK,), 1, tl.uint8), mask=m)


@triton.jit
def _reduce_partials(x_ptr, pmin_ptr, pmax_ptr, N, CHUNK, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    start = pid * CHUNK
    lidx = tl.arange(0, BLOCK)
    acc_min = tl.full((BLOCK,), float("inf"), tl.float32)
    acc_max = tl.full((BLOCK,), float("-inf"), tl.float32)
    for i in range(start, start + CHUNK, BLOCK):
        idx = i + lidx
        m = idx < N
        v = tl.load(x_ptr + idx, mask=m, other=0.0).to(tl.float32)
        acc_min = tl.minimum(acc_min, tl.where(m, v, float("inf")))
        acc_max = tl.maximum(acc_max, tl.where(m, v, float("-inf")))
    tl.store(pmin_ptr + pid, tl.min(acc_min, axis=0))
    tl.store(pmax_ptr + pid, tl.max(acc_max, axis=0))


@triton.jit
def _finalize_pt(
    pmin_ptr,
    pmax_ptr,
    rmin_ptr,
    rmax_ptr,
    scale_ptr,
    zp_ptr,
    G,
    avg,
    qmin,
    qmax,
    dnegf,
    dposf,
    zsymf,
    OBS: tl.constexpr,
    FQ: tl.constexpr,
    SYM: tl.constexpr,
    FLOOR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    one = tl.arange(0, 1)
    zf = tl.zeros((1,), tl.float32)
    qmnf = qmin.to(tl.float32)
    qmxf = qmax.to(tl.float32)
    rmin = tl.load(rmin_ptr + one).to(tl.float32)
    rmax = tl.load(rmax_ptr + one).to(tl.float32)
    if OBS:
        lidx = tl.arange(0, BLOCK)
        acc_min = tl.full((BLOCK,), float("inf"), tl.float32)
        acc_max = tl.full((BLOCK,), float("-inf"), tl.float32)
        for i in range(0, G, BLOCK):
            idx = i + lidx
            m = idx < G
            pmn = tl.load(pmin_ptr + idx, mask=m, other=float("inf"))
            pmx = tl.load(pmax_ptr + idx, mask=m, other=float("-inf"))
            acc_min = tl.minimum(acc_min, pmn)
            acc_max = tl.maximum(acc_max, pmx)
        cmn = tl.min(acc_min, axis=0) + zf
        cmx = tl.max(acc_max, axis=0) + zf
        avgv = avg.to(tl.float32) + zf
        nmin = rmin + avgv * (cmn - rmin)
        nmax = rmax + avgv * (cmx - rmax)
        rmin = tl.where(tl.abs(rmin) == float("inf"), cmn, nmin)
        rmax = tl.where(tl.abs(rmax) == float("inf"), cmx, nmax)
        tl.store(rmin_ptr + one, rmin.to(rmin_ptr.dtype.element_ty))
        tl.store(rmax_ptr + one, rmax.to(rmax_ptr.dtype.element_ty))
    if FQ:
        qmnv = qmnf + zf
        qmxv = qmxf + zf
        mn = tl.minimum(rmin, zf)
        mx = tl.maximum(rmax, zf)
        sc_a = (mx - mn) / (qmxv - qmnv)
        deg = sc_a == 0.0
        sc_a = tl.where(deg, 0.1 + zf, sc_a)
        z_a = _round_half_even(qmnv - (mn / sc_a))
        z_a = tl.minimum(tl.maximum(z_a, qmnv), qmxv)
        z_a = tl.where(deg, tl.where(qmnv < 0.0, qmxv, qmnv), z_a)
        sc_v = sc_a
        z_v = z_a
        if SYM:
            dneg = dnegf + zf
            dpos = dposf + zf
            both = (mn < 0.0) & (mx > 0.0)
            sc_s = tl.maximum((-mn) / dneg, mx / dpos)
            sc_s = tl.where(sc_s == 0.0, 0.1 + zf, sc_s)
            z_s = zsymf + zf
            sc_v = tl.where(both, sc_s, sc_a)
            z_v = tl.where(both, z_s, z_a)
        sc_v = tl.maximum(sc_v, FLOOR + zf)
        tl.store(scale_ptr + one, sc_v.to(scale_ptr.dtype.element_ty))
        tl.store(zp_ptr + one, z_v.to(zp_ptr.dtype.element_ty))


@triton.jit
def _fq_elem(
    x_ptr,
    out_ptr,
    mask_ptr,
    scale_ptr,
    zp_ptr,
    N,
    CHUNK,
    qmin,
    qmax,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    start = pid * CHUNK
    lidx = tl.arange(0, BLOCK)
    sc = tl.load(scale_ptr).to(tl.float32)
    z = tl.load(zp_ptr).to(tl.float32)
    qmnf = qmin.to(tl.float32)
    qmxf = qmax.to(tl.float32)
    for i in range(start, start + CHUNK, BLOCK):
        off = i + lidx
        m = off < N
        v = tl.load(x_ptr + off, mask=m, other=0.0).to(tl.float32)
        q = _round_half_even(v / sc) + z
        valid = (q >= qmnf) & (q <= qmxf)
        qc = tl.minimum(tl.maximum(q, qmnf), qmxf)
        res = (qc - z) * sc
        tl.store(out_ptr + off, res.to(out_ptr.dtype.element_ty), mask=m)
        tl.store(mask_ptr + off, valid.to(tl.uint8), mask=m)


@triton.jit
def _copy_elem(x_ptr, out_ptr, mask_ptr, N, CHUNK, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    start = pid * CHUNK
    lidx = tl.arange(0, BLOCK)
    ones = tl.full((BLOCK,), 1, tl.uint8)
    for i in range(start, start + CHUNK, BLOCK):
        off = i + lidx
        m = off < N
        v = tl.load(x_ptr + off, mask=m, other=0.0)
        tl.store(out_ptr + off, v, mask=m)
        tl.store(mask_ptr + off, ones, mask=m)


_SCALAR_TYPE_NAMES = {
    torch.float16: "Half",
    torch.bfloat16: "BFloat16",
    torch.float64: "Double",
}
_GROUP_BLOCK = 4096
_RED_BLOCK = 4096
_LARGE_BLOCK = 8192
_SINGLE_MAX = 4096
_PARTIAL_G = 16
_ELEM_G = 16
_SCALE_FLOOR = 6.1e-5


def _fused_moving_avg_obs_fq_helper(
    self,
    observer_on,
    fake_quant_on,
    running_min,
    running_max,
    scale,
    zero_point,
    averaging_const,
    quant_min,
    quant_max,
    ch_axis,
    per_row_fake_quant=False,
    symmetric_quant=False,
):
    if self.dtype is not torch.float32:
        scalar_type = _SCALAR_TYPE_NAMES.get(self.dtype, str(self.dtype))
        raise RuntimeError(f"expected scalar type Float but found {scalar_type}")

    per_channel = bool(per_row_fake_quant)
    if per_channel and int(ch_axis) != 0:
        raise RuntimeError(
            "Per-channel FakeQuant in fused_moving_avg_obs_fake_quant is only "
            "supported on axis == 0"
        )

    obs = int(observer_on) != 0
    fq = int(fake_quant_on) != 0
    sym = bool(symmetric_quant)

    x = self
    shape = x.shape
    N = x.numel()
    dev = x.device
    if N == 0:
        return (
            torch.empty(shape, dtype=torch.float32, device=dev),
            torch.empty(shape, dtype=torch.bool, device=dev),
        )

    xf = x.view(-1) if x.is_contiguous() else x.contiguous().view(-1)
    if per_channel:
        C = shape[0] if x.dim() > 0 else 1
        if C == 0:
            return (
                torch.empty(shape, dtype=torch.float32, device=dev),
                torch.empty(shape, dtype=torch.bool, device=dev),
            )
        INNER = N // C
    else:
        C = 1
        INNER = N

    out = torch.empty(N, dtype=torch.float32, device=dev)
    mask = torch.empty(N, dtype=torch.bool, device=dev)
    mask_u8 = mask.view(torch.uint8)

    qmin = int(quant_min)
    qmax = int(quant_max)
    dneg_h = float((qmax - qmin + 1) // 2)
    dpos_h = float((qmax - qmin) // 2)
    zsym_h = float(round((qmin + qmax) * 0.5))

    if (not per_channel) and N > _SINGLE_MAX:
        G = _PARTIAL_G
        chunk = (N + G - 1) // G
        if obs:
            pmin = torch.empty(G, dtype=torch.float32, device=dev)
            pmax = torch.empty(G, dtype=torch.float32, device=dev)
            _reduce_partials[(G,)](
                xf, pmin, pmax, N, chunk, BLOCK=_RED_BLOCK, num_warps=1
            )
        else:
            G = 0
            pmin = pmax = scale
        _finalize_pt[(1,)](
            pmin,
            pmax,
            running_min,
            running_max,
            scale,
            zero_point,
            G,
            float(averaging_const),
            qmin,
            qmax,
            dneg_h,
            dpos_h,
            zsym_h,
            OBS=obs,
            FQ=fq,
            SYM=sym,
            FLOOR=_SCALE_FLOOR,
            BLOCK=_RED_BLOCK,
            num_warps=1,
        )
        n_prog = min(triton.cdiv(N, _LARGE_BLOCK), _ELEM_G)
        echunk = (N + n_prog - 1) // n_prog
        if fq:
            _fq_elem[(n_prog,)](
                xf,
                out,
                mask_u8,
                scale,
                zero_point,
                N,
                echunk,
                qmin,
                qmax,
                BLOCK=_LARGE_BLOCK,
                num_warps=1,
            )
        else:
            _copy_elem[(n_prog,)](
                xf,
                out,
                mask_u8,
                N,
                echunk,
                BLOCK=_LARGE_BLOCK,
                num_warps=1,
            )
        return out.view(shape), mask.view(shape)

    _fused_kernel[(C,)](
        xf,
        out,
        mask_u8,
        running_min,
        running_max,
        scale,
        zero_point,
        INNER,
        float(averaging_const),
        qmin,
        qmax,
        dneg_h,
        dpos_h,
        zsym_h,
        OBS=obs,
        FQ=fq,
        SYM=sym,
        FLOOR=_SCALE_FLOOR,
        BLOCK=_GROUP_BLOCK,
        num_warps=1,
    )
    return out.view(shape), mask.view(shape)
