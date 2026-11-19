"""choose_qparams_optimized - host-driven search with Triton error kernels.

Reference semantics (aten / flag_gems; callable on this build as
torch.choose_qparams_optimized, the underscored spelling is not compiled
into this CPU-only torch):
  x   = first `numel` contiguous float32 elements
  lo  = min(x), hi = max(x)
  step = (hi - lo) / n_bins,  qmax = 2**bit_width - 1
  err(lo_c, hi_c):  scale = half((hi_c - half(lo_c)) / qmax),  inv = 1/scale
                    q = clamp(rint((x - half(lo_c)) * inv), 0, qmax)
                    err = sum((x - (q*scale + half(lo_c)))**2)
  The range shrinks one step at a time from the low or the high side for as
  long as an improvement exists; the final state is returned as (hi, lo)
  1-element float32 tensors (the aten schema's scale/zero_point slots carry
  the adjusted max/min).

Port notes (triton-riscv, vs the upstream flag_gems file):
- Only upstream's fallback plan is ported, used for every size: a
  host-driven shrink loop whose candidate losses are evaluated by Triton
  kernels. The upstream fused single-launch kernel relies on a
  data-dependent device-side while loop and libdevice nearbyint, neither
  of which this backend guarantees.
- The half() rounding (float32 -> float16 -> float32, as at::Half does)
  stays on the host via struct packing, exactly like upstream's fallback.
  An in-kernel float32 -> float16 conversion crashes this toolchain
  natively (probed: results correct, heap corrupted on exit), so no
  float16 value ever enters a kernel.
- The rounding inside the error kernel is the floor-based round-half-to-
  even reconstruction (probed equal to torch.round), not libdevice.rint.
- An empty request (numel <= 0) raises ValueError mirroring the aten
  reference ("input tensor is empty and has no data"); the upstream module
  would return search garbage here.

Parity envelope (same caveat the upstream test carries): the shrink
decisions compare float32 sums of squared errors, so aten's sequential
accumulation and any blocked/tree reduction order diverge as the element
count grows, which can flip a near-tie decision into a different but
equally valid range. The port matches the aten reference wherever the
decision margin exceeds that reduction-order noise (probed: the upstream
shape matrix at numel <= 1024, and selected larger configurations).
"""

import logging
import struct

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_ERROR_BLOCK = 1024


@triton.jit
def _rint(x):
    """Round half to even (std::nearbyint) built from floor only."""
    f = tl.floor(x)
    d = x - f
    is_odd = 2.0 * tl.floor(f * 0.5) != f
    up = (d > 0.5) | ((d == 0.5) & is_odd)
    return f + tl.where(up, 1.0, 0.0)


@triton.jit
def _minmax_kernel(x_ptr, out_ptr, numel, BLOCK: tl.constexpr):
    vmin = tl.full((BLOCK,), float("inf"), tl.float32)
    vmax = tl.full((BLOCK,), float("-inf"), tl.float32)
    for off in range(0, numel, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        m = idx < numel
        v = tl.load(x_ptr + idx, mask=m, other=0.0)
        vmin = tl.minimum(vmin, tl.where(m, v, float("inf")))
        vmax = tl.maximum(vmax, tl.where(m, v, float("-inf")))
    tl.store(out_ptr + 0, tl.min(vmin))
    tl.store(out_ptr + 1, tl.max(vmax))


@triton.jit
def _qerror_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    xmin1,
    scale1,
    inv1,
    xmin2,
    scale2,
    inv2,
    qmax,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)

    q1 = _rint((x - xmin1) * inv1)
    q1 = tl.minimum(tl.maximum(q1, 0.0), qmax)
    d1 = x - (q1 * scale1 + xmin1)
    err1 = tl.sum(tl.where(mask, d1 * d1, 0.0))

    q2 = _rint((x - xmin2) * inv2)
    q2 = tl.minimum(tl.maximum(q2, 0.0), qmax)
    d2 = x - (q2 * scale2 + xmin2)
    err2 = tl.sum(tl.where(mask, d2 * d2, 0.0))

    tl.store(out_ptr + 2 * pid + 0, err1)
    tl.store(out_ptr + 2 * pid + 1, err2)


@triton.jit
def _reduce2_kernel(p_ptr, out_ptr, n_rows, BLOCK: tl.constexpr):
    r = tl.arange(0, BLOCK)
    m = r < n_rows
    a = tl.load(p_ptr + 2 * r, mask=m, other=0.0)
    b = tl.load(p_ptr + 2 * r + 1, mask=m, other=0.0)
    tl.store(out_ptr + 0, tl.sum(a))
    tl.store(out_ptr + 1, tl.sum(b))


@triton.jit
def _store2_kernel(a_ptr, b_ptr, va, vb):
    tl.store(a_ptr, va)
    tl.store(b_ptr, vb)


def _half_round(value):
    """float32 -> float16 -> float32, as at::Half does on the reference side."""
    return struct.unpack("<e", struct.pack("<e", value))[0]


def _make_params(xmin, xmax, qmax):
    xmin_h = _half_round(xmin)
    data_range = xmax - xmin_h
    scale = 1.0 if data_range == 0.0 else _half_round(data_range / qmax)
    inv = 0.0 if scale == 0.0 else 1.0 / scale
    return xmin_h, scale, inv


def run(input, numel, n_bins, ratio, bit_width):
    numel = int(numel)
    n_bins = int(n_bins)
    bit_width = int(bit_width)
    ratio = float(ratio)

    if numel <= 0:
        raise ValueError(
            "choose_qparams_optimized: input tensor is empty and has no data"
        )

    if input.dim() == 1 and input.is_contiguous():
        x = input
    else:
        x = input.contiguous().reshape(-1)

    device = x.device
    mm = torch.empty(2, dtype=torch.float32, device=device)
    _minmax_kernel[(1,)](x, mm, numel, BLOCK=_ERROR_BLOCK)
    xmin, xmax = (float(v) for v in mm.tolist())

    stepsize = (xmax - xmin) / n_bins
    min_bins = int(n_bins * (1.0 - ratio))
    qmax = float((1 << bit_width) - 1)

    nblk = max(1, (numel + _ERROR_BLOCK - 1) // _ERROR_BLOCK)
    partial = torch.empty((nblk, 2), dtype=torch.float32, device=device)
    buf = torch.empty(2, dtype=torch.float32, device=device)
    rblock = max(2, triton.next_power_of_2(nblk))

    def errors(p1, p2):
        _qerror_kernel[(nblk,)](
            x,
            partial,
            numel,
            p1[0],
            p1[1],
            p1[2],
            p2[0],
            p2[1],
            p2[2],
            qmax,
            BLOCK=_ERROR_BLOCK,
        )
        _reduce2_kernel[(1,)](partial, buf, nblk, BLOCK=rblock)
        vals = buf.tolist()
        return float(vals[0]), float(vals[1])

    cur_min, cur_max = xmin, xmax
    params0 = _make_params(cur_min, cur_max, qmax)
    cur_loss, _ = errors(params0, params0)

    threshold = min_bins * stepsize
    while cur_min + threshold < cur_max:
        params1 = _make_params(cur_min + stepsize, cur_max, qmax)
        params2 = _make_params(cur_min, cur_max - stepsize, qmax)
        loss1, loss2 = errors(params1, params2)

        if cur_loss < loss1 and cur_loss < loss2:
            break

        if loss1 < loss2:
            cur_min = cur_min + stepsize
            cur_loss = loss1
        else:
            cur_max = cur_max - stepsize
            cur_loss = loss2

    max_out = torch.empty(1, dtype=torch.float32, device=device)
    min_out = torch.empty(1, dtype=torch.float32, device=device)
    _store2_kernel[(1,)](max_out, min_out, float(cur_max), float(cur_min))
    return max_out, min_out


def choose_qparams_optimized(input, numel, n_bins, ratio, bit_width):
    logger.debug("GEMS CHOOSE_QPARAMS_OPTIMIZED")

    # The aten reference is float32-only; reject other dtypes explicitly.
    assert input.dtype == torch.float32, (
        f"choose_qparams_optimized only supports float32 input, got {input.dtype}"
    )
    return run(input, numel, n_bins, ratio, bit_width)


__all__ = ["choose_qparams_optimized", "run"]
