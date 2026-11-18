import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as _ld

_BLOCK = 1024


@triton.jit
def _max_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    acc = tl.full((BLOCK,), float("-inf"), tl.float32)
    for i in range(0, n, BLOCK):
        idx = i + tl.arange(0, BLOCK)
        m = idx < n
        v = tl.load(x_ptr + idx, mask=m, other=float("-inf")).to(tl.float32)
        acc = tl.maximum(acc, tl.where(m, v, float("-inf")))
    tl.store(out_ptr, tl.max(acc, axis=0))


@triton.jit
def _norm_kernel(x_ptr, out_ptr, n, ORD: tl.constexpr, KIND: tl.constexpr, BLOCK: tl.constexpr):
    acc_s = tl.zeros((1,), tl.float32)
    acc_max = tl.full((BLOCK,), float("-inf"), tl.float32)
    acc_min = tl.full((BLOCK,), float("inf"), tl.float32)
    for i in range(0, n, BLOCK):
        idx = i + tl.arange(0, BLOCK)
        m = idx < n
        v = tl.load(x_ptr + idx, mask=m, other=0.0).to(tl.float32)
        a = tl.abs(v)
        if KIND == 0:
            acc_s = acc_s + tl.sum(tl.where(m & (v != 0.0), 1.0, 0.0) + tl.zeros((BLOCK,), tl.float32))
        elif KIND == 1:
            acc_max = tl.maximum(acc_max, tl.where(m, a, float("-inf")))
        elif KIND == 2:
            acc_min = tl.minimum(acc_min, tl.where(m, a, float("inf")))
        else:
            acc_s = acc_s + tl.sum(tl.where(m, _ld.pow(a, ORD), 0.0) + tl.zeros((BLOCK,), tl.float32))
    if KIND == 1:
        tl.store(out_ptr, tl.max(acc_max, axis=0))
    elif KIND == 2:
        tl.store(out_ptr, tl.min(acc_min, axis=0))
    elif KIND == 0:
        tl.store(out_ptr, tl.sum(acc_s, axis=0))
    else:
        tl.store(out_ptr, _ld.pow(tl.sum(acc_s, axis=0), 1.0 / ORD))


@triton.jit
def _zero_kernel(x_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    m = idx < n
    tl.store(x_ptr + idx, tl.zeros((BLOCK,), tl.float32) * 0.0, mask=m)


def _flat(tensors):
    out = []
    for t in tensors:
        if isinstance(t, (list, tuple)):
            out.extend(t)
        else:
            out.append(t)
    return out


def _run_list(tensors, kern, out_dtype, **kw):
    results = []
    for t in tensors:
        contig = t if t.is_contiguous() else t.contiguous()
        out = torch.empty((), dtype=out_dtype, device=t.device)
        n = contig.numel()
        kern[(1,)](contig, out, n, BLOCK=_BLOCK, **kw)
        results.append(out)
    return results


def _foreach_max(self):
    tensors = _flat(self)
    if any(t.numel() == 0 for t in tensors):
        raise RuntimeError(
            "max(): Expected reduction dim to be specified for input.numel() == 0."
        )
    out_dtype = torch.float32 if tensors and tensors[0].dtype.is_floating_point else torch.int64
    return _run_list(tensors, _max_kernel, out_dtype)


def _foreach_norm(self, ord=2, dtype=None):
    tensors = _flat(self)
    if not tensors:
        return []
    out_dtype = dtype if dtype is not None else tensors[0].dtype
    if ord == 0:
        kind, o = 0, 1.0
    elif ord == float("inf"):
        kind, o = 1, 1.0
    elif ord == float("-inf"):
        kind, o = 2, 1.0
    else:
        kind, o = 3, float(ord)
    return _run_list(tensors, _norm_kernel, out_dtype, ORD=o, KIND=kind)


def _zero_list(tensors):
    for t in tensors:
        if t.is_contiguous():
            n = t.numel()
            if n:
                _zero_kernel[(triton.cdiv(n, _BLOCK),)](t.view(-1), n, BLOCK=_BLOCK)
        else:
            t.copy_(torch.zeros_like(t))


def _foreach_zero_(self):
    _zero_list(_flat(self))
    return None


def _foreach_zero(self):
    tensors = _flat(self)
    _zero_list(tensors)
    return tensors
