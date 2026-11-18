"""``aten::_foreach_*`` binary family: table-generated wrappers over a shared
pair/ternary executor.

The overload set mirrors the ATen schemas audited upstream:

* ``List``      -- two tensor lists paired position by position
* ``Scalar``    -- one scalar broadcast across every entry
* ``ScalarList``-- one scalar per entry
* ``Tensor``    -- one tensor broadcast across the whole list (``add``/``mul``/
  ``div`` only)

``_WITH_ALPHA`` records the overloads whose schema carries ``alpha`` (``add``
and ``sub``); the generated wrappers all keep ``alpha`` as a default keyword
for signature uniformity, and no dispatcher ever passes it elsewhere.  ``lerp`` and
``addcmul``/``addcdiv`` are ternary in every overload and get their own
executor entry point, and ``pow.ScalarAndTensor`` reverses the base/exponent
roles, so it also stands alone.
"""

import triton
import triton.language as tl
import triton.language.extra.libdevice as _ld

import torch

_BLOCK = 1024

# pair op ids (list & scalar kernels)
_ADD = 0
_SUB = 1
_MUL = 2
_DIV = 3
_CLAMP_MAX = 4  # clamp_max(x, m) == min(x, m)
_CLAMP_MIN = 5  # clamp_min(x, m) == max(x, m)
_MAXIMUM = 6
_MINIMUM = 7
_POW = 8
_COPY = 9
_POW_ST = 10  # pow with a scalar base: base ** x

# ternary op ids
_LERP = 0
_ADDCMUL = 1
_ADDCDIV = 2


@triton.jit
def _pair_kernel(x_ptr, y_ptr, o_ptr, alpha, n, OP: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    m = idx < n
    x = tl.load(x_ptr + idx, mask=m, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + idx, mask=m, other=0.0).to(tl.float32)
    if OP == 0:
        r = x + alpha * y
    elif OP == 1:
        r = x - alpha * y
    elif OP == 2:
        r = x * y
    elif OP == 3:
        r = x / y
    elif OP == 4:
        r = tl.minimum(x, y)
    elif OP == 5:
        r = tl.maximum(x, y)
    elif OP == 6:
        r = tl.maximum(x, y)
    elif OP == 7:
        r = tl.minimum(x, y)
    elif OP == 8:
        r = _ld.pow(x, y)
    else:  # COPY
        r = y
    tl.store(o_ptr + idx, r.to(o_ptr.dtype.element_ty), mask=m)


@triton.jit
def _scalar_kernel(x_ptr, o_ptr, y, alpha, n, OP: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    m = idx < n
    x = tl.load(x_ptr + idx, mask=m, other=0.0).to(tl.float32)
    if OP == 0:
        r = x + alpha * y
    elif OP == 1:
        r = x - alpha * y
    elif OP == 2:
        r = x * y
    elif OP == 3:
        r = x / y
    elif OP == 4:
        r = tl.minimum(x, y)
    elif OP == 5:
        r = tl.maximum(x, y)
    elif OP == 6:
        r = tl.maximum(x, y)
    elif OP == 7:
        r = tl.minimum(x, y)
    elif OP == 8:
        r = _ld.pow(x, y)
    else:  # POW_ST: scalar base ** elementwise exponent
        r = _ld.pow(tl.where(m, y, y), x)
    tl.store(o_ptr + idx, r.to(o_ptr.dtype.element_ty), mask=m)


@triton.jit
def _ternary_kernel(
    a_ptr, b_ptr, c_ptr, w_ptr, o_ptr, val, n,
    OP: tl.constexpr, HAS_W: tl.constexpr, BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    m = idx < n
    a = tl.load(a_ptr + idx, mask=m, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + idx, mask=m, other=0.0).to(tl.float32)
    if OP == 0:  # lerp: a + w * (b - a); weight from w_ptr or val
        if HAS_W:
            w = tl.load(w_ptr + idx, mask=m, other=0.0).to(tl.float32)
        else:
            w = val
        r = a + w * (b - a)
    else:
        c = tl.load(c_ptr + idx, mask=m, other=1.0).to(tl.float32)
        if HAS_W:
            w = tl.load(w_ptr + idx, mask=m, other=0.0).to(tl.float32)
        else:
            w = val
        if OP == 1:  # addcmul: a + value * b * c
            r = a + w * (b * c)
        else:  # addcdiv: a + value * (b / c)
            r = a + w * (b / c)
    tl.store(o_ptr + idx, r.to(o_ptr.dtype.element_ty), mask=m)


def _pair_op(name):
    return globals()["_" + name.upper()]


def _contig(t):
    return t if t.is_contiguous() else t.contiguous()


def _run_pair(x, y, op, alpha, inplace):
    """One (self, other) pair; ``y`` is a tensor here (List/Tensor overload)."""
    if y.shape != x.shape:
        y = y.expand(x.shape)
    yc = _contig(y)
    if inplace and not x.is_contiguous():
        staging = _run_pair(x.contiguous(), y, op, alpha, inplace=False)
        x.copy_(staging)
        return x
    xc = _contig(x)
    dst = x if inplace else torch.empty_like(x)
    dstc = dst if dst.is_contiguous() else torch.empty(x.shape, dtype=x.dtype)
    n = xc.numel()
    if n:
        _pair_kernel[(triton.cdiv(n, _BLOCK),)](
            xc.view(-1), yc.view(-1), dstc.view(-1), float(alpha), n, OP=op, BLOCK=_BLOCK
        )
    if dstc is not dst:
        dst.copy_(dstc)
    return dst


def _run_scalar(x, s, op, alpha, inplace):
    if inplace and not x.is_contiguous():
        staging = _run_scalar(x.contiguous(), s, op, alpha, inplace=False)
        x.copy_(staging)
        return x
    xc = _contig(x)
    dst = x if inplace else torch.empty_like(x)
    dstc = dst if dst.is_contiguous() else torch.empty(x.shape, dtype=x.dtype)
    n = xc.numel()
    if n:
        _scalar_kernel[(triton.cdiv(n, _BLOCK),)](
            xc.view(-1), dstc.view(-1), float(s), float(alpha), n, OP=op, BLOCK=_BLOCK
        )
    if dstc is not dst:
        dst.copy_(dstc)
    return dst


def _apply_list(xs, ys, op, alpha, inplace):
    outs = [_run_pair(x, y, op, alpha, inplace) for x, y in zip(xs, ys)]
    return None if inplace else outs


def _apply_scalar(xs, s, op, alpha, inplace):
    if isinstance(s, (list, tuple)):
        return _apply_scalarlist(xs, s, op, alpha, inplace)
    outs = [_run_scalar(x, s, op, alpha, inplace) for x in xs]
    return None if inplace else outs


def _apply_scalarlist(xs, ss, op, alpha, inplace):
    outs = [_run_scalar(x, s, op, alpha, inplace) for x, s in zip(xs, ss)]
    return None if inplace else outs


def _apply_tensor(xs, t, op, alpha, inplace):
    outs = [_run_pair(x, t, op, alpha, inplace) for x in xs]
    return None if inplace else outs


def _apply_ternary(self, tensors1, tensors2, weights, op, inplace):
    """Ternary executor.

    ``lerp`` passes ``weights`` (tensor list / scalar / scalar list) and
    ``tensors2=None``; ``addcmul``/``addcdiv`` pass ``tensor1``/``tensor2``
    lists and a ``value`` that is a scalar, a scalar list, or a broadcast
    tensor.
    """
    n_list = len(self)
    outs = []
    for i in range(n_list):
        a = self[i]
        b = tensors1[i]
        if inplace and not a.is_contiguous():
            sub = [a.contiguous()]
            sub_b = [b.contiguous() if isinstance(b, torch.Tensor) else b]
            sub_w = [weights[i] if isinstance(weights, list) else weights] if weights is not None else None
            sub_c = [tensors2[i].contiguous()] if tensors2 is not None else None
            res = _apply_ternary(sub, sub_b, sub_c, sub_w, op, inplace=False)
            a.copy_(res[0])
            outs.append(a)
            continue
        ac = _contig(a)
        bc = _contig(b) if isinstance(b, torch.Tensor) else None
        dst = a if inplace else torch.empty_like(a)
        dstc = dst if dst.is_contiguous() else torch.empty(a.shape, dtype=a.dtype)
        w = weights[i] if isinstance(weights, list) else weights
        if tensors2 is not None:
            cc = _contig(tensors2[i])
            if isinstance(w, torch.Tensor):
                wc = _contig(w.expand(a.shape)) if w.shape != a.shape else _contig(w)
                _ternary_kernel[(triton.cdiv(a.numel(), _BLOCK),)](
                    ac.view(-1), bc.view(-1), cc.view(-1), wc.view(-1), dstc.view(-1),
                    1.0, a.numel(), OP=op, HAS_W=True, BLOCK=_BLOCK,
                )
            else:
                _ternary_kernel[(triton.cdiv(a.numel(), _BLOCK),)](
                    ac.view(-1), bc.view(-1), cc.view(-1), ac.view(-1), dstc.view(-1),
                    float(w), a.numel(), OP=op, HAS_W=False, BLOCK=_BLOCK,
                )
        else:
            if isinstance(w, torch.Tensor):
                wc = _contig(w.expand(a.shape)) if w.shape != a.shape else _contig(w)
                _ternary_kernel[(triton.cdiv(a.numel(), _BLOCK),)](
                    ac.view(-1), bc.view(-1), ac.view(-1), wc.view(-1), dstc.view(-1),
                    1.0, a.numel(), OP=_LERP, HAS_W=True, BLOCK=_BLOCK,
                )
            else:
                _ternary_kernel[(triton.cdiv(a.numel(), _BLOCK),)](
                    ac.view(-1), bc.view(-1), ac.view(-1), ac.view(-1), dstc.view(-1),
                    float(w), a.numel(), OP=_LERP, HAS_W=False, BLOCK=_BLOCK,
                )
        if dstc is not dst:
            dst.copy_(dstc)
        outs.append(dst)
    return None if inplace else outs


# ---------------------------------------------------------------------------
# generated wrappers (mirror of the upstream overload table)
# ---------------------------------------------------------------------------

_WITH_ALPHA = {("add", "List"), ("add", "Tensor"), ("sub", "List")}

_OVERLOADS = {
    "add": ("List", "Scalar", "ScalarList", "Tensor"),
    "sub": ("List", "Scalar", "ScalarList"),
    "mul": ("List", "Scalar", "ScalarList", "Tensor"),
    "div": ("List", "Scalar", "ScalarList", "Tensor"),
    "clamp_max": ("List", "Scalar", "ScalarList"),
    "clamp_min": ("List", "Scalar", "ScalarList"),
    "maximum": ("List", "Scalar", "ScalarList"),
    "minimum": ("List", "Scalar", "ScalarList"),
    "pow": ("List", "Scalar", "ScalarList"),
}

_LERP_OVERLOADS = ("List", "Scalar", "ScalarList")
_TERNARY_OVERLOADS = ("Scalar", "ScalarList", "Tensor")


def _make_list(name, inplace, with_alpha):
    del with_alpha
    op = _pair_op(name)

    def fn(self, other, alpha=1.0):
        return _apply_list(self, other, op, alpha, inplace)

    return fn


def _make_scalar(name, inplace, with_alpha):
    op = _pair_op(name)

    def fn(self, scalar, alpha=1.0):
        return _apply_scalar(self, scalar, op, alpha, inplace)

    return fn


def _make_tensor(name, inplace, with_alpha):
    op = _pair_op(name)

    def fn(self, tensor, alpha=1.0):
        return _apply_tensor(self, tensor, op, alpha, inplace)

    return fn


def _make_lerp(ov, inplace):
    if ov == "List":
        def fn(self, tensors1, weight):
            return _apply_ternary(self, tensors1, None, weight, _LERP, inplace)
    elif ov == "ScalarList":
        def fn(self, tensors1, scalars):
            return _apply_ternary(self, tensors1, None, list(scalars), _LERP, inplace)
    else:
        def fn(self, tensors1, weight):
            return _apply_ternary(self, tensors1, None, weight, _LERP, inplace)
    return fn


def _make_ternary(name, ov, inplace):
    op = _ADDCMUL if name == "addcmul" else _ADDCDIV

    def fn(self, tensor1, tensor2, value=1.0):
        if isinstance(value, (list, tuple)):
            weights = list(value)
        else:
            weights = value
        return _apply_ternary(self, tensor1, tensor2, weights, op, inplace)

    return fn


_globals = globals()
_WRAPPERS = {}

for _name, _ovs in _OVERLOADS.items():
    for _inplace in (False, True):
        _suffix = "_" if _inplace else ""
        for _ov in _ovs:
            _alpha = (_name, _ov) in _WITH_ALPHA
            if _ov == "List":
                _fn = _make_list(_name, _inplace, _alpha)
            elif _ov == "Tensor":
                _fn = _make_tensor(_name, _inplace, _alpha)
            else:
                _fn = _make_scalar(_name, _inplace, _alpha)
            _attr = f"_foreach_{_name}{_suffix}_{_ov}"
            _fn.__name__ = _attr
            _fn.__qualname__ = _attr
            _globals[_attr] = _fn
            _WRAPPERS[f"_foreach_{_name}{_suffix}.{_ov}"] = _fn

for _ov in _LERP_OVERLOADS:
    for _inplace in (False, True):
        _suffix = "_" if _inplace else ""
        _fn = _make_lerp(_ov, _inplace)
        _attr = f"_foreach_lerp{_suffix}_{_ov}"
        _fn.__name__ = _attr
        _fn.__qualname__ = _attr
        _globals[_attr] = _fn
        _WRAPPERS[f"_foreach_lerp{_suffix}.{_ov}"] = _fn

for _name in ("addcmul", "addcdiv"):
    for _inplace in (False, True):
        _suffix = "_" if _inplace else ""
        for _ov in _TERNARY_OVERLOADS:
            _fn = _make_ternary(_name, _ov, _inplace)
            _attr = f"_foreach_{_name}{_suffix}_{_ov}"
            _fn.__name__ = _attr
            _fn.__qualname__ = _attr
            _globals[_attr] = _fn
            _WRAPPERS[f"_foreach_{_name}{_suffix}.{_ov}"] = _fn


def registered_wrappers():
    """ATen overload key -> wrapper, covering the whole binary family."""
    return dict(_WRAPPERS)


# ---------------------------------------------------------------------------
# standalone entries
# ---------------------------------------------------------------------------


def _foreach_copy(self, src, non_blocking=False):
    del non_blocking
    return _apply_list(self, src, _COPY, 1.0, inplace=False)


def _foreach_copy_(self, src, non_blocking=False):
    del non_blocking
    _apply_list(self, src, _COPY, 1.0, inplace=True)
    return None


def _foreach_pow_ScalarAndTensor(self, exponent):
    """``pow.ScalarAndTensor``: scalar base, tensor-list exponent."""
    return [_run_scalar(t, float(self), _POW_ST, 1.0, inplace=False) for t in exponent]


_WRAPPERS["_foreach_copy.List"] = _foreach_copy
_WRAPPERS["_foreach_copy_.List"] = _foreach_copy_
_WRAPPERS["_foreach_pow.ScalarAndTensor"] = _foreach_pow_ScalarAndTensor
