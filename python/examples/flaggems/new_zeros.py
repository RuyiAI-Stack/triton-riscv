import torch
import triton
import triton.language as tl

# new_zeros: aten::new_zeros -- zeros tensor of ``size`` inheriting self's
# dtype/layout/device (kwargs override; pin state probed via is_pinned).
# Contiguous buffers zero byte-granularly through a uint8 view (complex and
# bool need no special casing that way); strided buffers zero through a
# constexpr shape/stride decode that writes only the view's elements.


@triton.jit
def _zero_bytes(p, N, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    tl.store(p + offs, tl.zeros([BLOCK], tl.uint8), mask=offs < N)


@triton.jit
def _zero_strided(
    p,
    N,
    NDIM: tl.constexpr,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    k = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = k < N
    off = tl.zeros([BLOCK], tl.int64)
    rem = k
    for i in tl.static_range(NDIM):
        s = SHAPE[i]
        off += (rem % s) * STRIDES[i]
        rem = rem // s
    tl.store(p + off, tl.zeros([BLOCK], p.dtype.element_ty), mask=mask)


def _as_size(size):
    # Nested sequences, torch.Size and bool extents are accepted (bool
    # extents coerce to 0/1 exactly as the native schema does).
    if isinstance(size, torch.Size):
        return tuple(int(e) for e in size)
    if isinstance(size, (list, tuple)):
        out = []
        for e in size:
            if isinstance(e, bool):
                out.append(int(e))
            else:
                out.append(e)
        return tuple(out)
    if isinstance(size, torch.Tensor):
        return tuple(int(e) for e in size.tolist())
    raise TypeError(
        "new_zeros(): argument 'size' must be a sequence of integers, "
        f"not {type(size).__name__}"
    )


def _zero_buffer(t):
    n = t.numel()
    if n == 0:
        return t
    if t.is_contiguous():
        nbytes = n * t.element_size()
        _zero_bytes[(triton.cdiv(nbytes, 1024),)](
            t.view(torch.uint8), nbytes, BLOCK=1024
        )
    else:
        _zero_strided[(triton.cdiv(n, 256),)](
            t,
            n,
            NDIM=t.dim(),
            SHAPE=tuple(t.shape),
            STRIDES=tuple(t.stride()),
            BLOCK=256,
        )
    return t


def new_zeros(
    self, size, *, dtype=None, layout=None, device=None, pin_memory=None, out=None
):
    """Zeros tensor of ``size`` defaulting to self's dtype/layout/device."""
    size_t = _as_size(size)
    if dtype is None:
        dtype = self.dtype
    if layout is None:
        layout = self.layout
    if device is None:
        device = self.device
    if pin_memory is None:
        try:
            pin_memory = bool(self.is_pinned())
        except Exception:  # sparse/meta sources have no pin state
            pin_memory = False
    if out is None:
        buf = torch.empty(
            size_t,
            dtype=dtype,
            layout=layout,
            device=device,
            pin_memory=pin_memory,
        )
    else:
        buf = out
        if tuple(buf.shape) != size_t:
            buf = buf.resize_(size_t)
    _zero_buffer(buf)
    return buf


def new_zeros_out(
    self, size, *, dtype=None, layout=None, device=None, pin_memory=None, out=None
):
    """``aten::new_zeros.out``: zero-fill the caller's buffer and return it."""
    return new_zeros(
        self,
        size,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
        out=out,
    )
