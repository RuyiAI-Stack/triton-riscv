import torch
import triton
import triton.language as tl

# _reshape_from_tensor: aten::_reshape_from_tensor -- reshape by an int64
# shape *tensor*. When the target is viewable the result is a plain view
# (upstream behavior, checked there via the ATen computeStride helper);
# otherwise a Triton gather materializes the copy. Complex inputs gather
# through their real-pair views; lazy conj/neg bits are resolved first.
# (Upstream dispatches nine dtype-specialized kernels for the copy; this
# port uses the family-standard strided gather, same values.)


@triton.jit
def _rft_gather_kernel(
    src,
    dst,
    numel,
    SHAPE: tl.constexpr,
    INPUT_STRIDES: tl.constexpr,
    NDIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Row-major decode over the source's logical shape; the destination is
    # contiguous.
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    rem = offs.to(tl.int64)
    src_off = tl.zeros([BLOCK], dtype=tl.int64)
    for d in tl.static_range(NDIM - 1, -1, -1):
        idx = rem % SHAPE[d]
        rem = rem // SHAPE[d]
        src_off += idx * INPUT_STRIDES[d]
    vals = tl.load(src + src_off, mask=mask)
    tl.store(dst + offs, vals, mask=mask)


def _numel_of(shape):
    n = 1
    for d in shape:
        n *= d
    return n


def _make_target(shape, ref_numel):
    """Validate ``shape`` (an int64 1-D tensor) and return the target extents."""
    if not isinstance(shape, torch.Tensor):
        raise RuntimeError(
            "aten::_reshape_from_tensor() Expected a value of type 'Tensor' for "
            "argument 'shape' but instead found type '%s'." % type(shape).__name__
        )
    if shape.dtype != torch.int64:
        raise RuntimeError("expected scalar type Long but found %s" % str(shape.dtype))
    if shape.dim() != 1:
        raise RuntimeError(
            "Expected shape_tensor.dim() == 1 to be true, but got false."
        )
    dims = shape.tolist()
    inferred = -1
    known = 1
    for i, d in enumerate(dims):
        if d == -1:
            if inferred >= 0:
                raise RuntimeError("only one dimension can be inferred")
            inferred = i
        elif d < 0:
            raise RuntimeError(
                "invalid shape dimension %d at index %d of shape %s" % (d, i, dims)
            )
        else:
            known *= d
    if inferred >= 0:
        if known == 0 or ref_numel % known != 0:
            raise RuntimeError(
                "shape '%s' is invalid for input of size %d" % (dims, ref_numel)
            )
        dims[inferred] = ref_numel // known
    if _numel_of(dims) != ref_numel:
        raise RuntimeError(
            "shape '%s' is invalid for input of size %d" % (dims, ref_numel)
        )
    return dims


def _reshape_from_tensor(self, shape):
    """``self`` reshaped by the int64 tensor ``shape``; a view when viewable, else a copy."""
    dims = _make_target(shape, self.numel())
    try:
        return self.view(dims)
    except RuntimeError:
        pass
    src = self
    if src.is_conj():
        src = src.resolve_conj()
    if getattr(src, "is_neg", lambda: False)():
        src = src.resolve_neg()
    out = torch.empty(dims, dtype=self.dtype, device=self.device)
    numel = out.numel()
    if numel == 0:
        return out
    if src.is_complex():
        gather_src = torch.view_as_real(src)
        gather_dst = torch.view_as_real(out)
        numel = gather_dst.numel()
    else:
        gather_src = src
        gather_dst = out
    grid = (triton.cdiv(numel, 1024),)
    _rft_gather_kernel[grid](
        gather_src,
        gather_dst,
        numel,
        SHAPE=tuple(gather_src.shape),
        INPUT_STRIDES=tuple(gather_src.stride()),
        NDIM=gather_src.dim(),
        BLOCK=1024,
    )
    return out
