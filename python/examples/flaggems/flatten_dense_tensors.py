import torch
import triton
import triton.language as tl

# flatten_dense_tensors: aten::flatten_dense_tensors -- concatenate the
# flattened views of every tensor into one dense 1-D tensor. Upstream fanned
# the list out into 2/3/4/8-slot pointer-select kernels plus a
# binary-search metadata kernel, all built on selecting between raw
# data_ptr() integers cast to typed pointers; this backend cannot select
# between pointers, so this port launches one plain kernel per tensor
# (rule 2b) with identical results. Strided inputs go through a padded
# static_range decode kernel (shapes padded with 1, strides with 0).
#
# Deviations from upstream, each pinned to the aten oracle (probed on
# torch 2.15.0+cpu):
# - passing a bare Tensor / a list with a non-Tensor element now raises
#   aten's exact pybind schema errors; the empty list raises aten's
#   ValueError("torch.cat(): expected a non-empty list of Tensors")
#   (upstream raised RuntimeError).
# - upstream cast mixed-dtype groups to the promoted output dtype inside
#   the kernels at store time; this port casts on the host with .to()
#   before launching (value-identical for every dtype involved).
# - upstream rejected complex and fp8 dtypes; aten flattens them fine, so
#   they are copied through companion views (complex -> real component
#   with doubled count, bool/fp8 -> uint8), bit-for-bit.
# - tensors with more than 8 dims (upstream fed a dynamic MAX_DIM) are
#   materialized with .contiguous() and flat-copied.

_BLOCK = 1024

_REAL_OF_COMPLEX = {
    torch.complex64: torch.float32,
    torch.complex128: torch.float64,
}

# 1-byte dtypes Triton should not touch as scalars; copied as raw uint8.
_BYTEWISE_DTYPES = (
    torch.bool,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
)


@triton.jit(do_not_specialize=["out_offset", "numel"])
def _flatten_copy_kernel(out_ptr, src_ptr, out_offset, numel, BLOCK: tl.constexpr):
    """Copy ``numel`` contiguous elements of one tensor to ``out`` at ``out_offset``."""
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < numel
    val = tl.load(src_ptr + offs, mask=mask)
    tl.store(out_ptr + out_offset.to(tl.int64) + offs, val, mask=mask)


@triton.jit(
    do_not_specialize=[
        "out_offset",
        "numel",
        "d0",
        "d1",
        "d2",
        "d3",
        "d4",
        "d5",
        "d6",
        "d7",
        "s0",
        "s1",
        "s2",
        "s3",
        "s4",
        "s5",
        "s6",
        "s7",
    ]
)
def _flatten_strided_kernel(
    out_ptr,
    src_ptr,
    out_offset,
    numel,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    BLOCK: tl.constexpr,
):
    """Gather one strided tensor in logical row-major order (8 padded dims)."""
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < numel
    rem = offs
    phys = tl.zeros([BLOCK], dtype=tl.int64)

    c = rem % d7
    phys += c * s7
    rem = rem // d7
    c = rem % d6
    phys += c * s6
    rem = rem // d6
    c = rem % d5
    phys += c * s5
    rem = rem // d5
    c = rem % d4
    phys += c * s4
    rem = rem // d4
    c = rem % d3
    phys += c * s3
    rem = rem // d3
    c = rem % d2
    phys += c * s2
    rem = rem // d2
    c = rem % d1
    phys += c * s1
    rem = rem // d1
    phys += rem * s0

    val = tl.load(src_ptr + phys, mask=mask)
    tl.store(out_ptr + out_offset.to(tl.int64) + offs, val, mask=mask)


def _launch_flat(out, src, out_offset, numel):
    grid = (triton.cdiv(numel, _BLOCK),)
    _flatten_copy_kernel[grid](out, src, out_offset, numel, BLOCK=_BLOCK)


def _launch_strided(out, src, out_offset):
    dims = [int(d) for d in src.shape]
    strs = [int(s) for s in src.stride()]
    while len(dims) < 8:
        dims.append(1)
        strs.append(0)
    grid = (triton.cdiv(src.numel(), _BLOCK),)
    _flatten_strided_kernel[grid](
        out,
        src,
        out_offset,
        src.numel(),
        dims[0],
        dims[1],
        dims[2],
        dims[3],
        dims[4],
        dims[5],
        dims[6],
        dims[7],
        strs[0],
        strs[1],
        strs[2],
        strs[3],
        strs[4],
        strs[5],
        strs[6],
        strs[7],
        BLOCK=_BLOCK,
    )


def _flatten_one(out, src, out_offset):
    if src.dim() == 0:
        src = src.reshape(1)
    real = _REAL_OF_COMPLEX.get(src.dtype)
    if real is not None:
        if not src.is_contiguous():
            src = src.contiguous()
        _launch_flat(out.view(real), src.view(real), out_offset * 2, src.numel() * 2)
        return
    if src.dtype in _BYTEWISE_DTYPES:
        src_b = src.view(torch.uint8)
        out_b = out.view(torch.uint8)
        if src_b.is_contiguous():
            _launch_flat(out_b, src_b, out_offset, src_b.numel())
        else:
            _launch_strided(out_b, src_b, out_offset)
        return
    if src.is_contiguous() or src.dim() > 8:
        _launch_flat(out, src.contiguous(), out_offset, src.numel())
        return
    _launch_strided(out, src, out_offset)


def flatten_dense_tensors(tensors):
    """``aten::flatten_dense_tensors(Tensor[] tensors) -> Tensor``."""
    if isinstance(tensors, torch.Tensor):
        raise RuntimeError(
            "aten::flatten_dense_tensors() Expected a value of type "
            "'List[Tensor]' for argument 'tensors' but instead found type "
            "'Tensor'.\n"
            f"Position: 0\nValue: {tensors}\n"
            "Declaration: aten::flatten_dense_tensors(Tensor[] tensors) -> Tensor\n"
            "Cast error details: Unable to cast Python instance of type "
            "<class 'torch.Tensor'> to C++ type '?' (#define "
            "PYBIND11_DETAILED_ERROR_MESSAGES or compile in debug mode for details)"
        )
    for item in tensors:
        if not isinstance(item, torch.Tensor):
            raise RuntimeError(
                "aten::flatten_dense_tensors() Expected a value of type "
                "'List[Tensor]' for argument 'tensors' but instead found type "
                "'list'.\n"
                f"Position: 0\nValue: {list(tensors)}\n"
                "Declaration: aten::flatten_dense_tensors(Tensor[] tensors) -> Tensor\n"
                "Cast error details: Unable to cast Python instance of type "
                "<class 'list'> to C++ type '?' (#define "
                "PYBIND11_DETAILED_ERROR_MESSAGES or compile in debug mode for details)"
            )
    tensors = list(tensors)
    if tensors:
        device = tensors[0].device
        if any(t.device != device for t in tensors):
            raise RuntimeError("all tensors must be on the same device")
        if any(t.layout != torch.strided for t in tensors):
            raise RuntimeError("flatten_dense_tensors requires dense tensors")
        if any(t.is_conj() or t.is_neg() for t in tensors):
            raise NotImplementedError(
                "lazy conjugate/negative tensors are not supported"
            )
    if len(tensors) == 0:
        raise ValueError("torch.cat(): expected a non-empty list of Tensors")

    # Single contiguous tensor: aten itself returns the zero-copy flat view.
    if len(tensors) == 1:
        t0 = tensors[0]
        if t0.is_contiguous():
            return t0.view(-1)
        out = torch.empty(t0.numel(), dtype=t0.dtype, device=t0.device)
        if t0.numel() > 0:
            _flatten_one(out, t0, 0)
        return out

    out_dtype = tensors[0].dtype
    for t in tensors[1:]:
        out_dtype = torch.promote_types(out_dtype, t.dtype)

    out = torch.empty(
        sum(t.numel() for t in tensors), dtype=out_dtype, device=tensors[0].device
    )
    offset = 0
    for t in tensors:
        if t.numel() > 0:
            src = t if t.dtype == out_dtype else t.to(out_dtype)
            _flatten_one(out, src, offset)
        offset += t.numel()
    return out
