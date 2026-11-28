import logging
import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)
_INT32_MAX = torch.iinfo(torch.int32).max

_BLOCK_SIZE = 512

_CPP_DTYPE_NAMES = {
    torch.float16: "c10::Half",
    torch.float32: "float",
    torch.float64: "double",
    torch.bfloat16: "c10::BFloat16",
    torch.bool: "bool",
    torch.uint8: "unsigned char",
    torch.int8: "signed char",
    torch.int16: "short int",
    torch.int32: "int",
    torch.int64: "long int",
    torch.complex32: "c10::complex<c10::Half>",
    torch.complex64: "c10::complex<float>",
    torch.complex128: "c10::complex<double>",
    torch.float8_e4m3fn: "c10::Float8_e4m3fn",
    torch.float8_e5m2: "c10::Float8_e5m2",
}

_BYTEWISE_DTYPES = (torch.bool, torch.float8_e4m3fn, torch.float8_e5m2)


@triton.jit
def slice_copy_kernel(
    inp_ptr,
    out_ptr,
    numel,
    inner,
    slice_len,
    dim_size,
    start,
    step,
    BLOCK_SIZE: tl.constexpr,
    INT64_INDEX: tl.constexpr,
):
    pid = tl.program_id(0)
    if INT64_INDEX:
        pid = pid.to(tl.int64)

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    outer_idx = offsets // (slice_len * inner)
    slice_idx = (offsets // inner) % slice_len
    inner_idx = offsets % inner

    dim_index = start + slice_idx * step
    input_offset = outer_idx * dim_size * inner + dim_index * inner + inner_idx

    val = tl.load(inp_ptr + input_offset, mask=mask)
    tl.store(out_ptr + offsets, val, mask=mask)


@triton.jit
def slice_copy_kernel_inner1(
    inp_ptr,
    out_ptr,
    numel,
    slice_len,
    dim_size,
    start,
    step,
    BLOCK_SIZE: tl.constexpr,
    INT64_INDEX: tl.constexpr,
):
    pid = tl.program_id(0)
    if INT64_INDEX:
        pid = pid.to(tl.int64)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    outer_idx = offsets // slice_len
    slice_idx = offsets - outer_idx * slice_len
    dim_index = start + slice_idx * step
    input_offset = outer_idx * dim_size + dim_index

    val = tl.load(inp_ptr + input_offset, mask=mask)
    tl.store(out_ptr + offsets, val, mask=mask)


@triton.jit
def slice_copy_kernel_step1(
    inp_ptr,
    out_ptr,
    numel,
    inner,
    slice_len,
    dim_size,
    start,
    BLOCK_SIZE: tl.constexpr,
    INT64_INDEX: tl.constexpr,
):
    pid = tl.program_id(0)
    if INT64_INDEX:
        pid = pid.to(tl.int64)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    outer_idx = offsets // (slice_len * inner)
    inner_idx = offsets % inner
    slice_idx = offsets // inner - outer_idx * slice_len

    dim_index = start + slice_idx
    input_offset = outer_idx * dim_size * inner + dim_index * inner + inner_idx

    val = tl.load(inp_ptr + input_offset, mask=mask)
    tl.store(out_ptr + offsets, val, mask=mask)


def _dtype_name(dtype: torch.dtype) -> str:
    return _CPP_DTYPE_NAMES.get(dtype, str(dtype).replace("torch.", ""))


def _resolve_slice_bounds(dim_size, start, end):
    if start is None:
        start = 0
    if end is None:
        end = dim_size
    if start < 0:
        start += dim_size
    if end < 0:
        end += dim_size
    start = max(0, min(start, dim_size))
    end = max(0, min(end, dim_size))
    return start, end


def _native_copy_(out: torch.Tensor, src: torch.Tensor):
    # Call native copy_ directly so the strided scatter does not re-enter
    # any overridden copy kernels.
    return torch.ops.aten.copy_.default.redispatch(_FALLBACK_KEYSET, out, src, False)


def _reinterpret_for_triton(inp, out, inner):
    # Complex tensors are copied through view_as_real (inner doubles);
    # bool/float8 through uint8 -- bit-for-bit reinterpretations.
    if inp.is_complex():
        return torch.view_as_real(inp), torch.view_as_real(out), inner * 2
    if inp.dtype in _BYTEWISE_DTYPES:
        return inp.view(torch.uint8), out.view(torch.uint8), inner
    return inp, out, inner


def _launch_slice_copy(inp, out, dim, start, slice_len, step):
    # Gather inp[..., start:start + slice_len * step:step, ...] into the
    # contiguous ``out`` with a Triton kernel. Unlike the upstream file,
    # which routes CPU inputs through a native aten copy, this port launches
    # the same Triton gather on every device (values identical).
    inp = inp.resolve_conj().resolve_neg()
    inp = inp.contiguous()

    inner = 1
    for d in range(dim + 1, inp.ndim):
        inner *= inp.size(d)
    dim_size = inp.size(dim)

    inp, out, inner = _reinterpret_for_triton(inp, out, inner)

    numel = out.numel()
    int64_index = max(numel, inp.numel()) > _INT32_MAX

    grid = (triton.cdiv(numel, _BLOCK_SIZE),)

    if inner == 1:
        slice_copy_kernel_inner1[grid](
            inp,
            out,
            numel,
            slice_len,
            dim_size,
            start,
            step,
            BLOCK_SIZE=_BLOCK_SIZE,
            INT64_INDEX=int64_index,
        )
    elif step == 1:
        slice_copy_kernel_step1[grid](
            inp,
            out,
            numel,
            inner,
            slice_len,
            dim_size,
            start,
            BLOCK_SIZE=_BLOCK_SIZE,
            INT64_INDEX=int64_index,
        )
    else:
        slice_copy_kernel[grid](
            inp,
            out,
            numel,
            inner,
            slice_len,
            dim_size,
            start,
            step,
            BLOCK_SIZE=_BLOCK_SIZE,
            INT64_INDEX=int64_index,
        )


def _slice_view(inp, dim, start, slice_len, step):
    sizes = list(inp.shape)
    sizes[dim] = slice_len
    strides = list(inp.stride())
    strides[dim] = strides[dim] * step
    offset = inp.storage_offset() + start * inp.stride(dim)
    return inp.as_strided(sizes, strides, offset)


def _slice_copy_impl(inp, dim, start, end, step, out):
    ndim = inp.ndim
    if ndim == 0:
        raise IndexError("slice() cannot be applied to a 0-dim tensor.")
    if dim < -ndim or dim >= ndim:
        raise IndexError(
            f"Dimension out of range (expected to be in range of "
            f"[{-ndim}, {ndim - 1}], but got {dim})"
        )
    if step <= 0:
        raise RuntimeError("slice step must be positive")

    dim = dim % ndim
    dim_size = inp.size(dim)
    start, end = _resolve_slice_bounds(dim_size, start, end)
    slice_len = max(0, (end - start + step - 1) // step)

    target_shape = list(inp.shape)
    target_shape[dim] = slice_len

    if out is None:
        out = torch.empty(target_shape, dtype=inp.dtype, device=inp.device)
    else:
        if list(out.shape) != target_shape:
            out.resize_(target_shape)
        if out.dtype != inp.dtype:
            raise RuntimeError(
                f"Expected out tensor to have dtype {_dtype_name(inp.dtype)}, "
                f"but got {_dtype_name(out.dtype)} instead"
            )
        if out.device != inp.device:
            raise RuntimeError(
                f"Expected out tensor to have device {inp.device}, "
                f"but got {out.device} instead"
            )

    if out.numel() == 0:
        return out

    # A strided / aliasing / internally overlapping ``out`` cannot take the
    # flat-offset store directly; those cases (non-contiguity covers the
    # overlapping strides) go through a contiguous staging buffer and the
    # native strided copy_, which raises ATen's overlap diagnostic.
    needs_staging = not out.is_contiguous() or torch._C._is_alias_of(inp, out)

    dst = (
        torch.empty(target_shape, dtype=out.dtype, device=out.device)
        if needs_staging
        else out
    )
    _launch_slice_copy(inp, dst, dim, start, slice_len, step)
    if needs_staging:
        _native_copy_(out, dst)
    return out


def slice_copy(inp, dim=0, start=None, end=None, step=1):
    """Copy of ``inp.slice(dim, start, end, step)`` via a Triton gather."""
    return _slice_copy_impl(inp, dim, start, end, step, out=None)


def slice_copy_out(inp, dim=0, start=None, end=None, step=1, *, out):
    """``slice_copy`` writing into ``out`` (resized/dtype-checked like aten)."""
    return _slice_copy_impl(inp, dim, start, end, step, out=out)
