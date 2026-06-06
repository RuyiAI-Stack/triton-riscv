import torch
import triton
import triton.language as tl


_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)


@triton.jit
def _to_copy_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    inp = tl.load(in_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, inp, mask=mask)


def _resolve_dtype(x: torch.Tensor, dtype: torch.dtype | None) -> torch.dtype:
    if dtype is None:
        return x.dtype
    if isinstance(dtype, torch.dtype):
        return dtype
    raise TypeError(f"Unsupported dtype argument type: {type(dtype)!r}")


def _resolve_device(x: torch.Tensor, device: torch.device | None) -> torch.device:
    if device is None:
        return x.device
    return torch.device(device)


def _normalize_memory_format(
    memory_format: torch.memory_format | None,
) -> torch.memory_format:
    if memory_format is None:
        return torch.preserve_format
    return memory_format


def _allocate_preserve_format(x: torch.Tensor, empty_kwargs: dict) -> torch.Tensor:
    if torch.ops.aten.is_non_overlapping_and_dense(x):
        return torch.empty_strided(x.size(), x.stride(), **empty_kwargs)
    return torch.empty_like(x, memory_format=torch.preserve_format, **empty_kwargs)


def to_copy(
    x,
    *,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
    non_blocking=False,
    memory_format=None,
):
    if (layout is not None and layout != torch.strided) or x.layout != torch.strided:
        raise NotImplementedError("to_copy currently supports strided tensors only.")
    if pin_memory is not None:
        raise NotImplementedError("to_copy does not yet support pin_memory=True.")
    if x.is_quantized:
        raise NotImplementedError("Quantized tensors are not supported in to_copy yet.")

    target_dtype = _resolve_dtype(x, dtype)
    target_device = _resolve_device(x, device)
    target_memory_format = _normalize_memory_format(memory_format)

    if x.dtype.is_complex or target_dtype.is_complex:
        out = torch.ops.aten._to_copy.default.redispatch(
            _FALLBACK_KEYSET,
            x,
            dtype=target_dtype,
            layout=layout,
            device=target_device,
            pin_memory=pin_memory,
            non_blocking=non_blocking,
            memory_format=target_memory_format,
        )
        return out

    if target_device != x.device:
        out = torch.ops.aten._to_copy.default.redispatch(
            _FALLBACK_KEYSET,
            x,
            dtype=target_dtype,
            layout=layout,
            device=target_device,
            pin_memory=pin_memory,
            non_blocking=non_blocking,
            memory_format=target_memory_format,
        )
        return out

    empty_kwargs = {"dtype": target_dtype, "device": target_device}

    if target_memory_format is torch.preserve_format:
        out = _allocate_preserve_format(x, empty_kwargs)
    else:
        out = torch.empty_like(x, memory_format=target_memory_format, **empty_kwargs)

    n_elements = x.numel()
    if n_elements == 0:
        return out

    x_contig = x.contiguous()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _to_copy_kernel[grid](x_contig, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out
