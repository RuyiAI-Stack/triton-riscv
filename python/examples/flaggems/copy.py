import torch
import triton
import triton.language as tl


@triton.jit
def copy_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x, mask=mask)


def _can_use_triton(dst, src):
    if dst.layout != torch.strided or src.layout != torch.strided:
        return False
    if dst.device != src.device:
        return False
    if dst.is_quantized or src.is_quantized:
        return False
    if src.is_complex() or dst.is_complex():
        return False
    return True


def _expand_like(src, target_shape):
    if src.shape == target_shape:
        return src
    return src.expand(target_shape)


def copy_(
    dst: torch.Tensor,
    src: torch.Tensor,
    non_blocking: bool = False,
):
    if isinstance(src, (int, float, bool)):
        src = torch.tensor(src, device=dst.device)
    elif not isinstance(src, torch.Tensor):
        raise TypeError("unsupport src type for copy_: ", type(src))

    if dst._is_zerotensor():
        raise RuntimeError(
            "ZeroTensors are immutable. Call clone() before copy_."
        )
    if src._is_zerotensor():
        return dst.zero_()

    if torch._C._is_alias_of(dst, src):
        # Align with PyTorch: if metadata fully matches, this is a no-op.
        if (
            dst.storage_offset() == src.storage_offset()
            and dst.stride() == src.stride()
            and dst.size() == src.size()
            and dst.dtype == src.dtype
            and dst.device == src.device
            and dst.is_conj() == src.is_conj()
            and dst.is_neg() == src.is_neg()
        ):
            return dst
        # Otherwise defer to PyTorch for well-defined semantics on overlapping writes.
        return torch.ops.aten.copy_.default.redispatch(
            _FALLBACK_KEYSET, dst, src, non_blocking
        )

    if src.numel() > 2**31 - 1 or dst.numel() > 2**31 - 1:
        return torch.ops.aten.copy_.default.redispatch(
            _FALLBACK_KEYSET, dst, src, non_blocking
        )

    if not _can_use_triton(dst, src):
        return torch.ops.aten.copy_.default.redispatch(
            _FALLBACK_KEYSET, dst, src, non_blocking
        )

    if dst.numel() == 0:
        # Respect PyTorch behaviour: empty tensors should still validate broadcast.
        return torch.ops.aten.copy_.default.redispatch(
            _FALLBACK_KEYSET, dst, src, non_blocking
        )

    try:
        broadcast_shape = torch.broadcast_shapes(dst.shape, src.shape)
    except RuntimeError as exc:
        raise RuntimeError(str(exc)) from exc

    if torch.Size(broadcast_shape) != dst.shape:
        raise RuntimeError(
            f"The broadcast shape {broadcast_shape} does not match "
            f"destination shape {tuple(dst.shape)}"
        )

    expanded_src = _expand_like(src, dst.shape)
    expanded_src = expanded_src.contiguous()
    dst_contig = dst.contiguous() if not dst.is_contiguous() else dst

    n_elements = dst_contig.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    copy_kernel[grid](
        expanded_src, dst_contig, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )

    if dst_contig.data_ptr() != dst.data_ptr():
        dst.copy_(dst_contig)
    return dst


def copy(
    template: torch.Tensor,
    src: torch.Tensor,
    *,
    non_blocking: bool | None = False,
):
    out = torch.empty_strided(
        template.size(),
        template.stride(),
        dtype=template.dtype,
        device=template.device,
    )
    copy_(out, src, non_blocking=bool(non_blocking))
    return out
