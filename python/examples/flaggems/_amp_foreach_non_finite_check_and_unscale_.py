import torch
import triton
import triton.language as tl

# aten dtype names used in its dispatch error messages (probed on
# torch 2.15 CPU); only reached for dtypes outside the supported set.
_ATEN_DTYPE_NAMES = {
    torch.bool: "Bool",
    torch.uint8: "Byte",
    torch.int8: "Char",
    torch.int16: "Short",
    torch.int32: "Int",
    torch.int64: "Long",
    torch.float16: "Half",
    torch.bfloat16: "BFloat16",
    torch.float32: "Float",
    torch.float64: "Double",
    torch.complex64: "ComplexFloat",
    torch.complex128: "ComplexDouble",
}

_SUPPORTED_GRAD_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
)


@triton.jit
def _amp_foreach_non_finite_check_and_unscale_kernel(
    x_ptr,
    inv_scale_ptr,
    out_ptr,
    n_elements,
    IS_F64: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    scale = tl.load(inv_scale_ptr)

    if IS_F64:
        # fp64 grads keep double precision (aten computes them in
        # double; fp32 opmath would flush subnormals away).
        res = (x.to(tl.float64) * scale.to(tl.float64)).to(x.dtype)
    else:
        # fp16/bf16/fp32: aten widens to fp32 opmath, multiplies every
        # value (non-finite included: inf*s stays inf, nan*s stays nan),
        # and rounds back to the storage dtype.
        res = (x.to(tl.float32) * scale.to(tl.float32)).to(x.dtype)

    tl.store(out_ptr + offsets, res, mask=mask)


def _amp_foreach_non_finite_check_and_unscale_(tensors, found_inf, inv_scale):
    """Unscale gradients in place and flag non-finite values.

    Mirrors ``aten::_amp_foreach_non_finite_check_and_unscale_`` (probed
    on torch 2.15 CPU):

    - every grad tensor (fp16/bf16/fp32/fp64) is multiplied in place by
      the fp32 ``inv_scale`` scalar;
    - ``found_inf`` is filled with 1.0 when any tensor held a non-finite
      value *before* scaling -- it is never cleared, and an fp16
      overflow produced by the scaling itself does not set it;
    - all tensors are processed (no early return once found_inf is set).

    Upstream FlagGems diverges in three points pinned to aten here: it
    re-checks finiteness after scaling (so overflow would set
    found_inf), silently skips non-float tensors (aten raises
    NotImplementedError), and widens fp64 grads to fp32.
    """
    if not isinstance(tensors, (list, tuple)):
        raise TypeError(f"Expected list or tuple of tensors, got {type(tensors)}")

    # aten gate order (probed): inv_scale, then found_inf.
    if inv_scale.dtype != torch.float32:
        raise RuntimeError("inv_scale must be a float tensor.")
    if found_inf.dtype != torch.float32:
        raise RuntimeError("found_inf must be a float tensor.")

    for tensor in tensors:
        if tensor.dtype not in _SUPPORTED_GRAD_DTYPES:
            name = _ATEN_DTYPE_NAMES.get(tensor.dtype, str(tensor.dtype))
            raise NotImplementedError(
                "_amp_foreach_non_finite_check_and_unscale_cpu "
                f"not implemented for '{name}'"
            )

        # Pre-scale finiteness check, per tensor, in aten's loop order:
        # fill 1.0 (never reset, never added to).
        if not bool(torch.isfinite(tensor).all()):
            found_inf.fill_(1.0)

        num_elements = tensor.numel()
        if num_elements == 0:
            continue

        # The kernel walks flat memory, so run it on a contiguous copy
        # and write back through the view (aten's mul_ writes through
        # non-contiguous grads the same way).
        src = tensor.contiguous()
        out = torch.empty_like(src)

        BLOCK_SIZE = 1024
        grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
        _amp_foreach_non_finite_check_and_unscale_kernel[grid](
            src,
            inv_scale,
            out,
            num_elements,
            IS_F64=tensor.dtype == torch.float64,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        tensor.copy_(out)
