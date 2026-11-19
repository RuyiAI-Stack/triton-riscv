import torch
import triton
import triton.language as tl


@triton.jit
def trace_backward_kernel(
    grad_ptr,
    out_ptr,
    num_diag,
    diag_stride,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_diag

    grad = tl.load(grad_ptr).to(out_ptr.type.element_ty)

    # out has been pre-filled with zeros; only the diagonal entries are
    # written: flat offset d * (M + 1) is element (d, d) of a contiguous
    # (N, M) tensor.
    diag_offsets = offsets * diag_stride
    tl.store(out_ptr + diag_offsets, grad, mask=mask)


def trace_backward(grad, sizes):
    """Backward of ``torch.trace`` for a 2D input of shape ``sizes``.

    Mirrors ``aten::trace_backward`` (probed on torch 2.15 CPU): a zero
    (N, M) tensor of grad's dtype with the scalar ``grad`` on the main
    diagonal -- min(N, M) entries, i.e. ``grad * eye(N, M)`` including
    non-square inputs.  aten's error contracts are mirrored in order:
    non-matrix sizes, then the zeros() size check, then the 0-dim grad
    check.

    Upstream FlagGems writes ceil(N * M / (M + 1)) diagonal slots,
    which overshoots the true diagonal for skinny matrices ((5, 1)
    would fill three entries where aten fills one) -- clamped to
    min(N, M) here.
    """
    if len(sizes) != 2:
        # aten's message for sizes that do not describe a matrix.
        raise RuntimeError("expected matrix input")

    N, M = int(sizes[0]), int(sizes[1])

    # torch.zeros raises aten's own negative-size error here, matching
    # aten's check order (sizes -> zeros -> grad dimensionality).
    out = torch.zeros((N, M), dtype=grad.dtype, device=grad.device)

    numel = N * M
    if numel == 0:
        return out

    if grad.dim() != 0:
        raise RuntimeError(
            "index_fill_ only supports a 0-dimensional value tensor, "
            f"but got tensor with {grad.dim()} dimension(s)."
        )

    num_diag = min(N, M)
    diag_stride = M + 1

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(num_diag, BLOCK_SIZE),)

    trace_backward_kernel[grid](
        grad,
        out,
        num_diag,
        diag_stride,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out
