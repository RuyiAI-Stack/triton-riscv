import torch
import triton
import triton.language as tl

# block_diag: torch.block_diag as a Triton scatter. Upstream had three
# kernels (regular-stride read, uniform pointer-array, varlen pointer-array);
# the pointer-array variants loaded raw addresses from an int64 tensor and
# cast them to typed pointers, which this backend cannot select between, so
# this port launches one plain copy kernel per input tensor (rule 2b) with
# identical results. The uniform-shape fast path is subsumed by the general
# path.
#
# Deviations from upstream, each pinned to the torch.block_diag oracle
# (probed on torch 2.15.0+cpu):
# - upstream unpacked a single list/tuple argument; aten rejects it with
#   TypeError("expected Tensor as element 0 in argument 0, but got list"),
#   so the unpacking is gone and elements are validated.
# - an input with ndim > 2 hit a bare ``assert``; aten raises RuntimeError
#   ("torch.block_diag: Input tensors must have 2 or fewer dimensions.
#   Input {i} has {n} dimensions").
# - the empty call returned ``torch.tensor([])`` (shape (0,)); aten returns
#   a shape-(1, 0) float32 tensor.
# - bool tensors are copied through a uint8 view (bit-identical), matching
#   the repo's convention for 1-byte Triton storage.

_BLOCK_SIZE = 1024


@triton.jit(do_not_specialize=["row_off", "col_off", "rows", "cols", "total_cols"])
def _block_diag_copy_kernel(
    out_ptr,
    src_ptr,
    row_off,
    col_off,
    rows,
    cols,
    total_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """Copy one contiguous (rows, cols) block to its diagonal slot in ``out``."""
    tile = tl.program_id(0)
    offs = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < rows * cols

    r = offs // cols
    c = offs % cols
    out_idx = (row_off + r) * total_cols + (col_off + c)

    val = tl.load(src_ptr + offs, mask=mask)
    tl.store(out_ptr + out_idx, val, mask=mask)


def block_diag(*tensors):
    """Build a block-diagonal matrix from 0-D/1-D/2-D inputs."""
    for i, t in enumerate(tensors):
        if not isinstance(t, torch.Tensor):
            raise TypeError(
                f"expected Tensor as element {i} in argument 0, "
                f"but got {type(t).__name__}"
            )
    if len(tensors) == 0:
        return torch.empty((1, 0), dtype=torch.float32)
    for i, t in enumerate(tensors):
        if t.ndim > 2:
            raise RuntimeError(
                f"torch.block_diag: Input tensors must have 2 or fewer "
                f"dimensions. Input {i} has {t.ndim} dimensions"
            )

    # 0-D -> (1, 1), 1-D -> (1, N), 2-D kept
    tensors_2d = []
    for t in tensors:
        if t.ndim == 0:
            tensors_2d.append(t.unsqueeze(0).unsqueeze(0))
        elif t.ndim == 1:
            tensors_2d.append(t.unsqueeze(0))
        else:
            tensors_2d.append(t)

    total_rows = sum(t.shape[0] for t in tensors_2d)
    total_cols = sum(t.shape[1] for t in tensors_2d)

    out_dtype = tensors_2d[0].dtype
    for t in tensors_2d[1:]:
        out_dtype = torch.result_type(
            torch.empty(0, dtype=out_dtype), torch.empty(0, dtype=t.dtype)
        )
    device = tensors_2d[0].device

    out = torch.zeros((total_rows, total_cols), dtype=out_dtype, device=device)

    cur_row = 0
    cur_col = 0
    for t in tensors_2d:
        rows, cols = t.shape
        numel = rows * cols
        if numel > 0:
            src = (
                t
                if (t.is_contiguous() and t.dtype == out_dtype)
                else t.contiguous().to(out_dtype)
            )
            if src.dtype == torch.bool:
                src = src.view(torch.uint8)
                dst = out.view(torch.uint8)
            else:
                dst = out
            grid = (triton.cdiv(numel, _BLOCK_SIZE),)
            _block_diag_copy_kernel[grid](
                dst,
                src,
                cur_row,
                cur_col,
                rows,
                cols,
                total_cols,
                BLOCK_SIZE=_BLOCK_SIZE,
            )
        cur_row += rows
        cur_col += cols

    return out
