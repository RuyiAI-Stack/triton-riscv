import torch
import triton
import triton.language as tl


# is_same_size: aten::is_same_size -- whether two tensors have identical
# shapes. Rank mismatches are settled on the host (the shape vectors would
# not even be the same length); when ranks agree, the per-dimension
# comparison runs in a Triton kernel over the two shape vectors, reducing
# to a single flag with an and-reduction (tl.min over 0/1).


@triton.jit
def is_same_size_kernel(
    self_shape_ptr,
    other_shape_ptr,
    out_ptr,
    ndim,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < ndim

    # Out-of-range lanes load 0 on both sides so they compare equal and stay
    # neutral under the `and` reduction below.
    self_dims = tl.load(self_shape_ptr + offsets, mask=mask, other=0)
    other_dims = tl.load(other_shape_ptr + offsets, mask=mask, other=0)

    same = tl.min((self_dims == other_dims).to(tl.int32), axis=0)
    tl.store(out_ptr, same)


def is_same_size(self: torch.Tensor, other: torch.Tensor) -> bool:
    """Whether ``self`` and ``other`` have identical shapes (rank compared on host)."""
    self_shape = tuple(self.shape)
    other_shape = tuple(other.shape)
    if len(self_shape) != len(other_shape):
        return False
    ndim = len(self_shape)
    if ndim == 0:
        return True
    self_shape_t = torch.tensor(self_shape, dtype=torch.int64)
    other_shape_t = torch.tensor(other_shape, dtype=torch.int64)
    out = torch.empty((), dtype=torch.int32)
    # One block covers the whole shape vector: tensor ranks are small.
    BLOCK_SIZE = triton.next_power_of_2(ndim)
    is_same_size_kernel[(1,)](
        self_shape_t,
        other_shape_t,
        out,
        ndim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return bool(out.item())
