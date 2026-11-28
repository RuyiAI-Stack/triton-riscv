import triton
import triton.language as tl

# fill_diagonal_: aten::fill_diagonal_ -- fill the main diagonal in place.
# The diagonal stride is the sum of the tensor's strides; with wrap=True on
# a 2-d input whose height exceeds width+1, the diagonal restarts every
# (width+1) rows (block-diagonal wrap), with a storage bounds check.


@triton.jit
def fill_diagonal_kernel(
    inp_ptr,
    fill_value,
    main_size,
    wrap_size,
    wrap_offset,
    diagonal_stride,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < main_size + wrap_size
    storage_offsets = tl.where(
        offsets < main_size,
        offsets * diagonal_stride,
        wrap_offset + (offsets - main_size) * diagonal_stride,
    )
    tl.store(inp_ptr + storage_offsets, fill_value, mask=mask)


def fill_diagonal_(self, fill_value, wrap=False):
    """Fill ``self``'s diagonal in place (wrap restarts it every width+1 rows)."""
    n_dims = self.ndim
    if n_dims < 2:
        raise RuntimeError("dimensions must larger than 1")
    height, width = self.shape[:2]
    if n_dims > 2 and any(dim != height for dim in self.shape[1:]):
        raise RuntimeError("all dimensions of input must be of equal length")
    diagonal_stride = sum(self.stride())
    main_size = min(height, width)
    wrap_size = 0
    wrap_offset = 0
    if wrap and n_dims == 2 and height > width + 1:
        logical_step = width + 1
        wrap_size = (self.numel() + logical_step - 1) // logical_step - main_size
        wrap_offset = self.stride(0) * logical_step
        last_storage_offset = (
            self.storage_offset() + wrap_offset + (wrap_size - 1) * diagonal_stride
        )
        storage_size = self.untyped_storage().nbytes() // self.element_size()
        if wrap_size > 0 and last_storage_offset >= storage_size:
            raise RuntimeError(
                "setStorage: wrapped diagonal is out of bounds for storage"
            )
    count = main_size + wrap_size
    if count > 0:
        fill_diagonal_kernel[(triton.cdiv(count, 1024),)](
            self,
            fill_value,
            main_size,
            wrap_size,
            wrap_offset,
            diagonal_stride,
            BLOCK_SIZE=1024,
        )
    return self
