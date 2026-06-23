import torch

import triton
import triton.language as tl


"""

|-----|-----|-----|-----|
|     |     |     |     |
|-----|-----|-----|-----|
|     |     |     |     |
|-----|-----|-----|-----|

Each instance loads the entire column
"""


@triton.jit
def kernel(
    x_ptr,
    y_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE_ROW: tl.constexpr,
    BLOCK_SIZE_COL: tl.constexpr,
):
    pid0 = tl.program_id(axis=0)
    input_desc = tl.make_tensor_descriptor(
        base=x_ptr,
        shape=[n_cols, n_rows],
        strides=[BLOCK_SIZE_ROW, 1],
        block_shape=[1, BLOCK_SIZE_ROW],
    )
    offsets = [pid0, 0]
    x = input_desc.load(offsets)
    output_desc = tl.make_tensor_descriptor(
        base=y_ptr,
        shape=[n_cols, n_rows],
        strides=[BLOCK_SIZE_ROW, 1],
        block_shape=[1, BLOCK_SIZE_ROW],
    )
    output_desc.store(offsets, x)


def test(device):
    n_rows = 4
    n_cols = 2
    x = (
        torch.arange(0, n_rows * n_cols, 1, device=device, dtype=torch.float32)
        .reshape([n_cols, n_rows])
        .T
    )
    output = torch.empty_strided(
        [n_rows, n_cols], [1, n_rows], device=device, dtype=x.dtype
    )
    output.fill_(-1)
    BLOCK_SIZE_ROW = n_rows
    BLOCK_SIZE_COL = n_cols

    def grid(meta):
        return (n_cols,)

    kernel[grid](
        x,
        output,
        n_rows,
        n_cols,
        BLOCK_SIZE_ROW=BLOCK_SIZE_ROW,
        BLOCK_SIZE_COL=BLOCK_SIZE_COL,
    )

    torch.testing.assert_close(output, x, rtol=0.001, atol=1e-5)
