"""Standalone Triton source for dequantize_197x768."""

import triton
import triton.language as tl


@triton.jit
def dequantize(
    X,
    Row,
    Column,
    Out,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Convert AME INT32 accumulators back to FP32 using row/column scales."""
    # Every program stays in one row: full rows are multiples of BLOCK,
    # and the classifier has only one row. Preserve the existing launch ABI.
    tl.static_assert(COLS % BLOCK == 0 or ROWS == 1)
    start = tl.program_id(0) * BLOCK
    index = start + tl.arange(0, BLOCK)
    valid = index < ROWS * COLS
    if ROWS == 1:
        row_index = 0
        column_index = index
        column = tl.load(Column + column_index, valid, other=1.0)
    else:
        row_index = start // COLS
        column_index = start % COLS + tl.arange(0, BLOCK)
        # COLS is a multiple of BLOCK, so no program crosses a row.
        column = tl.load(Column + column_index)
    value = tl.load(X + index, valid, other=0).to(tl.float32)
    row = tl.load(Row + row_index)
    tl.store(Out + index, (value * row) * column, valid)
