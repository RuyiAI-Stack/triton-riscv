import torch

import triton
import triton.language as tl
import benchmark
from triton.backends.triton_shared.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())


@triton.jit
def blockptr_complex_offset_kernel(
    input_ptr,
    output_ptr,
    rows,
    cols,
    output_rows,
    output_cols,
    input_stride_row,
    input_stride_col,
    output_stride_row,
    output_stride_col,
    row_offset,
    col_offset,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    pid_row = tl.program_id(axis=0)
    pid_col = tl.program_id(axis=1)
    tile_row = pid_row * BLOCK_ROWS
    tile_col = pid_col * BLOCK_COLS

    input_desc = tl.make_tensor_descriptor(
        base=input_ptr,
        shape=(rows, cols),
        strides=(input_stride_row, input_stride_col),
        block_shape=(BLOCK_ROWS, BLOCK_COLS),
    )
    values = input_desc.load((tile_row + row_offset, tile_col + col_offset))
    values = values * 2.0 + 1.0

    output_desc = tl.make_tensor_descriptor(
        base=output_ptr,
        shape=(output_rows, output_cols),
        strides=(output_stride_row, output_stride_col),
        block_shape=(BLOCK_ROWS, BLOCK_COLS),
    )
    output_desc.store((tile_row, tile_col), values)


def run_blockptr_complex_offset(rows, cols):
    input_tensor = torch.arange(rows * cols, device="cpu", dtype=torch.float32).reshape(
        rows, cols
    )
    output = torch.empty((rows - 8, cols - 8), device="cpu", dtype=torch.float32)
    block_rows = 8
    block_cols = 8
    row_offset = 4
    col_offset = 4
    grid = (
        triton.cdiv(output.shape[0], block_rows),
        triton.cdiv(output.shape[1], block_cols),
    )
    blockptr_complex_offset_kernel[grid](
        input_tensor,
        output,
        input_tensor.shape[0],
        input_tensor.shape[1],
        output.shape[0],
        output.shape[1],
        input_tensor.stride(0),
        input_tensor.stride(1),
        output.stride(0),
        output.stride(1),
        row_offset,
        col_offset,
        BLOCK_ROWS=block_rows,
        BLOCK_COLS=block_cols,
    )
    return output


def bench_blockptr_complex_offset(rows, cols):
    input_tensor = torch.arange(rows * cols, device="cpu", dtype=torch.float32).reshape(
        rows, cols
    )
    benchmark.compare_providers(
        f"bench_blockptr_complex_offset(rows={rows}, cols={cols})",
        {
            "torch": lambda: input_tensor[4 : rows - 4, 4 : cols - 4] * 2.0 + 1.0,
            "triton-riscv": lambda: run_blockptr_complex_offset(rows, cols),
        },
    )


if __name__ == "__main__":
    benchmark.select_cpu_backend()
    for rows, cols in [(1600, 1600), (3200, 3200), (6400, 6400)]:
        bench_blockptr_complex_offset(rows, cols)
