import torch

import triton
import triton.language as tl
import benchmark
from triton.backends.triton_shared.driver import CPUDriver


triton.runtime.driver.set_active(CPUDriver())


@triton.jit
def load_2d_tensor_block_kernel(
    input_ptr,
    output_ptr,
    rows,
    cols,
    input_stride_row,
    input_stride_col,
    output_stride_row,
    output_stride_col,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    pid_row = tl.program_id(axis=0)
    pid_col = tl.program_id(axis=1)

    input_desc = tl.make_tensor_descriptor(
        base=input_ptr,
        shape=(rows, cols),
        strides=(input_stride_row, input_stride_col),
        block_shape=(BLOCK_ROWS, BLOCK_COLS),
    )
    offsets = (pid_row * BLOCK_ROWS, pid_col * BLOCK_COLS)
    values = input_desc.load(offsets)
    values = values * 2.0 + 1.0

    output_desc = tl.make_tensor_descriptor(
        base=output_ptr,
        shape=(rows, cols),
        strides=(output_stride_row, output_stride_col),
        block_shape=(BLOCK_ROWS, BLOCK_COLS),
    )
    output_desc.store(offsets, values)


def run_load_2d_tensor_block(rows, cols):
    input_tensor = torch.arange(rows * cols, device="cpu", dtype=torch.float32).reshape(
        rows, cols
    )
    output = torch.empty_like(input_tensor)
    block_rows = 4
    block_cols = 4
    grid = (triton.cdiv(rows, block_rows), triton.cdiv(cols, block_cols))
    load_2d_tensor_block_kernel[grid](
        input_tensor,
        output,
        rows,
        cols,
        input_tensor.stride(0),
        input_tensor.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_ROWS=block_rows,
        BLOCK_COLS=block_cols,
    )
    return output


def bench_load_2d_tensor_block(rows, cols):
    input_tensor = torch.arange(rows * cols, device="cpu", dtype=torch.float32).reshape(
        rows, cols
    )
    benchmark.compare_providers(
        f"bench_load_2d_tensor_block(rows={rows}, cols={cols})",
        {
            "torch": lambda: input_tensor * 2.0 + 1.0,
            "triton-riscv": lambda: run_load_2d_tensor_block(rows, cols),
        },
    )


if __name__ == "__main__":
    benchmark.select_cpu_backend()
    for rows, cols in [(16 * 32, 16 * 32), (32 * 32, 32 * 32), (64 * 32, 32 * 32)]:
        bench_load_2d_tensor_block(rows, cols)
