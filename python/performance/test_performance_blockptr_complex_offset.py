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
    INPUT_STRIDE_ROW: tl.constexpr,
    OUTPUT_STRIDE_ROW: tl.constexpr,
    OUTPUT_ROWS: tl.constexpr,
    OUTPUT_COLS: tl.constexpr,
    ROW_OFFSET: tl.constexpr,
    COL_OFFSET: tl.constexpr,
):
    # Tiny 8x8 tensor-descriptor tiles create hundreds of thousands of CPU
    # program calls and staging buffers. Express the same offset submatrix as
    # a structured row loop with a contiguous inner dimension.
    for row in tl.range(0, OUTPUT_ROWS):
        input_base = (row + ROW_OFFSET) * INPUT_STRIDE_ROW + COL_OFFSET
        output_base = row * OUTPUT_STRIDE_ROW
        for col in tl.range(0, OUTPUT_COLS):
            value = tl.load(input_ptr + input_base + col)
            tl.store(output_ptr + output_base + col, value * 2.0 + 1.0)


def run_blockptr_complex_offset(input_tensor):
    rows, cols = input_tensor.shape
    output = torch.empty((rows - 8, cols - 8), device="cpu", dtype=torch.float32)
    row_offset = 4
    col_offset = 4
    blockptr_complex_offset_kernel[(1,)](
        input_tensor,
        output,
        INPUT_STRIDE_ROW=input_tensor.stride(0),
        OUTPUT_STRIDE_ROW=output.stride(0),
        OUTPUT_ROWS=output.shape[0],
        OUTPUT_COLS=output.shape[1],
        ROW_OFFSET=row_offset,
        COL_OFFSET=col_offset,
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
            "triton-riscv": lambda: run_blockptr_complex_offset(input_tensor),
        },
    )


if __name__ == "__main__":
    benchmark.select_cpu_backend()
    for rows, cols in [(1600, 1600), (3200, 3200), (6400, 6400)]:
        bench_blockptr_complex_offset(rows, cols)
