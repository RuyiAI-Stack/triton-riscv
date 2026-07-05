import torch

import triton
import triton.language as tl
import benchmark
from triton.backends.triton_shared.driver import CPUDriver


triton.runtime.driver.set_active(CPUDriver())


@triton.jit
def load_2d_tensor_col_kernel(
    input_ptr,
    output_ptr,
    rows,
    cols,
    input_stride_row,
    input_stride_col,
    output_stride_row,
    output_stride_col,
    BLOCK_ROWS: tl.constexpr,
):
    pid_col = tl.program_id(axis=0)

    input_block = tl.make_block_ptr(
        base=input_ptr,
        shape=(rows, cols),
        strides=(input_stride_row, input_stride_col),
        offsets=(0, pid_col),
        block_shape=(BLOCK_ROWS, 1),
        order=(1, 0),
    )
    values = tl.load(input_block, boundary_check=(0, 1))

    output_block = tl.make_block_ptr(
        base=output_ptr,
        shape=(rows, cols),
        strides=(output_stride_row, output_stride_col),
        offsets=(0, pid_col),
        block_shape=(BLOCK_ROWS, 1),
        order=(1, 0),
    )
    tl.store(output_block, values, boundary_check=(0, 1))


def run_load_2d_tensor_col(rows, cols):
    input_tensor = torch.arange(rows * cols, device="cpu", dtype=torch.float32).reshape(
        rows, cols
    )
    output = torch.empty_like(input_tensor)
    load_2d_tensor_col_kernel[(cols,)](
        input_tensor,
        output,
        rows,
        cols,
        input_tensor.stride(0),
        input_tensor.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_ROWS=rows,
    )
    return output


def bench_load_2d_tensor_col(rows, cols):
    input_tensor = torch.arange(rows * cols, device="cpu", dtype=torch.float32).reshape(
        rows, cols
    )
    benchmark.compare_providers(
        f"bench_load_2d_tensor_col(rows={rows}, cols={cols})",
        {
            "torch": lambda: input_tensor.clone(),
            "triton-riscv": lambda: run_load_2d_tensor_col(rows, cols),
        },
    )


if __name__ == "__main__":
    benchmark.select_cpu_backend()
    bench_load_2d_tensor_col(8 * 32, 4 * 32)
