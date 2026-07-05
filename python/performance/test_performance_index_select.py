import torch

import triton
import triton.language as tl
import benchmark
from triton.backends.triton_shared.driver import CPUDriver


triton.runtime.driver.set_active(CPUDriver())


@triton.jit
def index_select_row_kernel(
    input_ptr,
    output_ptr,
    indices_ptr,
    stride_i,
    stride_m,
    stride_n,
    output_stride_m,
    output_stride_n,
    BLOCK_I: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_offsets = tl.arange(0, BLOCK_I)
    row_indices = tl.load(indices_ptr + row_offsets * stride_i)
    col_offsets = tl.arange(0, BLOCK_N)
    input_ptrs = (
        input_ptr + row_indices[:, None] * stride_m + col_offsets[None, :] * stride_n
    )
    values = tl.load(input_ptrs)
    output_ptrs = (
        output_ptr
        + row_offsets[:, None] * output_stride_m
        + col_offsets[None, :] * output_stride_n
    )
    tl.store(output_ptrs, values)


def run_index_select(rows, cols, picks):
    input_tensor = torch.arange(rows * cols, device="cpu", dtype=torch.float32).reshape(
        rows, cols
    )
    if picks <= 1:
        indices = torch.zeros((picks,), device="cpu", dtype=torch.int32)
    else:
        step = max((rows - 1) // (picks - 1), 1)
        indices = torch.arange(picks, device="cpu", dtype=torch.int32) * step
    output = torch.empty((picks, cols), device="cpu", dtype=torch.float32)
    index_select_row_kernel[(1,)](
        input_tensor,
        output,
        indices,
        indices.stride(0),
        input_tensor.stride(0),
        input_tensor.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_I=picks,
        BLOCK_N=cols,
    )
    return output


def bench_index_select(rows, cols, picks):
    input_tensor = torch.arange(rows * cols, device="cpu", dtype=torch.float32).reshape(
        rows, cols
    )
    if picks <= 1:
        indices = torch.zeros((picks,), device="cpu", dtype=torch.int64)
    else:
        step = max((rows - 1) // (picks - 1), 1)
        indices = (torch.arange(picks, device="cpu") * step).to(torch.int64)
    benchmark.compare_providers(
        f"bench_index_select(rows={rows}, cols={cols}, picks={picks})",
        {
            "torch": lambda: input_tensor.index_select(0, indices),
            "triton-riscv": lambda: run_index_select(rows, cols, picks),
        },
    )


if __name__ == "__main__":
    benchmark.select_cpu_backend()
    for rows, cols, picks in [
        (8 * 32, 4 * 32, 4 * 32),
        (16 * 32, 8 * 32, 8 * 32),
        (32 * 32, 16 * 32, 16 * 32),
    ]:
        bench_index_select(rows, cols, picks)
