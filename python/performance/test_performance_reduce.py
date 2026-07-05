import torch

import triton
import triton.language as tl
import benchmark
from triton.backends.triton_shared.driver import CPUDriver


triton.runtime.driver.set_active(CPUDriver())


@triton.jit
def reduce_kernel(input_ptr, output_ptr, stride_row, n_cols, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    values = tl.load(input_ptr + pid * stride_row + offsets, mask=mask, other=0.0)
    total = tl.sum(values, axis=0)
    tl.store(output_ptr + pid, total)


def run_reduce(input_tensor):
    rows, cols = input_tensor.shape
    output = torch.empty((rows,), device="cpu", dtype=torch.float32)
    block_size = triton.next_power_of_2(cols)
    reduce_kernel[(rows,)](
        input_tensor,
        output,
        input_tensor.stride(0),
        cols,
        BLOCK_SIZE=block_size,
    )
    return output


def bench_reduce(rows, cols):
    input_tensor = torch.arange(rows * cols, device="cpu", dtype=torch.float32).reshape(
        rows, cols
    )
    benchmark.compare_providers(
        f"bench_reduce(rows={rows}, cols={cols})",
        {
            "torch": lambda: torch.sum(input_tensor, dim=1),
            "triton-riscv": lambda: run_reduce(input_tensor),
        },
    )


if __name__ == "__main__":
    benchmark.select_cpu_backend()
    for rows, cols in [(16 * 16, 16 * 16), (32 * 16, 32 * 16), (64 * 16, 64 * 16)]:
        bench_reduce(rows, cols)
