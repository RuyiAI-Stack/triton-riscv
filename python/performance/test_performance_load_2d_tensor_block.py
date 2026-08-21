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
    N_ELEMENTS: tl.constexpr,
):
    # A scalar structured loop maps better to CPU vectorization than a Triton
    # tensor load here: the latter stages every tile through two temporary
    # buffers before and after the elementwise expression.
    for offset in tl.range(0, N_ELEMENTS):
        value = tl.load(input_ptr + offset)
        tl.store(output_ptr + offset, value * 2.0 + 1.0)


def run_load_2d_tensor_block(rows, cols):
    input_tensor = torch.arange(rows * cols, device="cpu", dtype=torch.float32).reshape(
        rows, cols
    )
    output = torch.empty_like(input_tensor)
    n_elements = input_tensor.numel()
    load_2d_tensor_block_kernel[(1,)](
        input_tensor,
        output,
        N_ELEMENTS=n_elements,
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
