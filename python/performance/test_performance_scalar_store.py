import torch

import triton
import triton.language as tl
import benchmark
from triton.backends.triton_shared.driver import CPUDriver


triton.runtime.driver.set_active(CPUDriver())


@triton.jit
def scalar_store_kernel(output_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    base_ptr = output_ptr + pid * BLOCK_SIZE
    inner_extent = 2
    for i in range(0, BLOCK_SIZE // 2):
        for j in range(0, inner_extent):
            index = i * inner_extent + j
            tl.store(base_ptr + index, index)


def run_scalar_store(size):
    assert size > 0
    assert size % 2 == 0
    output = torch.empty((size,), device="cpu", dtype=torch.float32)
    scalar_store_kernel[(1,)](output, BLOCK_SIZE=size)
    return output


def bench_scalar_store(size):
    benchmark.compare_providers(
        f"bench_scalar_store(size={size})",
        {
            "torch": lambda: torch.arange(size, device="cpu", dtype=torch.float32),
            "triton-riscv": lambda: run_scalar_store(size),
        },
    )


if __name__ == "__main__":
    benchmark.select_cpu_backend()
    for size in [8 * 256, 16 * 256, 32 * 256]:
        bench_scalar_store(size)
