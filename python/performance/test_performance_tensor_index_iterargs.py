import torch

import triton
import triton.language as tl
import benchmark
from triton.backends.triton_shared.driver import CPUDriver


triton.runtime.driver.set_active(CPUDriver())


@triton.jit
def tensor_index_iterargs_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    STEPS: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE * STEPS + tl.arange(0, BLOCK_SIZE)
    output_offsets = offsets
    for step in range(0, STEPS):
        load_mask = offsets < n_elements
        store_mask = output_offsets < n_elements
        values = tl.load(input_ptr + offsets, mask=load_mask, other=-1)
        tl.store(output_ptr + output_offsets, values, mask=store_mask)
        offsets += BLOCK_SIZE
        output_offsets += BLOCK_SIZE


def run_tensor_index_iterargs(size):
    input_tensor = torch.arange(size, device="cpu", dtype=torch.int32)
    output = torch.full((size,), -1, device="cpu", dtype=torch.int32)
    block_size = 8
    steps = 4
    grid = (triton.cdiv(size, block_size * steps),)
    tensor_index_iterargs_kernel[grid](
        input_tensor,
        output,
        size,
        BLOCK_SIZE=block_size,
        STEPS=steps,
    )
    return output


def bench_tensor_index_iterargs(size):
    benchmark.compare_providers(
        f"bench_tensor_index_iterargs(size={size})",
        {
            "torch": lambda: torch.arange(size, device="cpu", dtype=torch.int32),
            "triton-riscv": lambda: run_tensor_index_iterargs(size),
        },
    )


if __name__ == "__main__":
    benchmark.select_cpu_backend()
    for size in [16 * 256, 32 * 256, 64 * 256]:
        bench_tensor_index_iterargs(size)
