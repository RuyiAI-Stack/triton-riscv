import torch

import triton
import triton.language as tl
import benchmark
from triton.backends.triton_shared.driver import CPUDriver


triton.runtime.driver.set_active(CPUDriver())


@triton.jit
def gather_scatter_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = tl.arange(0, BLOCK_SIZE)
    output_offsets = pid * BLOCK_SIZE + offsets
    gather_offsets = pid * BLOCK_SIZE + offsets // 10 + 5
    load_mask = gather_offsets < n_elements
    store_mask = output_offsets < n_elements
    values = tl.load(input_ptr + gather_offsets, mask=load_mask, other=0)
    tl.store(output_ptr + output_offsets, values, mask=store_mask)


def run_gather_scatter(size):
    input_tensor = torch.arange(size, device="cpu", dtype=torch.int32)
    output = torch.full((size,), -1, device="cpu", dtype=torch.int32)
    block_size = 64
    grid = (triton.cdiv(size, block_size),)
    gather_scatter_kernel[grid](
        input_tensor,
        output,
        size,
        BLOCK_SIZE=block_size,
    )
    return output


def bench_gather_scatter(size):
    def torch_reference():
        input_tensor = torch.arange(size, device="cpu", dtype=torch.int32)
        output_offsets = torch.arange(size, device="cpu")
        gather_offsets = (output_offsets // 64) * 64 + (output_offsets % 64) // 10 + 5
        return torch.where(
            gather_offsets < size,
            input_tensor[gather_offsets.clamp(max=size - 1)],
            torch.zeros_like(output_offsets, dtype=torch.int32),
        )

    benchmark.compare_providers(
        f"bench_gather_scatter(size={size})",
        {
            "torch": torch_reference,
            "triton-riscv": lambda: run_gather_scatter(size),
        },
    )


if __name__ == "__main__":
    benchmark.select_cpu_backend()
    for size in [640, 1280, 2560]:
        bench_gather_scatter(size)
