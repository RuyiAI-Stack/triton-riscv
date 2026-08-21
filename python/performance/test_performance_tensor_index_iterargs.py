import torch

import triton
import triton.language as tl
import benchmark
from triton.backends.triton_shared.driver import CPUDriver, prepare_cpu_kernel


triton.runtime.driver.set_active(CPUDriver())
_prepared_kernels = {}


@triton.jit
def tensor_index_iterargs_kernel(
    input_ptr,
    output_ptr,
    N_ELEMENTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    STEPS: tl.constexpr,
):
    # Keep the iter-arg stepping semantics but avoid a tiny 8-element program
    # for every 32 values on CPU.
    for base in tl.range(0, N_ELEMENTS, BLOCK_SIZE * STEPS):
        for step in tl.range(0, STEPS):
            step_base = base + step * BLOCK_SIZE
            for lane in tl.range(0, BLOCK_SIZE):
                offset = step_base + lane
                tl.store(output_ptr + offset, tl.load(input_ptr + offset))


def run_tensor_index_iterargs(input_tensor):
    size = input_tensor.numel()
    output = torch.empty_like(input_tensor)
    block_size = 8
    steps = 4
    assert size % (block_size * steps) == 0
    runner = _prepared_kernels.get(size)
    if runner is None:
        runner = prepare_cpu_kernel(
            tensor_index_iterargs_kernel,
            (1,),
            input_tensor,
            output,
            N_ELEMENTS=size,
            BLOCK_SIZE=block_size,
            STEPS=steps,
        )
        _prepared_kernels[size] = runner
    runner(input_tensor, output)
    return output


def bench_tensor_index_iterargs(size):
    input_tensor = torch.arange(size, device="cpu", dtype=torch.int32)
    benchmark.compare_providers(
        f"bench_tensor_index_iterargs(size={size})",
        {
            "torch": lambda: input_tensor.clone(),
            "triton-riscv": lambda: run_tensor_index_iterargs(input_tensor),
        },
    )


if __name__ == "__main__":
    benchmark.select_cpu_backend()
    for size in [16 * 256, 32 * 256, 64 * 256]:
        bench_tensor_index_iterargs(size)
