import torch

import triton
import triton.language as tl
import benchmark
from triton.backends.triton_shared.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())


@triton.jit
def addptr_kernel(input_ptr, output_ptr, n_elements):
    pid = tl.program_id(0)
    block_start = pid * 2
    in1 = input_ptr + block_start
    in2 = in1 + 1
    out1 = output_ptr + block_start
    out2 = out1 + 1
    mask1 = block_start < n_elements
    mask2 = block_start + 1 < n_elements
    tl.store(out1, tl.load(in1, mask=mask1, other=0.0), mask=mask1)
    tl.store(out2, tl.load(in2, mask=mask2, other=0.0), mask=mask2)


def run_addptr(x):
    assert x.ndim == 1
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = (triton.cdiv(n_elements, 2),)
    addptr_kernel[grid](x, output, n_elements)
    return output


def bench_addptr(size):
    x = torch.arange(size, device="cpu", dtype=torch.float32)
    benchmark.compare_providers(
        f"bench_addptr(size={size})",
        {
            "torch": lambda: x.clone(),
            "triton-riscv": lambda: run_addptr(x),
        },
    )


if __name__ == "__main__":
    benchmark.select_cpu_backend()
    for size in [1024000, 2048000, 4096000, 8192000]:
        bench_addptr(size)
