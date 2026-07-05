import torch

import triton
import triton.language as tl
import benchmark
from triton.backends.triton_shared.driver import CPUDriver


triton.runtime.driver.set_active(CPUDriver())


@triton.jit
def early_return_kernel(input_ptr, output_ptr, n_elements):
    pid = tl.program_id(0)
    if pid >= n_elements:
        return
    value = tl.load(input_ptr + pid)
    if value == -1:
        return
    tl.store(output_ptr + pid, value + 1)


def run_early_return(x):
    assert x.ndim == 1
    output = torch.full((x.numel(),), -1, device=x.device, dtype=x.dtype)
    early_return_kernel[(x.numel(),)](x, output, x.numel())
    return output


def bench_early_return(size):
    x = torch.arange(size, device="cpu", dtype=torch.int32)
    benchmark.compare_providers(
        f"bench_early_return(size={size})",
        {
            "torch": lambda: torch.where(x == -1, x, x + 1),
            "triton-riscv": lambda: run_early_return(x),
        },
    )


if __name__ == "__main__":
    benchmark.select_cpu_backend()
    for size in [2**10 * 100, 2**12 * 100, 2**14 * 100]:
        bench_early_return(size)
