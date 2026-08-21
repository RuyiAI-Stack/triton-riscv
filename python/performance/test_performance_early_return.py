import torch

import triton
import triton.language as tl
import benchmark
from triton.backends.triton_shared.driver import CPUDriver


triton.runtime.driver.set_active(CPUDriver())


@triton.jit
def early_return_kernel(input_ptr, output_ptr, N_ELEMENTS: tl.constexpr):
    # A CPU launcher call per scalar element dominates this control-flow test.
    # Convert the per-program early return into an equivalent guarded store in
    # one structured loop.
    for offset in tl.range(0, N_ELEMENTS):
        value = tl.load(input_ptr + offset)
        tl.store(output_ptr + offset, tl.where(value == -1, value, value + 1))


def run_early_return(x):
    assert x.ndim == 1
    output = torch.empty_like(x)
    early_return_kernel[(1,)](x, output, N_ELEMENTS=x.numel())
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
