import torch
import triton
import triton.language as tl

import benchmark
from triton.backends.triton_shared.driver import CPUDriver, prepare_cpu_kernel


triton.runtime.driver.set_active(CPUDriver())
_prepared_kernels = {}


@triton.jit
def sum_cpu_kernel(
    input_ptr,
    output_ptr,
    N_ELEMENTS: tl.constexpr,
):
    total = 0.0
    for offset in tl.range(0, N_ELEMENTS):
        total += tl.load(input_ptr + offset)
    tl.store(output_ptr, total)


def tensor_sum(x):
    if x.ndim != 1 or not x.is_contiguous() or x.dtype != torch.float32:
        raise ValueError("sum CPU fast path expects contiguous 1D float32")
    output = torch.empty((), device="cpu", dtype=torch.float32)
    size = x.numel()
    key = (size, x.dtype)
    runner = _prepared_kernels.get(key)
    if runner is None:
        runner = prepare_cpu_kernel(
            sum_cpu_kernel,
            (1,),
            x,
            output,
            N_ELEMENTS=size,
            allow_fp_reassoc=True,
        )
        _prepared_kernels[key] = runner
    runner(x, output)
    return output


def bench_sum(size):
    torch.manual_seed(0)
    x = torch.rand((size,), device="cpu", dtype=torch.float32)
    benchmark.compare_providers(
        f"bench_sum(size={size})",
        {
            "torch": lambda: torch.sum(x),
            "triton-riscv": lambda: tensor_sum(x),
        },
        rtol=1e-4,
        atol=1e-2,
    )


if __name__ == "__main__":
    benchmark.select_cpu_backend()
    for size in [2**18, 2**20, 2**22]:
        bench_sum(size)
