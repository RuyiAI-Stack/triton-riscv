import torch
import torch.nn.functional as F
import triton
import triton.language as tl

import benchmark
from triton.backends.triton_shared.driver import CPUDriver, prepare_cpu_kernel


triton.runtime.driver.set_active(CPUDriver())
_prepared_kernels = {}


@triton.jit
def silu_cpu_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    value = tl.load(input_ptr + offsets, mask=mask)
    result = value / (1.0 + tl.exp(-value))
    tl.store(output_ptr + offsets, result, mask=mask)


def silu(x):
    if x.ndim != 1 or not x.is_contiguous() or x.dtype != torch.float32:
        raise ValueError("SiLU CPU fast path expects contiguous 1D float32")
    output = torch.empty_like(x)
    size = x.numel()
    key = (size, x.dtype)
    runner = _prepared_kernels.get(key)
    if runner is None:
        runner = prepare_cpu_kernel(
            silu_cpu_kernel,
            (triton.cdiv(size, 1024),),
            x,
            output,
            size,
            BLOCK_SIZE=1024,
        )
        _prepared_kernels[key] = runner
    runner(x, output, size)
    return output


def bench_silu(size):
    torch.manual_seed(0)
    x = torch.randn((size,), device="cpu", dtype=torch.float32)
    benchmark.compare_providers(
        f"bench_silu(size={size})",
        {
            "torch": lambda: F.silu(x),
            "triton-riscv": lambda: silu(x),
        },
        rtol=1e-4,
        atol=1e-4,
    )


if __name__ == "__main__":
    benchmark.select_cpu_backend()
    for size in [2**18, 2**20, 2**22]:
        bench_silu(size)
