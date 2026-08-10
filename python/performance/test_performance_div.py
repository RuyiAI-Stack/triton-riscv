import torch
import triton
import triton.language as tl

import benchmark
from triton.backends.triton_shared.driver import CPUDriver, prepare_cpu_kernel


triton.runtime.driver.set_active(CPUDriver())
_prepared_kernels = {}


@triton.jit
def div_cpu_kernel(
    numerator_ptr,
    denominator_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    numerator = tl.load(numerator_ptr + offsets, mask=mask)
    denominator = tl.load(denominator_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, numerator / denominator, mask=mask)


def tensor_div(numerator, denominator):
    if (
        numerator.ndim != 1
        or denominator.shape != numerator.shape
        or not numerator.is_contiguous()
        or not denominator.is_contiguous()
        or numerator.dtype != torch.float32
        or denominator.dtype != torch.float32
    ):
        raise ValueError(
            "div CPU fast path expects equal contiguous 1D float32 tensors"
        )
    output = torch.empty_like(numerator)
    size = numerator.numel()
    key = (size, numerator.dtype, denominator.dtype)
    runner = _prepared_kernels.get(key)
    if runner is None:
        runner = prepare_cpu_kernel(
            div_cpu_kernel,
            (triton.cdiv(size, 1024),),
            numerator,
            denominator,
            output,
            size,
            BLOCK_SIZE=1024,
        )
        _prepared_kernels[key] = runner
    runner(numerator, denominator, output, size)
    return output


def bench_div(size):
    torch.manual_seed(0)
    numerator = torch.randn((size,), device="cpu", dtype=torch.float32)
    denominator = torch.rand((size,), device="cpu", dtype=torch.float32) + 0.5
    benchmark.compare_providers(
        f"bench_div(size={size})",
        {
            "torch": lambda: torch.div(numerator, denominator),
            "triton-riscv": lambda: tensor_div(numerator, denominator),
        },
        rtol=1e-5,
        atol=1e-6,
    )


if __name__ == "__main__":
    benchmark.select_cpu_backend()
    for size in [2**18, 2**20, 2**22]:
        bench_div(size)
