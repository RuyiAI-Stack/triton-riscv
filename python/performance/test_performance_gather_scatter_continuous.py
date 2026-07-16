import torch

import triton
import triton.language as tl
import benchmark
from triton.backends.triton_shared.driver import CPUDriver


triton.runtime.driver.set_active(CPUDriver())


@triton.jit
def gather_scatter_continuous_kernel(
    input_ptr,
    output_ptr,
    X: tl.constexpr,
    Y: tl.constexpr,
    Z: tl.constexpr,
):
    # The decomposed x/y/z expression is contiguous by construction. Preserve
    # the gather/scatter indexing while lowering it to one structured CPU loop.
    for xy in tl.range(0, X * Y):
        x = xy // Y
        y = xy % Y
        for z in tl.range(0, Z):
            offset = x * (Y * Z) + y * Z + z
            tl.store(output_ptr + offset, tl.load(input_ptr + offset))


def run_gather_scatter_continuous(input_tensor, x, y, z):
    if input_tensor.numel() != x * y * z:
        raise ValueError("input_tensor.numel() must equal x * y * z")
    if not input_tensor.is_contiguous():
        raise ValueError("input_tensor must be contiguous")
    output = torch.empty_like(input_tensor)
    gather_scatter_continuous_kernel[(1,)](
        input_tensor,
        output,
        X=x,
        Y=y,
        Z=z,
    )
    return output


def bench_gather_scatter_continuous(x, y, z):
    input_tensor = torch.arange(x * y * z, device="cpu", dtype=torch.int32)
    benchmark.compare_providers(
        f"bench_gather_scatter_continuous(x={x}, y={y}, z={z})",
        {
            "torch": lambda: input_tensor.clone(),
            "triton-riscv": lambda: run_gather_scatter_continuous(
                input_tensor, x, y, z
            ),
        },
    )


if __name__ == "__main__":
    benchmark.select_cpu_backend()
    for x, y, z in [(256, 256, 2), (256, 512, 2), (512, 512, 2)]:
        bench_gather_scatter_continuous(x, y, z)
