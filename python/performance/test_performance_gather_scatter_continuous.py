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
    offsets_xy = tl.arange(0, X * Y)
    offsets_z = tl.arange(0, Z)

    offsets_x = offsets_xy[:, None] // Y
    offsets_y = offsets_xy[:, None] % Y
    offsets_z = offsets_z[None, :]

    linear_offsets = offsets_x * (Y * Z) + offsets_y * Z + offsets_z
    values = tl.load(input_ptr + linear_offsets)
    tl.store(output_ptr + linear_offsets, values)


def run_gather_scatter_continuous(x, y, z):
    input_tensor = torch.arange(x * y * z, device="cpu", dtype=torch.int32)
    output = torch.empty((x * y * z,), device="cpu", dtype=torch.int32)
    gather_scatter_continuous_kernel[(1,)](
        input_tensor,
        output,
        X=x,
        Y=y,
        Z=z,
    )
    return output


def bench_gather_scatter_continuous(x, y, z):
    benchmark.compare_providers(
        f"bench_gather_scatter_continuous(x={x}, y={y}, z={z})",
        {
            "torch": lambda: torch.arange(x * y * z, device="cpu", dtype=torch.int32),
            "triton-riscv": lambda: run_gather_scatter_continuous(x, y, z),
        },
    )


if __name__ == "__main__":
    benchmark.select_cpu_backend()
    for x, y, z in [(256, 256, 2), (256, 512, 2), (512, 512, 2)]:
        bench_gather_scatter_continuous(x, y, z)
