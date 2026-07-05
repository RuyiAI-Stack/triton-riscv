import torch

import triton
import triton.language as tl
import benchmark
from triton.backends.triton_shared.driver import CPUDriver


triton.runtime.driver.set_active(CPUDriver())


@triton.jit
def swap_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(x_ptr + offsets, y, mask=mask)
    tl.store(y_ptr + offsets, x, mask=mask)


def run_swap(x, y):
    assert x.ndim == 1
    assert y.ndim == 1
    assert x.numel() == y.numel()
    n_elements = x.numel()

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    swap_kernel[grid](x, y, n_elements, BLOCK_SIZE=1024)
    return x, y


def bench_swap(size):
    x = torch.rand(size, device="cpu", dtype=torch.float32)
    y = torch.rand(size, device="cpu", dtype=torch.float32)
    benchmark.compare_providers(
        f"bench_swap(size={size})",
        {
            "torch": lambda: (y.clone(), x.clone()),
            "triton-riscv": lambda: run_swap(x.clone(), y.clone()),
        },
    )


if __name__ == "__main__":
    benchmark.select_cpu_backend()
    for size in [2**10 * 16, 2**12 * 16, 2**14 * 16]:
        bench_swap(size)
