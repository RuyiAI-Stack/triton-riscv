import torch

import triton
import triton.language as tl
import benchmark
from triton.backends.triton_shared.driver import CPUDriver


triton.runtime.driver.set_active(CPUDriver())


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)


def run_vec_add(x, y):
    assert x.ndim == 1
    assert y.ndim == 1
    assert x.shape == y.shape
    assert x.numel() == y.numel()
    output = torch.empty_like(x)
    n_elements = output.numel()

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


def bench_vec_add(size):
    x = torch.rand(size, device="cpu", dtype=torch.float32)
    y = torch.rand(size, device="cpu", dtype=torch.float32)
    benchmark.compare_providers(
        f"bench_vec_add(size={size})",
        {
            "torch": lambda: x + y,
            "triton-riscv": lambda: run_vec_add(x, y),
        },
    )


if __name__ == "__main__":
    benchmark.select_cpu_backend()
    for size in [2**20, 2**22, 2**24]:
        bench_vec_add(size)
