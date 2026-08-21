import torch

import triton
import triton.language as tl
import benchmark
from triton.backends.triton_shared.driver import CPUDriver, prepare_cpu_kernel


triton.runtime.driver.set_active(CPUDriver())
_prepared_kernels = {}


@triton.jit
def swap_out_kernel(
    x_ptr,
    y_ptr,
    output_x_ptr,
    output_y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_x_ptr + offsets, y, mask=mask)
    tl.store(output_y_ptr + offsets, x, mask=mask)


def run_swap_out(x, y):
    assert x.ndim == 1
    assert y.ndim == 1
    assert x.numel() == y.numel()
    n_elements = x.numel()
    output_x = torch.empty_like(x)
    output_y = torch.empty_like(y)

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    key = (n_elements, x.dtype, y.dtype)
    runner = _prepared_kernels.get(key)
    if runner is None:
        runner = prepare_cpu_kernel(
            swap_out_kernel,
            grid({"BLOCK_SIZE": 2048}),
            x,
            y,
            output_x,
            output_y,
            n_elements,
            BLOCK_SIZE=2048,
        )
        _prepared_kernels[key] = runner
    runner(x, y, output_x, output_y, n_elements)
    return output_x, output_y


@triton.jit
def swap_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(x_ptr + offsets, y, mask=mask)
    tl.store(y_ptr + offsets, x, mask=mask)


def run_swap(x, y):
    """Compatibility entry point preserving the original in-place API."""
    if x.ndim != 1 or y.ndim != 1 or x.numel() != y.numel():
        raise ValueError("x and y must be one-dimensional tensors of equal length")
    n_elements = x.numel()
    swap_kernel[(triton.cdiv(n_elements, 2048),)](x, y, n_elements, BLOCK_SIZE=2048)
    return x, y


def bench_swap(size):
    x = torch.rand(size, device="cpu", dtype=torch.float32)
    y = torch.rand(size, device="cpu", dtype=torch.float32)
    benchmark.compare_providers(
        f"bench_swap(size={size})",
        {
            "torch": lambda: (y.clone(), x.clone()),
            "triton-riscv": lambda: run_swap_out(x, y),
        },
    )


if __name__ == "__main__":
    benchmark.select_cpu_backend()
    for size in [2**10 * 16, 2**12 * 16, 2**14 * 16]:
        bench_swap(size)
