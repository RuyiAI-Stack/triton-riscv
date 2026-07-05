import torch

import triton
import triton.language as tl
import benchmark
from triton.backends.triton_shared.driver import CPUDriver


triton.runtime.driver.set_active(CPUDriver())


@triton.jit
def modulo_kernel(
    input_ptr,
    output_ptr,
    n_rows,
    n_cols,
    input_stride_row,
    input_stride_col,
    output_stride_row,
    output_stride_col,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= n_rows:
        return
    cols = tl.arange(0, BLOCK_SIZE)
    wrapped_cols = (pid + cols) % n_cols
    input_ptrs = input_ptr + pid * input_stride_row + wrapped_cols * input_stride_col
    output_ptrs = output_ptr + pid * output_stride_row + cols * output_stride_col
    mask = cols < n_cols
    values = tl.load(input_ptrs, mask=mask, other=0.0)
    tl.store(output_ptrs, values, mask=mask)


def run_modulo(x):
    assert x.ndim == 2
    assert x.shape[1] > 0
    output = torch.empty_like(x)
    block_size = triton.next_power_of_2(x.shape[1])
    modulo_kernel[(x.shape[0],)](
        x,
        output,
        x.shape[0],
        x.shape[1],
        x.stride(0),
        x.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_SIZE=block_size,
    )
    return output


def bench_modulo(size):
    x = torch.arange(size * size, device="cpu", dtype=torch.float32).reshape(size, size)
    cols = torch.arange(size, device="cpu")
    wrapped_cols = (torch.arange(size, device="cpu")[:, None] + cols[None, :]) % size
    benchmark.compare_providers(
        f"bench_modulo(size={size})",
        {
            "torch": lambda: torch.gather(x, 1, wrapped_cols),
            "triton-riscv": lambda: run_modulo(x),
        },
    )


if __name__ == "__main__":
    benchmark.select_cpu_backend()
    for size in [64 * 32, 128 * 32, 256 * 32]:
        bench_modulo(size)
