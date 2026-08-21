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
    N_ROWS: tl.constexpr,
    N_COLS: tl.constexpr,
):
    for row in tl.range(0, N_ROWS):
        for col in tl.range(0, N_COLS):
            wrapped_col = (row + col) % N_COLS
            value = tl.load(input_ptr + row * N_COLS + wrapped_col)
            tl.store(output_ptr + row * N_COLS + col, value)


def run_modulo(x):
    assert x.ndim == 2
    assert x.shape[1] > 0
    output = torch.empty_like(x)
    modulo_kernel[(1,)](
        x,
        output,
        N_ROWS=x.shape[0],
        N_COLS=x.shape[1],
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
