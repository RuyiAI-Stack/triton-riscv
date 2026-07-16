import torch

import triton
import triton.language as tl
import benchmark
from triton.backends.triton_shared.driver import CPUDriver


triton.runtime.driver.set_active(CPUDriver())


@triton.jit
def mask_loop(
    y_ptr,
    x_ptr,
    scale_ptr,
    SIZE: tl.constexpr,
):
    scale = tl.load(scale_ptr)
    for idx in tl.range(0, SIZE):
        x = tl.load(x_ptr + idx).to(tl.float32)
        tl.store(y_ptr + idx, x * scale)


def run_mask_loop_iter_arg(x, scale, rows, cols):
    y = torch.empty_like(x)
    mask_loop[(1,)](
        y,
        x,
        scale,
        SIZE=x.numel(),
    )
    return y.reshape(rows, cols)


def bench_mask_loop_iter_arg(rows, cols):
    x = torch.arange(rows * cols, device="cpu", dtype=torch.float32)
    scale = torch.ones((1,), device="cpu", dtype=torch.float32)
    benchmark.compare_providers(
        f"bench_mask_loop_iter_arg(rows={rows}, cols={cols})",
        {
            "torch": lambda: (x * scale).reshape(rows, cols),
            "triton-riscv": lambda: run_mask_loop_iter_arg(x, scale, rows, cols),
        },
    )


if __name__ == "__main__":
    benchmark.select_cpu_backend()
    for rows, cols in [
        (3 * 32 * 16, 5 * 32 * 16),
        (8 * 32 * 16, 16 * 32 * 16),
        (16 * 32 * 16, 32 * 32 * 16),
    ]:
        bench_mask_loop_iter_arg(rows, cols)
