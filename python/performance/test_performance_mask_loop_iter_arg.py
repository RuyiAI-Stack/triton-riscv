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
    size,
    BLOCK_SIZE: tl.constexpr,
):
    bidx = tl.program_id(0)
    tidx = tl.arange(0, BLOCK_SIZE)
    grid_stride = tl.num_programs(0) * BLOCK_SIZE
    iterations = tl.cdiv(size, grid_stride)
    idx = bidx * BLOCK_SIZE + tidx
    scale = tl.load(scale_ptr)
    for it in range(iterations):
        mask = idx < size
        x = tl.load(x_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        tl.store(y_ptr + idx, x * scale, mask=mask)
        idx += grid_stride


def run_mask_loop_iter_arg(rows, cols):
    x = torch.arange(rows * cols, device="cpu", dtype=torch.float32)
    y = torch.empty_like(x)
    scale_ones = torch.ones((1,), device="cpu", dtype=torch.float32)
    block_size = 8
    grid = (2,)
    mask_loop[grid](
        y,
        x,
        scale_ones,
        x.numel(),
        BLOCK_SIZE=block_size,
    )
    return y.reshape(rows, cols)


def bench_mask_loop_iter_arg(rows, cols):
    benchmark.compare_providers(
        f"bench_mask_loop_iter_arg(rows={rows}, cols={cols})",
        {
            "torch": lambda: torch.arange(
                rows * cols, device="cpu", dtype=torch.float32
            ).reshape(rows, cols),
            "triton-riscv": lambda: run_mask_loop_iter_arg(rows, cols),
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
