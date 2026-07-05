import os

import torch

import triton
import triton.language as tl
import benchmark

os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton-cache")
benchmark.select_cpu_backend()


@triton.jit
def splat_kernel(
    fill_value,
    output_ptr,
    stride_row,
    stride_col,
    BLOCK_SIZE_COL: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    values = tl.full((2, BLOCK_SIZE_COL), fill_value, dtype=tl.float32)
    offsets_row = 2 * pid + tl.arange(0, 2)
    offsets_col = tl.arange(0, BLOCK_SIZE_COL)
    output_ptrs = (
        output_ptr
        + offsets_row[:, None] * stride_row
        + offsets_col[None, :] * stride_col
    )
    tl.store(output_ptrs, values)


def run_splat(rows, cols, fill_value=123.456):
    assert rows > 0
    assert cols > 0
    assert rows % 2 == 0
    assert cols & (cols - 1) == 0
    output = torch.empty((rows, cols), device="cpu", dtype=torch.float32)
    splat_kernel[(rows // 2,)](
        fill_value,
        output,
        output.stride(0),
        output.stride(1),
        BLOCK_SIZE_COL=cols,
    )
    return output


def bench_splat(size):
    fill_value = 123.456
    benchmark.compare_providers(
        f"bench_splat(size={size})",
        {
            "torch": lambda: torch.full(
                (size, size), fill_value, device="cpu", dtype=torch.float32
            ),
            "triton-riscv": lambda: run_splat(size, size, fill_value),
        },
    )


if __name__ == "__main__":
    benchmark.select_cpu_backend()
    for size in [32 * 64, 64 * 64, 128 * 64]:
        bench_splat(size)
