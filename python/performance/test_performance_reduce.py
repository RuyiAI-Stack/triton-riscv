import torch

import triton
import triton.language as tl
import benchmark
from triton.backends.triton_shared.driver import CPUDriver, prepare_cpu_kernel


triton.runtime.driver.set_active(CPUDriver())
_prepared_kernels = {}


@triton.jit
def reduce_kernel(
    input_ptr,
    output_ptr,
    STRIDE_ROW: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    # Amortize CPU launcher overhead across several rows while retaining
    # Triton's tree reduction within each row.
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    cols = tl.arange(0, BLOCK_COLS)
    values = tl.load(input_ptr + rows[:, None] * STRIDE_ROW + cols[None, :])
    totals = tl.sum(values, axis=1)
    tl.store(output_ptr + rows, totals)


@triton.jit
def reduce_tail_kernel(
    input_ptr,
    output_ptr,
    STRIDE_ROW: tl.constexpr,
    N_ROWS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    cols = tl.arange(0, BLOCK_COLS)
    row_mask = rows < N_ROWS
    values = tl.load(
        input_ptr + rows[:, None] * STRIDE_ROW + cols[None, :],
        mask=row_mask[:, None],
        other=0.0,
    )
    totals = tl.sum(values, axis=1)
    tl.store(output_ptr + rows, totals, mask=row_mask)


def run_reduce(input_tensor):
    rows, cols = input_tensor.shape
    output = torch.empty((rows,), device="cpu", dtype=torch.float32)
    block_rows = 8
    key = (rows, cols, input_tensor.stride(0), input_tensor.dtype)
    runner = _prepared_kernels.get(key)
    if runner is None:
        kernel = reduce_kernel if rows % block_rows == 0 else reduce_tail_kernel
        grid = (
            (rows // block_rows,)
            if rows % block_rows == 0
            else (triton.cdiv(rows, block_rows),)
        )
        kernel_kwargs = {}
        if rows % block_rows:
            kernel_kwargs["N_ROWS"] = rows
        runner = prepare_cpu_kernel(
            kernel,
            grid,
            input_tensor,
            output,
            STRIDE_ROW=input_tensor.stride(0),
            BLOCK_ROWS=block_rows,
            BLOCK_COLS=cols,
            allow_fp_reassoc=True,
            **kernel_kwargs,
        )
        _prepared_kernels[key] = runner
    runner(input_tensor, output)
    return output


def bench_reduce(rows, cols):
    input_tensor = torch.arange(rows * cols, device="cpu", dtype=torch.float32).reshape(
        rows, cols
    )
    benchmark.compare_providers(
        f"bench_reduce(rows={rows}, cols={cols})",
        {
            "torch": lambda: torch.sum(input_tensor, dim=1),
            "triton-riscv": lambda: run_reduce(input_tensor),
        },
    )


if __name__ == "__main__":
    benchmark.select_cpu_backend()
    for rows, cols in [(16 * 16, 16 * 16), (32 * 16, 32 * 16), (64 * 16, 64 * 16)]:
        bench_reduce(rows, cols)
