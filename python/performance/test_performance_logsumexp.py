import torch
import triton
import triton.language as tl

import benchmark
from triton.backends.triton_shared.driver import CPUDriver, prepare_cpu_kernel


triton.runtime.driver.set_active(CPUDriver())
_prepared_kernels = {}


@triton.jit
def logsumexp_cpu_kernel(
    output_ptr,
    input_ptr,
    INPUT_ROW_STRIDE: tl.constexpr,
    N_ROWS: tl.constexpr,
    N_COLS: tl.constexpr,
):
    # Keep the stable max/exp/sum formulation, but express it as two
    # structured contiguous passes. This avoids the generic implementation's
    # full-row tensor temporaries and amortizes CPU launch overhead over all
    # rows in one program.
    for row in tl.range(0, N_ROWS):
        row_base = row * INPUT_ROW_STRIDE
        maximum = -float("inf")
        for col in tl.range(0, N_COLS):
            maximum = tl.maximum(maximum, tl.load(input_ptr + row_base + col))

        exp_sum = 0.0
        for col in tl.range(0, N_COLS):
            value = tl.load(input_ptr + row_base + col)
            exp_sum += tl.exp(value - maximum)

        result = maximum + tl.log(exp_sum)
        is_infinite = (maximum == -float("inf")) | (maximum == float("inf"))
        result = tl.where(is_infinite, maximum, result)
        tl.store(output_ptr + row, result)


def logsumexp(x):
    if x.ndim != 2 or not x.is_contiguous():
        raise ValueError("logsumexp CPU fast path expects a contiguous 2D tensor")
    rows, cols = x.shape
    if rows == 0 or cols == 0:
        raise ValueError("logsumexp CPU fast path expects non-empty dimensions")
    if x.dtype != torch.float32:
        raise ValueError("logsumexp CPU fast path currently expects float32")
    output = torch.empty((rows,), device="cpu", dtype=x.dtype)
    key = (rows, cols, x.stride(0), x.dtype)
    runner = _prepared_kernels.get(key)
    if runner is None:
        runner = prepare_cpu_kernel(
            logsumexp_cpu_kernel,
            (1,),
            output,
            x,
            INPUT_ROW_STRIDE=x.stride(0),
            N_ROWS=rows,
            N_COLS=cols,
            allow_fp_reassoc=True,
        )
        _prepared_kernels[key] = runner
    runner(output, x)
    return output


def bench_logsumexp(rows, cols):
    torch.manual_seed(0)
    x = torch.randn((rows, cols), device="cpu", dtype=torch.float32)

    benchmark.compare_providers(
        f"bench_logsumexp(rows={rows}, cols={cols})",
        {
            "torch": lambda: torch.logsumexp(x, dim=1),
            "triton-riscv": lambda: logsumexp(x),
        },
        rtol=1e-3,
        atol=1e-3,
    )


if __name__ == "__main__":
    benchmark.select_cpu_backend()
    for rows, cols in [(64, 512), (128, 1024), (256, 2048)]:
        bench_logsumexp(rows, cols)
